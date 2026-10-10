"""Opt-in MM request spans. No texts, keys, images or user identifiers are logged."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps
import json
import hashlib
from pathlib import Path
import threading
import time
import uuid

_active: ContextVar[dict | None] = ContextVar("mandol_mm_profile", default=None)
_write_lock = threading.Lock()


@contextmanager
def span(name):
    record = _active.get()
    if record is None:
        yield
        return
    started = time.perf_counter()
    status = 'ok'
    try:
        yield
    except BaseException:
        status = 'error'
        raise
    finally:
        record['spans'].append({'stage': name, 'start': started-record['start'],
                               'seconds': time.perf_counter()-started, 'status': status})


def stage(name):
    return lambda function: timed(function, name)


def timed(function, stage, *, candidates=False, fingerprint=False):
    @wraps(function)
    def call(*args, **kwargs):
        record = _active.get()
        if record is None:
            return function(*args, **kwargs)
        start = time.perf_counter()
        status = "ok"
        try:
            result = function(*args, **kwargs)
            if fingerprint and isinstance(result, str):
                record.setdefault("fingerprints", {})[stage] = hashlib.sha256(result.encode()).hexdigest()
            if candidates:
                record["candidates"][stage] = [
                    {"id": hit.logical_uid, "score": float(hit.score)} for hit in result]
            return result
        except BaseException:
            status = "error"
            raise
        finally:
            record["spans"].append({"stage": stage, "start": start - record["start"],
                "seconds": time.perf_counter() - start, "status": status})
    return call


def _wrap(instance, name, stage, **options):
    if instance is not None and hasattr(instance, name):
        setattr(instance, name, timed(getattr(instance, name), stage, **options))


class RequestProfiler:
    def __init__(self, path: Path | None, *, backend, add, search, extractor):
        self.path = path
        self._http_records = {}
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _wrap(add, "_decode", "validation_decode")
        _wrap(add, "prepare_units", "preparation_inclusive")
        _wrap(search, "prepare_query", "query_preparation_inclusive", fingerprint=True)
        _wrap(extractor, "extract", "vlm_remote_inclusive")
        if backend is not None:
            def observe(stage, values):
                if (record := _active.get()) is not None:
                    record.setdefault("retrieval", {})[stage] = values
            backend.retrieval.observer = observe
            _wrap(backend.state, "prepare", "sqlite_t1")
            _wrap(backend.state, "publish", "sqlite_t2")
            _wrap(backend.state, "get_memories", "hydration")
            _wrap(backend.state.lexical, "search", "bm25", candidates=True)
            _wrap(backend.retrieval, "verify_scope", "scope_verification")
            _wrap(backend.retrieval, "search", "retrieval_inclusive")
            for method, stage in (("upsert_projection", "qdrant_upsert"),
                ("get_projection_metadata", "qdrant_readback"), ("audit_rows", "qdrant_audit")):
                _wrap(backend.projection, method, stage)
            _wrap(backend.projection, "search_dense", "dense_candidates", candidates=True)
            native_sparse = backend.profile.options.get('sparse_implementation') == 'provider-native-v4'
            _wrap(backend.projection, "search_splade", "native_sparse_candidates" if native_sparse else "legacy_splade_candidates", candidates=True)
            _wrap(getattr(backend.encoders, "dense", None), "embed", "embedding_remote")
            _wrap(getattr(backend.encoders, "sparse", None), "encode_documents", "splade_remote")
            _wrap(backend.encoders, "encode_documents", "paired_dense_native_sparse_http")
            _wrap(backend.encoders, "encode_query", "paired_dense_native_sparse_http")
            _wrap(backend.reranker, "score", "reranker_remote")

    def call(self, operation, function, body, profile_id):
        if self.path is None:
            return function(body)
        http_record = self._http_records.get(profile_id)
        if http_record is not None:
            http_record["service_start"] = time.perf_counter()
        record = {"profile_id": profile_id, "operation": operation, "start": time.perf_counter(),
                  "spans": [], "candidates": {}}
        token = _active.set(record)
        try:
            value = function(body)
            if operation == "search":
                record["returned_ids"] = [h["id"] for h in value["data"]]
            record["status"] = "ok"
            return value
        except BaseException as exc:
            record["status"] = "error"
            record["error_class"] = type(exc).__name__
            # SDK wrappers can hide a transport/validation cause. Keep only
            # class names, never exception messages, URLs or response bodies.
            causes, seen, current = [], set(), exc
            while isinstance(current, BaseException) and id(current) not in seen and len(causes) < 5:
                seen.add(id(current))
                causes.append(type(current).__name__)
                current = current.__cause__ or getattr(current, "source", None) or current.__context__
            record["error_classes"] = causes
            raise
        finally:
            if http_record is not None:
                http_record["service_end"] = time.perf_counter()
            record["seconds"] = time.perf_counter() - record.pop("start")
            _active.reset(token)
            # Profiling must never change a durable Add outcome if its disk fills.
            try:
                with _write_lock, self.path.open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            except OSError:
                pass

    def response_id(self, response):
        if self.path is None:
            return None
        identifier = uuid.uuid4().hex
        response.headers["X-Mandol-Profile"] = identifier
        if (http_record := _http_active.get()) is not None:
            http_record["profile_id"] = identifier
            self._http_records[identifier] = http_record
        return identifier


# The ASGI layer separates JSON serialization and admission/validation overhead
# from service spans. It never captures request or response content.
_http_active: ContextVar[dict | None] = ContextVar('mandol_mm_http_profile', default=None)
from starlette.responses import JSONResponse


class ProfiledJSONResponse(JSONResponse):
    def render(self, content):
        record = _http_active.get()
        if record is None:
            return super().render(content)
        start = time.perf_counter()
        record['serialization_start'] = start
        try:
            return super().render(content)
        finally:
            record['json_serialization'] = time.perf_counter() - start


class HTTPProfileMiddleware:
    def __init__(self, app, *, profiler):
        self.app, self.profiler = app, profiler

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') not in ('/add', '/search'):
            return await self.app(scope, receive, send)
        record = {'start': time.perf_counter(), 'operation': scope['path'][1:]}
        token = _http_active.set(record)
        async def measured_receive():
            message = await receive()
            if message['type'] == 'http.request' and not message.get('more_body', False):
                record['body_received'] = time.perf_counter()
            return message
        async def measured_send(message):
            if message['type'] == 'http.response.start':
                record['status_code'] = message['status']
            await send(message)
        try:
            await self.app(scope, measured_receive, measured_send)
        finally:
            end = time.perf_counter()
            _http_active.reset(token)
            profile_id = record.get('profile_id')
            self.profiler._http_records.pop(profile_id, None)
            result = {'profile_id': profile_id, 'operation': record['operation'],
                'status_code': record.get('status_code'), 'asgi_total': end - record['start'],
                'json_serialization': record.get('json_serialization')}
            for name, first, last in (
                ('body_receive', 'start', 'body_received'),
                ('validation_auth_dispatch', 'body_received', 'service_start'),
                ('response_validation_dispatch', 'service_end', 'serialization_start')):
                result[name] = max(0., record[last] - record[first]) if first in record and last in record else None
            path = self.profiler.path.with_name(self.profiler.path.stem + '-http.jsonl')
            try:
                with _write_lock, path.open('a') as stream:
                    stream.write(json.dumps(result, allow_nan=False) + '\n')
            except OSError:
                pass
