"""Qualification-only loopback HTTP for the audited BGE and Querit references.

This module is independent of formal serving configuration and application
orchestration. Loading starts only in the explicitly launched app's lifespan.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request

from mandol_substrate.state.models import StateUnavailable
from . import bge_reference, querit_reference
from .reference_contract import pinned_snapshot, verify_files

SCHEMA = "mandol-native-reranker-http-v1"
LOGGER = logging.getLogger(__name__)
BGE_FILES = {
    "config.json": {
        "sha256": "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a"
    },
    "model.safetensors": {
        "sha256": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286"
    },
    "sentencepiece.bpe.model": {
        "sha256": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"
    },
    "special_tokens_map.json": {
        "sha256": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835"
    },
    "tokenizer.json": {
        "sha256": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15"
    },
    "tokenizer_config.json": {
        "sha256": "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4"
    },
}


def fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise TypeError("native profile must be an object")
    name = profile.get("model_name")
    module = {
        bge_reference.MODEL_NAME: bge_reference,
        querit_reference.MODEL_NAME: querit_reference,
    }.get(name)
    if module is None:
        raise ValueError("unsupported qualification native model")
    keys = ["dtype", "device", "max_pair_tokens", "batch_size", "max_batch_tokens"]
    if module is bge_reference:
        keys += ["sort_by", "compile_model"]
    try:
        settings = {key: profile[key] for key in keys}
    except KeyError as error:
        raise ValueError("incomplete frozen native profile") from error
    if module.expected_profile(**settings) != profile:
        raise ValueError("native profile differs from the audited reference contract")
    return module, settings


def native_loader(profile, model_snapshot):
    module, settings = validate_profile(profile)
    snapshot = pinned_snapshot(model_snapshot, module.MODEL_NAME, module.MODEL_REVISION)
    verify_files(
        snapshot, BGE_FILES if module is bge_reference else module.FILE_MANIFEST
    )
    if module is bge_reference:
        settings.pop(
            "sort_by"
        )  # The published .score default is unchanged input order.
        return module.NativeBgeReference.from_pretrained(snapshot, **settings)
    return module.NativeQueritReference.from_pretrained(snapshot, **settings)


def validate_scores(values, count, output_kind):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError("incomplete native score sequence")
    low, high = (0, 1) if output_kind == "probability" else (-1, 1)
    if any(
        type(value) not in (int, float)
        or not low <= value <= high
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("invalid native score type, range, or finiteness")
    return [float(value) for value in values]


class NativeInputError(ValueError):
    """The complete native query/document input is outside its frozen contract."""


def strict_json(data):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("non-finite JSON constant")

    return json.loads(
        data, object_pairs_hook=object_pairs, parse_constant=invalid_constant
    )


async def complete_owned(future):
    """Keep ownership through repeated caller cancellation until work completes."""
    cancelled = False
    while True:
        try:
            return await asyncio.shield(future), cancelled
        except asyncio.CancelledError:
            if future.cancelled():
                raise RuntimeError("owned native task was cancelled directly") from None
            cancelled = True


class TorchWorkerMetrics:
    """Observations from the sidecar's inference thread; no model policy changes."""

    def __init__(self, reference, device):
        import torch

        self.torch, self.device = torch, torch.device(device)
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in reference.model.parameters()
        )

    def snapshot(self, *, reset_peak=False):
        import psutil

        result = {
            "pid": os.getpid(),
            "host_rss_bytes": psutil.Process().memory_info().rss,
            "observed_unix_seconds": time.time(),
            "parameter_bytes": self.parameter_bytes,
            "torch_version": self.torch.__version__,
            "cuda_runtime_version": self.torch.version.cuda,
            "device": str(self.device),
            "scope": "sidecar process PyTorch allocator; device free/total include all processes",
        }
        if self.device.type != "cuda":
            return {**result, "cuda": None}
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        if reset_peak:
            cuda.reset_peak_memory_stats(self.device)
        free, total = cuda.mem_get_info(self.device)
        return {
            **result,
            "cuda": {
                "allocated_bytes": cuda.memory_allocated(self.device),
                "reserved_bytes": cuda.memory_reserved(self.device),
                "peak_allocated_bytes": cuda.max_memory_allocated(self.device),
                "peak_reserved_bytes": cuda.max_memory_reserved(self.device),
                "device_free_bytes": free,
                "device_total_bytes": total,
            },
        }


class NativeScoreWorker:
    """One owned inference thread; queue timeout never overlaps model cleanup."""

    def __init__(
        self,
        profile,
        loader,
        *,
        max_queue=4,
        timeout_seconds=120,
        max_documents=300,
        qualification_metrics=False,
        metrics_factory=TorchWorkerMetrics,
    ):
        validate_profile(profile)
        if type(max_queue) is not int or not 1 <= max_queue <= 64:
            raise ValueError("native pending queue must be in [1,64]")
        if type(max_documents) is not int or not 1 <= max_documents <= 1000:
            raise ValueError("native document bound must be in [1,1000]")
        if type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= 600:
            raise ValueError("native request deadline must be in (0,600]")
        self.profile = copy.deepcopy(profile)
        self.profile_sha256 = fingerprint(profile)
        self.output_kind = profile.get("output_kind", "probability")
        self.loader, self.timeout_seconds, self.max_documents = (
            loader,
            timeout_seconds,
            max_documents,
        )
        self.queue = asyncio.Queue(maxsize=max_queue)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="native-reranker"
        )
        self.model = self.task = None
        self.closing = None
        self.accepting = False
        self.failure = None
        self.qualification_metrics, self.metrics_factory = (
            qualification_metrics,
            metrics_factory,
        )
        self.metrics = None
        self.last_resources = None

    def check_model(self):
        try:
            actual = self.model.profile if self.model is not None else None
        except Exception as error:
            raise StateUnavailable("loaded native profile is unavailable") from error
        if actual != self.profile:
            raise StateUnavailable("loaded native profile changed or is unavailable")

    async def start(self):
        loading = asyncio.get_running_loop().run_in_executor(self.executor, self.loader)
        try:
            self.model, cancelled = await complete_owned(loading)
            if cancelled:
                raise asyncio.CancelledError
            self.check_model()
            if self.qualification_metrics:
                self.metrics = await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    lambda: self.metrics_factory(self.model, self.profile["device"]),
                )
            self.accepting = True
            self.task = asyncio.create_task(self._work())
        except BaseException:
            await self.close()
            raise

    def _score(self, query, documents):
        self.check_model()
        preparation = getattr(self.model, "preparation", None)
        if preparation is None:
            raise StateUnavailable("audited native token preparation is unavailable")
        prepared_audit = []
        try:
            for document in documents:
                pair = preparation.prepare(query, document)
                if pair.query != query:
                    raise StateUnavailable("native preparation changed the query")
                prepared_audit.append(
                    {
                        "pair_tokens": pair.token_count,
                        "input_ids_sha256": fingerprint(list(pair.input_ids)),
                        "original_document_chars": len(document),
                        "scored_document_chars": len(pair.document),
                        "document_was_truncated": pair.document != document,
                        "scored_document_sha256": hashlib.sha256(
                            pair.document.encode()
                        ).hexdigest(),
                    }
                )
        except ValueError as error:
            raise NativeInputError(
                "input exceeds the frozen native token contract"
            ) from error
        resources = {}
        if self.metrics is not None:
            resources["before_inference"] = self.metrics.snapshot(reset_peak=True)
        started = time.perf_counter()
        try:
            scores = validate_scores(
                self.model.score(query, documents), len(documents), self.output_kind
            )
        finally:
            if self.metrics is not None:
                native_seconds = time.perf_counter() - started
                resources["after_inference"] = self.metrics.snapshot()
                resources["native_call_seconds"] = native_seconds
                self.last_resources = resources
        self.check_model()
        return {
            "schema": SCHEMA,
            "model": self.profile["model_name"],
            "profile_sha256": self.profile_sha256,
            "output_kind": self.output_kind,
            "score_semantics": self.profile["score_semantics"],
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "documents_sha256": fingerprint(documents),
            "data": [
                {"index": index, "score": value} for index, value in enumerate(scores)
            ],
            "native_output_audit": {
                **copy.deepcopy(getattr(self.model, "last_metrics", {})),
                "http_preparation": prepared_audit,
                "worker_resources": resources,
            },
            "native_batch_metrics": copy.deepcopy(
                getattr(self.model, "last_batch_metrics", {})
            ),
        }

    async def _work(self):
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                query, documents, future = item
                if future.cancelled():
                    continue
                if self.failure:
                    future.set_exception(
                        StateUnavailable("native worker requires restart")
                    )
                    continue
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        self.executor, self._score, query, documents
                    )
                except NativeInputError:
                    if not future.done():
                        future.set_exception(
                            NativeInputError(
                                "input exceeds the frozen native token contract"
                            )
                        )
                except Exception as error:
                    self.failure, self.accepting = type(error).__name__, False
                    LOGGER.exception("Native qualification inference failed")
                    if not future.done():
                        future.set_exception(
                            StateUnavailable("native inference failed")
                        )
                else:
                    if not future.done():
                        future.set_result(result)
            finally:
                self.queue.task_done()

    async def submit(self, query, documents, *, remaining_seconds=None):
        if not self.accepting:
            raise HTTPException(503, "native worker is unavailable")
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait((query, documents, future))
        except asyncio.QueueFull as error:
            raise HTTPException(
                429, "native inference queue is full", headers={"Retry-After": "1"}
            ) from error
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                self.timeout_seconds
                if remaining_seconds is None
                else remaining_seconds,
            )
        except TimeoutError as error:
            future.cancel()
            raise HTTPException(504, "native inference deadline exceeded") from error
        except asyncio.CancelledError:
            future.cancel()
            raise
        except NativeInputError as error:
            raise HTTPException(
                400, "input exceeds the frozen native token contract"
            ) from error
        except StateUnavailable as error:
            raise HTTPException(503, "native inference failed") from error

    async def close(self):
        self.accepting = False
        if self.closing is None:
            self.closing = asyncio.create_task(self._close_owned())
        _result, cancelled = await complete_owned(self.closing)
        if cancelled:
            raise asyncio.CancelledError

    def resource_snapshot(self):
        self.check_model()
        if self.metrics is None:
            raise StateUnavailable("qualification metrics were not enabled")
        return {
            "schema": SCHEMA,
            "profile_sha256": self.profile_sha256,
            "model": self.profile["model_name"],
            "current": self.metrics.snapshot(),
            "last_inference": copy.deepcopy(self.last_resources),
        }

    async def _close_owned(self):
        if self.task is not None:
            while not self.queue.empty():
                item = self.queue.get_nowait()
                self.queue.task_done()
                if item is not None and not item[2].done():
                    item[2].set_exception(
                        StateUnavailable("native worker is shutting down")
                    )
            await self.queue.put(None)
            # An expired HTTP request can still own an in-flight native call.
            # Wait for that call before releasing its weight module.
            await self.task
            self.task = None
        try:
            if self.model is not None:
                await asyncio.get_running_loop().run_in_executor(
                    self.executor, lambda: self.model.close(release_cuda_cache=True)
                )
        finally:
            self.model = None
            self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(
    profile,
    model_snapshot=None,
    *,
    loader=None,
    max_queue=4,
    timeout_seconds=120,
    max_documents=300,
    max_body_bytes=8 << 20,
    max_http_requests=None,
    qualification_metrics=False,
    metrics_factory=TorchWorkerMetrics,
):
    if type(max_body_bytes) is not int or not 1 <= max_body_bytes <= 32 << 20:
        raise ValueError("native request body bound must be in [1,32MiB]")
    declared = copy.deepcopy(profile)
    max_http_requests = (
        max_queue + 1 if max_http_requests is None else max_http_requests
    )
    if type(max_http_requests) is not int or not 1 <= max_http_requests <= 128:
        raise ValueError("native HTTP admission bound must be in [1,128]")
    admission = {"active": 0, "limit": max_http_requests}
    worker = NativeScoreWorker(
        declared,
        loader or (lambda: native_loader(declared, model_snapshot)),
        max_queue=max_queue,
        timeout_seconds=timeout_seconds,
        max_documents=max_documents,
        qualification_metrics=qualification_metrics,
        metrics_factory=metrics_factory,
    )

    @asynccontextmanager
    async def lifespan(app):
        await worker.start()
        try:
            yield
        finally:
            await worker.close()

    app = FastAPI(lifespan=lifespan)
    app.state.native_worker = worker
    app.state.native_admission = admission
    metrics_gate = asyncio.Lock()

    @app.get("/gpu-profile")
    async def gpu_profile():
        if not qualification_metrics:
            raise HTTPException(404, "qualification metrics are disabled")
        if worker.closing is not None or worker.model is None:
            raise HTTPException(503, "native model is not available")
        if metrics_gate.locked():
            raise HTTPException(429, "native metrics observation is already pending")
        await metrics_gate.acquire()
        future = asyncio.get_running_loop().run_in_executor(
            worker.executor, worker.resource_snapshot
        )
        future.add_done_callback(lambda finished: metrics_gate.release())
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout_seconds)
        except TimeoutError as error:
            raise HTTPException(
                504, "native metrics observation deadline exceeded"
            ) from error
        except StateUnavailable as error:
            raise HTTPException(503, "native metrics are unavailable") from error

    @app.get("/health")
    async def health():
        try:
            worker.check_model()
        except StateUnavailable as error:
            raise HTTPException(503, "native model is not ready") from error
        if not worker.accepting:
            raise HTTPException(503, "native model is not accepting work")
        return {
            "schema": SCHEMA,
            "status": "loaded",
            "model": declared["model_name"],
            "profile": copy.deepcopy(worker.model.profile),
            "profile_sha256": worker.profile_sha256,
            "output_kind": worker.output_kind,
            "pending": worker.queue.qsize(),
            "max_pending": max_queue,
            "max_documents": max_documents,
            "timeout_seconds": timeout_seconds,
            "max_body_bytes": max_body_bytes,
            "active_http_requests": admission["active"],
            "max_http_requests": admission["limit"],
            "qualification_metrics": qualification_metrics,
        }

    async def score_body(request):
        chunks, size = [], 0
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        try:
            async with asyncio.timeout_at(deadline):
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > max_body_bytes:
                        raise HTTPException(413, "native request body is too large")
                    chunks.append(chunk)
        except TimeoutError as error:
            raise HTTPException(408, "native request body deadline exceeded") from error
        try:
            payload = strict_json(b"".join(chunks))
        except (ValueError, UnicodeDecodeError) as error:
            raise HTTPException(400, "invalid JSON request") from error
        if not isinstance(payload, dict) or set(payload) != {
            "model",
            "profile_sha256",
            "query",
            "documents",
        }:
            raise HTTPException(400, "invalid native request fields")
        if (
            payload["model"] != declared["model_name"]
            or payload["profile_sha256"] != worker.profile_sha256
        ):
            raise HTTPException(409, "native model/profile mismatch")
        documents = payload["documents"]
        if (
            not isinstance(payload["query"], str)
            or not isinstance(documents, list)
            or not 1 <= len(documents) <= max_documents
            or any(not isinstance(value, str) for value in documents)
        ):
            raise HTTPException(400, "invalid native query/document sequence")
        try:
            payload["query"].encode("utf-8")
            for document in documents:
                document.encode("utf-8")
        except UnicodeError as error:
            raise HTTPException(
                400, "native input must contain valid Unicode"
            ) from error
        return await worker.submit(
            payload["query"],
            documents,
            remaining_seconds=max(0, deadline - asyncio.get_running_loop().time()),
        )

    @app.post("/score")
    async def score(request: Request):
        # No await between checking and reserving a slot in this event loop.
        # A rejected request never starts reading or retaining its body.
        if admission["active"] >= admission["limit"]:
            raise HTTPException(
                429, "native HTTP admission is full", headers={"Retry-After": "1"}
            )
        admission["active"] += 1
        try:
            return await score_body(request)
        finally:
            admission["active"] -= 1

    return app


class NativeRerankerHttpClient:
    """Independent expected profile; complete, ordered results or StateUnavailable."""

    def __init__(
        self, url, profile, *, timeout_seconds=125, transport=None, async_transport=None
    ):
        validate_profile(profile)
        parsed = httpx.URL(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "model endpoint must be credential-free HTTP(S); use a protected service network"
            )
        self._profile, self.profile_sha256 = (
            copy.deepcopy(profile),
            fingerprint(profile),
        )
        self.client = httpx.Client(
            base_url=url, timeout=timeout_seconds, transport=transport, trust_env=False
        )
        self.async_client = httpx.AsyncClient(
            base_url=url,
            timeout=timeout_seconds,
            transport=async_transport,
            trust_env=False,
        )
        self.last_metrics, self.last_batch_metrics = {}, {}

    @property
    def profile(self):
        return copy.deepcopy(self._profile)

    def _payload(self, query, documents):
        if (
            not isinstance(query, str)
            or not isinstance(documents, (list, tuple))
            or not documents
            or any(not isinstance(value, str) for value in documents)
        ):
            raise ValueError("native scoring requires a query and document sequence")
        query.encode("utf-8")
        for document in documents:
            document.encode("utf-8")
        return {
            "model": self._profile["model_name"],
            "profile_sha256": self.profile_sha256,
            "query": query,
            "documents": list(documents),
        }

    def _parse(self, response, payload):
        try:
            response.raise_for_status()
            body = strict_json(response.content)
            kind = self._profile.get("output_kind", "probability")
            expected = {
                "schema": SCHEMA,
                "model": self._profile["model_name"],
                "profile_sha256": self.profile_sha256,
                "score_semantics": self._profile["score_semantics"],
                "output_kind": kind,
                "query_sha256": hashlib.sha256(payload["query"].encode()).hexdigest(),
                "documents_sha256": fingerprint(payload["documents"]),
            }
            if not isinstance(body, dict) or any(
                body.get(key) != value for key, value in expected.items()
            ):
                raise ValueError("native response identity differs")
            rows = body.get("data")
            count = len(payload["documents"])
            if (
                not isinstance(rows, list)
                or len(rows) != count
                or any(
                    not isinstance(row, dict)
                    or type(row.get("index")) is not int
                    or not 0 <= row["index"] < count
                    or "score" not in row
                    for row in rows
                )
            ):
                raise ValueError("incomplete or malformed native response rows")
            if len({row["index"] for row in rows}) != count:
                raise ValueError("duplicate native response indices")
            values = validate_scores(
                [row["score"] for row in sorted(rows, key=lambda row: row["index"])],
                count,
                kind,
            )
        except (httpx.HTTPError, ValueError, TypeError, OverflowError) as error:
            raise StateUnavailable(
                "invalid or unavailable native score response"
            ) from error
        self.last_metrics = body.get("native_output_audit", {})
        self.last_batch_metrics = body.get("native_batch_metrics", {})
        return values

    def score(self, query, documents):
        self.last_metrics, self.last_batch_metrics = {}, {}
        payload = self._payload(query, documents)
        try:
            response = self.client.post("/score", json=payload)
        except httpx.HTTPError as error:
            raise StateUnavailable("native scoring request failed") from error
        return self._parse(response, payload)

    async def ascore(self, query, documents):
        self.last_metrics, self.last_batch_metrics = {}, {}
        payload = self._payload(query, documents)
        try:
            response = await self.async_client.post("/score", json=payload)
        except httpx.HTTPError as error:
            raise StateUnavailable("native scoring request failed") from error
        return self._parse(response, payload)

    def _health(self, response):
        try:
            response.raise_for_status()
            body = strict_json(response.content)
            expected = {
                "schema": SCHEMA,
                "status": "loaded",
                "model": self._profile["model_name"],
                "profile": self._profile,
                "profile_sha256": self.profile_sha256,
                "output_kind": self._profile.get("output_kind", "probability"),
            }
            if not isinstance(body, dict) or any(
                body.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(
                    "native health profile differs from the frozen declaration"
                )
            return body
        except (httpx.HTTPError, ValueError, TypeError) as error:
            raise StateUnavailable(
                "native model is not ready with the expected profile"
            ) from error

    def health(self):
        try:
            response = self.client.get("/health")
        except httpx.HTTPError as error:
            raise StateUnavailable("native health request failed") from error
        return self._health(response)

    async def ahealth(self):
        try:
            response = await self.async_client.get("/health")
        except httpx.HTTPError as error:
            raise StateUnavailable("native health request failed") from error
        return self._health(response)

    def close(self):
        self.client.close()

    async def aclose(self):
        self.client.close()
        await self.async_client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18103)
    parser.add_argument("--max-pending", type=int, default=4)
    parser.add_argument("--max-documents", type=int, default=300)
    parser.add_argument("--max-http-requests", type=int)
    parser.add_argument("--qualification-metrics", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("qualification port must be in [1024,65535]")
    import uvicorn

    app = create_app(
        json.loads(args.profile.read_text()),
        args.model_snapshot,
        max_queue=args.max_pending,
        max_documents=args.max_documents,
        timeout_seconds=args.timeout_seconds,
        max_http_requests=args.max_http_requests,
        qualification_metrics=args.qualification_metrics,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=app.state.native_admission["limit"] + 8,
    )


if __name__ == "__main__":
    main()
