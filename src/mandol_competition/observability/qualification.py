"""Opt-in smoke spans on the real service; records no request/response content.

Disabled unless MANDOL_QUALIFICATION_TRACE_PATH is explicitly configured.
Wrappers observe return values, never change candidates, scores or exceptions.
"""
from contextvars import ContextVar
from functools import wraps
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import time

_active = ContextVar('mandol_qualification', default=None)
_lock = threading.Lock()


def measured(fn, stage, describe=None, describe_input=None):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        record = _active.get()
        if record is None:
            return fn(*args, **kwargs)
        started = time.perf_counter()
        span = {'stage': stage}
        if describe_input:
            try:
                span.update(describe_input(args, kwargs))
            except Exception as exc:
                span['observation_error_class'] = type(exc).__name__
        try:
            result = fn(*args, **kwargs)
            if describe:
                try:
                    span.update(describe(result))
                except Exception as exc:
                    span['observation_error_class'] = type(exc).__name__
            elif isinstance(result, (list, tuple, dict)):
                span['count'] = len(result)
            return result
        except BaseException as exc:
            span['error_class'] = type(exc).__name__
            raise
        finally:
            span['seconds'] = time.perf_counter() - started
            record['spans'].append(span)
    return wrapped


def _wrap(obj, name, label, describe=None, describe_input=None):
    if obj is not None and hasattr(obj, name):
        setattr(obj, name, measured(getattr(obj, name), label, describe, describe_input))


def install_qualification_trace(app, track):
    path = os.getenv('MANDOL_QUALIFICATION_TRACE_PATH')
    if not path:
        return app
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    memory = app.state.memory_service
    def operation(fn, name):
        @wraps(fn)
        def call(*args, **kwargs):
            started = time.perf_counter()
            record = {'timestamp': datetime.now(timezone.utc).isoformat(),
                      'track': track, 'operation': name, 'spans': []}
            token = _active.set(record)
            try:
                result = fn(*args, **kwargs)
                if name == 'search': record['final_count'] = len(result)
                return result
            except BaseException as exc:
                record['error_class'] = type(exc).__name__
                raise
            finally:
                record['seconds'] = time.perf_counter() - started
                if track == 'code': record['retrieval_path'] = memory._last_retrieval_path
                _active.reset(token)
                try:
                    with _lock, path.open('a') as stream:
                        stream.write(json.dumps(record, allow_nan=False) + '\n')
                except (OSError, TypeError, ValueError):
                    pass
        return call
    memory.add_messages = operation(memory.add_messages, 'add')
    memory.search = operation(memory.search, 'search')
    original_start = memory.start
    def start():
        result = original_start()
        state = memory.state if track == 'textual' else memory.store.state
        _wrap(state.lexical, 'search', 'bm25' if track == 'textual' else 'lexical')
        _wrap(state, 'get_memories', 'hydration')
        if track == 'textual':
            _wrap(memory.projection, 'search_dense', 'dense')
            _wrap(memory.projection, 'search_splade', 'native_sparse' if memory.settings.inference_contract == 'aml-official-eligible-v1' else 'legacy_splade')
            _wrap(memory.encoders, 'encode_documents', 'paired_dense_native_sparse_http')
            _wrap(memory.encoders, 'encode_query', 'paired_dense_native_sparse_http')
            _wrap(memory.dense, 'embed', 'embedding_http')
            _wrap(memory.splade, 'encode_documents', 'splade_http')
            _wrap(memory.reranker, 'rerank', 'rerank',
                  lambda scores: {'output_count': len(scores)},
                  lambda args, kwargs: {'input_count': len(args[1] if len(args) > 1 else kwargs['documents'])})
            def audit(value):
                fused, detail = value
                return {'fusion_count': len(fused), 'legs': {k: len(v) for k, v in detail['legs'].items()}}
            _wrap(memory.retriever, 'retrieve', 'retrieval', audit)
            # The frozen function itself is unchanged; wrap only in this
            # explicitly instrumented process to obtain its own duration.
            from mandol_competition.textual.core import retrieval
            _wrap(retrieval, 'fuse_legs', 'fusion')
        else:
            from mandol_competition.code.projection import QdrantDenseChannel
            from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
            from mandol_competition.code.software.experience_retrieval import EpisodeRetriever, EvidencePacker
            _wrap(memory._encoder, 'encode', 'embedding_http')
            _wrap(memory._encoder, 'encode_query', 'embedding_query_http')
            _wrap(memory, '_serialized_search', 'cache_lock_and_search_inclusive')
            _wrap(memory, '_projection', 'projection_preparation')
            _wrap(QdrantDenseChannel, 'search', 'dense')
            _wrap(EngineeringEpisodeBuilder, 'build', 'episode_construction', lambda _: {})
            def episodes(hits):
                return {'count': len(hits), 'evidence_counts': [len(h.evidence_uids) for h in hits],
                        'features': [h.features for h in hits]}
            _wrap(EpisodeRetriever, 'search', 'episode_retrieval', episodes)
            _wrap(EvidencePacker, 'pack', 'evidence_packing')
        return result
    memory.start = start
    return app
