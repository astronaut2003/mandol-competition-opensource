"""Content-free diagnostic context usable by domain code without HTTP imports."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import math
import re
import threading

from pydantic import ValidationError

_current = ContextVar('mandol_diagnostic_context', default=None)
_FIELDS = frozenset({'body', 'query', 'options', 'user_id', 'top_k', 'messages',
    'role', 'content', 'timestamp', 'request_id', 'session_id', 'data', 'id',
    'score', 'created_at', 'response', 'text', 'type', 'image_url', 'url'})
_STAGES = frozenset({'request', 'request_validation', 'add', 'add_receipt', 'search', 'scope',
    'query_input', 'embedding', 'retrieval', 'qdrant_dense', 'qdrant_sparse',
    'bm25', 'candidate_hydration', 'candidate_fusion', 'graph', 'reranker',
    'result_construction', 'evidence_budget', 'response', 'dependency'})


@dataclass
class DiagnosticContext:
    correlation_id: str
    operation: str
    track: str = 'textual'
    reported: set[int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)


def bind_context(context):
    return _current.set(context)


def reset_context(token):
    _current.reset(token)


def current_context():
    return _current.get()


def _identifier(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.<>-]{0,119}', value) else '[redacted]'


def validation_details(errors):
    """Pydantic input/msg/ctx/URL and caller-controlled field names never leave here."""
    return [dict(
        loc=[part if (type(part) is int or isinstance(part,str) and part in _FIELDS) else '[field]'
             for part in error.get('loc', ())],
        type=_identifier(error.get('type', 'unknown')))
        for error in errors[:128]]


def _exception_metadata(exc, errors=None):
    row = dict(exception_type=_identifier(type(exc).__name__))
    # These classes come from our transport, never provider messages or input.
    transport_class = getattr(exc, 'error_class', None)
    if isinstance(transport_class, str) and transport_class in {'ConnectError', 'ConnectTimeout', 'PoolTimeout',
            'ReadError', 'ReadTimeout', 'WriteError', 'WriteTimeout',
            'RemoteProtocolError', 'LocalProtocolError', 'CloseError',
            'deadline', 'deadline_before_dispatch', 'provider_rate_limited',
            'vision_recovery_cooldown', 'vision_recovery_exhausted',
            'vision_output_truncated', 'vision_output_invalid_json',
            'vision_output_finish_unverified', 'vision_output_refused', 'vision_output_ocr_repetition'}:
        row['inference_error_class'] = transport_class
    finish = getattr(exc, 'vlm_finish_reason', None)
    if isinstance(finish, str) and finish in {'stop','length','content_filter','tool_calls','function_call','unverified'}:
        row['vlm_finish_reason'] = finish
    for key in ('vlm_completion_tokens', 'vlm_output_token_budget'):
        value = getattr(exc, key, None)
        if type(value) is int and 0 <= value <= 1000000:
            row[key] = value
    unknown = getattr(exc, 'unknown_outcome', None)
    if type(unknown) is bool:
        row['unknown_outcome'] = unknown
    status = getattr(exc, 'status_code', None)
    if type(status) is int and 100 <= status <= 599:
        row['upstream_http_status'] = status
    delay = getattr(exc, 'retry_after', None)
    if type(delay) in (int, float) and math.isfinite(delay) and delay >= 0:
        row['retry_after_seconds'] = delay
    for name in ('provider_error_code', 'provider_request_id'):
        value = getattr(exc, name, None)
        if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', value):
            row[name] = value
    if errors is not None or isinstance(exc, ValidationError):
        row['validation'] = validation_details(exc.errors() if errors is None else errors)
    else:
        frames, tb = [], exc.__traceback__
        while tb is not None:
            frames.append(dict(file=_identifier(Path(tb.tb_frame.f_code.co_filename).name),
                function=_identifier(tb.tb_frame.f_code.co_name), line=tb.tb_lineno))
            tb = tb.tb_next
        row['stack'] = frames[-40:]
    return row


def record_failure(exc, *, stage='request', context=None, validation_errors=None):
    context = context or current_context()
    if context is None:
        return None
    with context.lock:
        if id(exc) in context.reported:
            return context.correlation_id
        context.reported.add(id(exc))
    row = dict(timestamp=datetime.now(timezone.utc).isoformat(), event=context.track+'_failure',
        track=context.track, operation=context.operation, correlation_id=context.correlation_id,
        stage=stage if stage in _STAGES else 'request', **_exception_metadata(exc,validation_errors))
    if 'validation' not in row:
        cause, seen, chain = exc.__cause__, {id(exc)}, []
        while cause is not None and id(cause) not in seen and len(chain)<4:
            seen.add(id(cause)); chain.append(_exception_metadata(cause)); cause=cause.__cause__
        if chain: row['causes']=chain
    logging.getLogger('mandol_competition.'+context.track+'.diagnostics').error(json.dumps(row, ensure_ascii=False, allow_nan=False))
    return context.correlation_id


def configure_diagnostics(track='textual'):
    assert track in ('textual','code','multimodal')
    _logger=logging.getLogger('mandol_competition.'+track+'.diagnostics')
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(handler)
    _logger.propagate = False
    _logger.setLevel(logging.INFO)


@contextmanager
def diagnostic_stage(stage):
    try:
        yield
    except Exception as exc:
        record_failure(exc, stage=stage)
        raise


def diagnostic_call(stage, fn, *args, **kwargs):
    with diagnostic_stage(stage):
        return fn(*args, **kwargs)
