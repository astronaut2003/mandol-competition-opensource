from collections import Counter
from datetime import datetime, timezone
import json
import ipaddress
import logging
import os
import re
import threading
import time
import uuid

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response


class SafeJSONFormatter(logging.Formatter):
    def format(self, record):
        # Do not format arbitrary third-party log arguments, traceback or bodies.
        # Emit only controlled event identifiers and bounded operational fields.
        event = str(record.msg)
        event = event if re.fullmatch('[a-z_]{1,64}', event) else 'runtime_event'
        result = {'timestamp': datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                  'level': record.levelname, 'event': event, 'logger': record.name}
        for key in ('track', 'operation', 'stage', 'backend', 'status', 'channel', 'recipe',
                    'error_class', 'duration_seconds', 'correlation_id'):
            if hasattr(record, key): result[key] = getattr(record, key)
        if record.exc_info: result['error_class'] = record.exc_info[0].__name__
        return json.dumps(result, ensure_ascii=False, allow_nan=False)


def configure_logging(level='INFO'):
    handler = logging.StreamHandler(); handler.setFormatter(SafeJSONFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ('httpx', 'httpcore', 'uvicorn.access'):
        logging.getLogger(name).setLevel(logging.WARNING)


class Metrics:
    def __init__(self, track):
        self.track = track
        self.lock = threading.Lock()
        self.count, self.duration, self.errors = Counter(), Counter(), Counter()
    def observe(self, operation, duration, status):
        with self.lock:
            self.count[operation] += 1; self.duration[operation] += duration
            if status >= 400: self.errors[operation] += 1
    def render(self, state=None, reranker=None, executor=None, *, optional_projection_channel="dense"):
        rows = []
        with self.lock:
            for op in sorted(self.count):
                labels = f'track="{self.track}",operation="{op}"'
                rows += [f'request_duration_seconds_count{{{labels}}} {self.count[op]}',
                         f'request_duration_seconds_sum{{{labels}}} {self.duration[op]:.9f}',
                         f'request_errors_total{{{labels}}} {self.errors[op]}']
        if state is not None:
            try:
                counts = {
                    'publication_pending': state.pending_count(),
                    'purge_pending': state._rows("SELECT count(*) n FROM scope_purges WHERE status='pending'")[0]['n'],
                    'projection_lag': state._rows('SELECT coalesce(sum(canonical_seq-published_seq),0) n FROM scopes')[0]['n'],
                    'retrieval_degraded_total': state._rows("SELECT coalesce(sum(count),0) n FROM runtime_events WHERE operation='search'")[0]['n'],
                    'reconcile_attempts_total': state._rows('SELECT coalesce(sum(attempts),0) n FROM mutations')[0]['n'],
                }
                counts['purge_pending'] += state._rows("SELECT count(*) n FROM projection_frontiers WHERE error_class='PurgePending'")[0]['n']
                if self.track == 'code':
                    counts['projection_lag'] = state._rows("SELECT coalesce(sum(max(0,s.published_seq-coalesce(f.published_seq,0))),0) n FROM scopes s LEFT JOIN projection_frontiers f ON f.scope=s.scope AND f.channel=?",
                        (optional_projection_channel,))[0]['n']
                rows += [f'{name}{{track="{self.track}"}} {value}' for name, value in counts.items()]
            except Exception:
                rows.append(f'metrics_dependency_unavailable{{track="{self.track}"}} 1')
        if reranker is not None and hasattr(reranker, 'snapshot'):
            snapshot = reranker.snapshot()
            for name in ('ready_replicas', 'configured_replicas', 'capacity_degraded',
                         'queue_depth', 'queue_seconds', 'rejected', 'failovers'):
                rows.append(f'reranker_{name}{{track="{self.track}"}} {float(snapshot[name])}')
            for replica in snapshot['replicas']:
                labels = f'track="{self.track}",replica="{replica["replica"]}"'
                for name in ('ready', 'quarantined', 'inflight', 'requests', 'successes', 'errors', 'seconds'):
                    rows.append(f'reranker_replica_{name}{{{labels}}} {float(replica[name])}')
        if executor is not None:
            rows += [f'api_workers_{name}{{track="{self.track}"}} {value}'
                     for name, value in executor.snapshot().items()]
        return '\n'.join(rows) + '\n'


def install_observability(app, track, *, ready, state=lambda: None, validation_error_handler=None, exception_status=None):
    metrics = Metrics(track); app.state.metrics = metrics
    logger = logging.getLogger('mandol_competition.requests')
    local_operations = os.getenv('MANDOL_INTERNAL_ENDPOINTS_LOOPBACK_ONLY') == '1'

    @app.middleware('http')
    async def observe(request: Request, call_next):
        path = request.url.path.rstrip('/')
        if local_operations and (path in {'/metrics', '/readyz', '/livez', '/admin'} or path.startswith('/admin/')):
            try:
                local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
            except ValueError:
                local = False
            if not local:
                # No reliance on an untrusted forwarded header. Public health
                # and AML Add/Search keep their documented contract/auth.
                return JSONResponse(status_code=404, content={'detail':'Not Found'})
        operation = {'/add': 'add', '/search': 'search', '/health': 'health', '/livez': 'live', '/readyz': 'ready', '/metrics': 'metrics', '/admin/purge': 'purge'}.get(request.url.path, 'other')
        context=getattr(request.state,'diagnostic_context',None)
        correlation = context.correlation_id if context is not None else uuid.uuid4().hex
        started = time.monotonic(); status = 500
        try:
            response = await call_next(request); status = response.status_code
            response.headers['X-Request-ID'] = correlation
            return response
        except Exception as exc:
            if exception_status is not None: status=exception_status(exc)
            raise
        finally:
            elapsed = time.monotonic() - started
            metrics.observe(operation, elapsed, status)
            logger.info('api_request', extra={'track': track, 'operation': operation,
                'duration_seconds': elapsed, 'correlation_id': correlation,
                'error_class': 'server' if status >= 500 else 'client' if status >= 400 else 'none'})

    async def invalid(request, exc):
        # Pydantic's default includes the raw input, possibly Base64 or secrets.
        return JSONResponse(status_code=422, content={'detail': {'reason': 'request validation failed'}})
    app.add_exception_handler(RequestValidationError,validation_error_handler or invalid)

    @app.get('/livez', include_in_schema=False)
    def live(): return {'status': 'alive'}

    @app.get('/readyz', include_in_schema=False)
    def readiness():
        try:
            ready()
        except Exception:
            raise HTTPException(503, 'required dependency unavailable') from None
        return {'status': 'ready'}

    @app.get('/metrics', include_in_schema=False)
    def prometheus():
        owner = getattr(app.state, 'memory_service', None) or getattr(app.state, 'backend', None)
        return Response(metrics.render(state(), getattr(owner, 'reranker', None),
                        getattr(app.state, 'deadline_executor', None),
                        optional_projection_channel=getattr(owner, 'optional_projection_channel', 'dense')),
                        media_type='text/plain; version=0.0.4')
