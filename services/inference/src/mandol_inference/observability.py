"""Content-free CPU bridge counters; these services bind loopback or VPC only."""
from collections import Counter
import threading
import time

from fastapi import HTTPException


def install_bridge_observability(app, *, ready, transports, extra=lambda: {}):
    started = time.monotonic()
    lock = threading.Lock()
    counters = Counter()

    @app.middleware('http')
    async def observe(request, call_next):
        operation = request.url.path if request.url.path in {
            '/v1/encode', '/v1/embeddings', '/v1/sparse', '/v1/score'} else 'probe'
        begin = time.monotonic()
        with lock:
            counters['inflight'] += 1
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            with lock:
                counters['inflight'] -= 1
                if operation != 'probe':
                    counters['requests'] += 1
                    counters['errors'] += status >= 400
                    counters['seconds'] += time.monotonic() - begin

    @app.get('/ready', include_in_schema=False)
    def readiness():
        try:
            ready()
        except Exception:
            raise HTTPException(503, 'required inference dependency unavailable') from None
        return {'status': 'ready'}

    @app.get('/ops/state', include_in_schema=False)
    def state():
        with lock:
            snapshot = dict(counters)
        return {'uptime_seconds': time.monotonic() - started, **snapshot,
                'transports': {name: client.snapshot() for name, client in transports().items()
                               if hasattr(client, 'snapshot')}, **extra()}
