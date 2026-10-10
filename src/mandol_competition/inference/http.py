"""Bounded pooled transport; errors never include response bodies or credentials."""

import hashlib
import math
import json
import os
import random
from contextlib import contextmanager
from contextvars import ContextVar
from email.utils import parsedate_to_datetime
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import httpx

from mandol_substrate.state.models import canonical_json
from .errors import InferenceSchemaError, InferenceUnavailable


_deadline = ContextVar('inference_deadline', default=None)


@contextmanager
def deadline_budget(seconds):
    """Bound a sequence of inference calls, including admission and retries."""
    until = time.monotonic() + seconds
    current = _deadline.get()
    token = _deadline.set(min(current, until) if current is not None else until)
    try:
        yield
    finally:
        _deadline.reset(token)


def retry_after_seconds(value, *, now=None):
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            stamp = parsedate_to_datetime(value)
            if stamp.tzinfo is None:
                return None
            seconds = max(0., stamp.timestamp() - (time.time() if now is None else now))
        except (TypeError, ValueError, OverflowError):
            return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def endpoint_origin(url):
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("credential-free HTTP inference endpoint required")
    return url.rstrip("/")


class HTTPTransport:
    def __init__(self, endpoint, *, timeout=30., max_attempts=3, max_connections=8,
                 api_key=None, transport=None, sleep=time.sleep, max_response_bytes=32 * 1024 * 1024,
                 jitter=random.uniform):
        if not math.isfinite(timeout) or not 0 < timeout <= 1800 or not 1 <= max_attempts <= 5:
            raise ValueError("bounded inference timeout and attempts required")
        if max_response_bytes < 1 or max_connections < 1:
            raise ValueError("positive response byte and connection bounds required")
        self.max_response_bytes = max_response_bytes
        self.endpoint, self.timeout = endpoint_origin(endpoint), timeout
        self.max_attempts, self._sleep, self._jitter = max_attempts, sleep, jitter
        self.client = httpx.Client(base_url=self.endpoint, timeout=timeout, trust_env=False,
            follow_redirects=False, transport=transport,
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            headers={"Authorization": "Bearer " + api_key} if api_key else {})
        self._admission = threading.BoundedSemaphore(max_connections)
        self._trace_path = os.getenv('MANDOL_INFERENCE_TRACE')
        self._trace_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._created = time.monotonic()
        self._stats = dict(requests=0, errors=0, retries=0, inflight=0, waiting=0, connections=0,
                           http_429=0, http_5xx=0, seconds=0., queue_seconds=0.)

    def snapshot(self):
        with self._stats_lock:
            return dict(self._stats, uptime_seconds=time.monotonic() - self._created)

    def request(self, method, path, payload=None, *, request_id=None, empty=False, retry_unknown=True):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("inference path must be relative to its configured origin")
        # Only a digest is transmitted as correlation; raw user/request IDs never become headers.
        key = hashlib.sha256(canonical_json([request_id, method, path, payload]).encode()).hexdigest()
        started = time.monotonic()
        until = min(started + self.timeout, _deadline.get() or float('inf'))
        connects = 0
        def trace(event, info):
            nonlocal connects
            if event == 'connection.connect_tcp.complete':
                connects += 1
        with self._stats_lock:
            self._stats['waiting'] += 1
        admitted = self._admission.acquire(timeout=max(0., until - time.monotonic()))
        with self._stats_lock:
            self._stats['waiting'] -= 1
            self._stats['queue_seconds'] += time.monotonic() - started
            self._stats['requests'] += 1
            if admitted:
                self._stats['inflight'] += 1
            else:
                self._stats['errors'] += 1
        if not admitted:
            raise InferenceUnavailable("backpressure")
        failed = True
        status = None
        try:
            for attempt in range(self.max_attempts):
                remaining = until - time.monotonic()
                if remaining <= 0:
                    raise InferenceUnavailable("deadline", unknown_outcome=method != "GET")
                try:
                    with self.client.stream(method, path, json=payload, timeout=remaining,
                            extensions={'trace': trace} if self._trace_path else None,
                            headers={"X-Request-ID": key, "Idempotency-Key": key}) as response:
                        # Bound decoded bytes, including compressed responses.
                        # Even status-only health must drain the bounded body;
                        # closing an unread HTTP/1.1 stream discards its socket.
                        body = bytearray()
                        # Small error responses are drained for keep-alive but
                        # never parsed, retained in logs or returned to callers.
                        bound = self.max_response_bytes if response.status_code == 200 else 65536
                        for chunk in response.iter_bytes():
                            if time.monotonic() >= until:
                                raise InferenceUnavailable("deadline", unknown_outcome=method != "GET")
                            if len(body) + len(chunk) > bound:
                                if response.status_code == 200:
                                    raise InferenceSchemaError("response_too_large")
                                break
                            body.extend(chunk)
                        if response.status_code == 200:
                            response._content = bytes(body)
                except httpx.RequestError as exc:
                    # A connect/pool failure precedes dispatch. Reads/writes or
                    # resets can follow a paid dispatch and remain unknown.
                    unknown = method != "GET" and not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
                    error = InferenceUnavailable(type(exc).__name__, unknown_outcome=unknown)
                    if (not retry_unknown and unknown) or attempt + 1 == self.max_attempts:
                        raise error from None
                    delay = self._jitter(0., min(.1 * 2**attempt, 1.))
                else:
                    status = response.status_code
                    with self._stats_lock:
                        self._stats['http_429'] += status == 429
                        self._stats['http_5xx'] += status >= 500
                    if status == 200:
                        if empty:
                            failed = False
                            return {}
                        try:
                            value = response.json()
                        except ValueError:
                            raise InferenceSchemaError("invalid_json") from None
                        if not isinstance(value, dict):
                            raise InferenceSchemaError("object_required")
                        failed = False
                        return value
                    header_delay = retry_after_seconds(response.headers.get('Retry-After'))
                    error = InferenceUnavailable("http_" + str(status), status_code=status,
                                                 unknown_outcome=status in {500, 502, 503, 504},
                                                 retry_after=header_delay)
                    if (status not in {408, 425, 429, 500, 502, 503, 504} or attempt + 1 == self.max_attempts
                            or not retry_unknown and error.unknown_outcome):
                        raise error
                    delay = self._jitter(0., min(.1 * 2**attempt, 1.))
                    if header_delay is not None:
                        delay = max(delay, header_delay)
                    elif status == 429:
                        delay = max(delay, 1.)
                if delay >= until - time.monotonic():
                    raise error
                with self._stats_lock:
                    self._stats['retries'] += 1
                self._sleep(delay)
            raise InferenceUnavailable("attempts_exhausted")
        finally:
            self._admission.release()
            with self._stats_lock:
                self._stats['inflight'] -= 1
                self._stats['errors'] += failed
                self._stats['connections'] += connects
                self._stats['seconds'] += time.monotonic() - started
            if self._trace_path:
                row = {'timestamp': time.time(), 'method': method, 'endpoint': self.endpoint, 'path': path,
                    'model': (payload or {}).get('model'), 'payload_sha256': key,
                    'input_sha256': hashlib.sha256(canonical_json((payload or {}).get('input')).encode()).hexdigest(),
                    'input_count': len((payload or {}).get('input', [])) if isinstance((payload or {}).get('input', []), (list, str)) else None,
                    'seconds': time.monotonic()-started, 'new_connections': connects,
                    'status': locals().get('status'), 'attempts': locals().get('attempt', 0)+1}
                try:
                    with self._trace_lock, Path(self._trace_path).open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                except OSError:
                    pass  # Observability must not change inference/publication outcomes.

    def health(self):
        self.request("GET", "/health", empty=True)
        return True

    def close(self):
        self.client.close()
