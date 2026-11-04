"""Bounded pooled transport; errors never include response bodies or credentials."""

import hashlib
import math
import json
import os
import random
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import httpx

from mandol_substrate.state.models import canonical_json
from mandol_substrate.deadlines import _deadline, deadline_budget, remaining_budget
from .errors import InferenceAdmissionDeferred, InferenceDeadlineExceeded, InferenceInputRejected, InferenceRateLimited, InferenceSchemaError, InferenceUnavailable
from .generation import is_openrouter_endpoint, validate_generation_proxy


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


def provider_error_metadata(response, body):
    """Extract identifiers only; never expose error messages or echoed input."""
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        value = {}
    value = value if isinstance(value, dict) else {}
    nested = value.get('error') if isinstance(value.get('error'), dict) else {}
    code = response.headers.get('X-Mandol-Upstream-Code') or value.get('code') or nested.get('code')
    request = (response.headers.get('X-Mandol-Upstream-Request-ID')
               or response.headers.get('X-Request-ID')
               or response.headers.get('X-DashScope-Request-ID')
               or value.get('request_id'))
    safe = lambda v: (str(v) if type(v) is int and 0 <= v <= 999999 else
                     v if isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', v) else None)
    return dict(provider_error_code=safe(code), provider_request_id=safe(request))


def _known_input_length_rejection(body):
    """Classify a bounded provider message in memory; never retain its text."""
    try:value=json.loads(body)
    except (ValueError,UnicodeError):return False
    if not isinstance(value,dict) or value.get('code')!='InvalidParameter':return False
    message=value.get('message')
    return isinstance(message,str) and len(message)<=16384 and bool(re.search(
        r'\b(?:query|document|input)\b.{0,80}(?:length|larger|max(?:imum)?)',message,re.I))


class HTTPTransport:
    def __init__(self, endpoint, *, timeout=30., max_attempts=3, max_connections=8,
                 api_key=None, transport=None, sleep=time.sleep, max_response_bytes=32 * 1024 * 1024,
                 jitter=random.uniform, proxy=None):
        if not math.isfinite(timeout) or not 0 < timeout <= 1800 or not 1 <= max_attempts <= 5:
            raise ValueError("bounded inference timeout and attempts required")
        if max_response_bytes < 1 or max_connections < 1:
            raise ValueError("positive response byte and connection bounds required")
        self.max_response_bytes = max_response_bytes
        self.endpoint, self.timeout = endpoint_origin(endpoint), timeout
        self.proxy = validate_generation_proxy(self.endpoint, proxy) if proxy is not None else None
        if self.proxy and transport is not None and not isinstance(transport, httpx.MockTransport):
            raise ValueError('a custom production transport cannot bypass the generation proxy')
        self.max_attempts, self._sleep, self._jitter = max_attempts, sleep, jitter
        from .budget import from_environment
        self.reliability = from_environment()
        self.client = httpx.Client(base_url=self.endpoint, timeout=timeout, trust_env=False,
            follow_redirects=False, transport=transport,
            proxy=self.proxy if transport is None else None,
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

    def request(self, method, path, payload=None, *, request_id=None, empty=False, retry_unknown=True,
                stream_decoder=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("inference path must be relative to its configured origin")
        if stream_decoder is not None and (not callable(stream_decoder) or method != 'POST'
                or path != '/chat/completions' or not isinstance(payload, dict)
                or payload.get('stream') is not True or empty or retry_unknown):
            raise ValueError('bounded non-repeating chat stream decoder required')
        # Only a digest is transmitted as correlation; raw user/request IDs never become headers.
        key = hashlib.sha256(canonical_json([request_id, method, path, payload]).encode()).hexdigest()
        started = time.monotonic()
        operational = self.reliability is not None and method != 'GET'
        seconds = self.reliability.seconds if operational else self.timeout
        attempts = self.reliability.attempts if operational else self.max_attempts
        until = min(started + seconds, _deadline.get() or float('inf'))
        internal_proxy = (urlsplit(self.endpoint).hostname in {'127.0.0.1', '::1', 'localhost'}
                          and path in {'/v1/encode', '/v1/embeddings', '/v1/sparse'})
        model = self.reliability.model(self.endpoint, payload) if operational else None
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
            if until <= time.monotonic():
                raise InferenceDeadlineExceeded()
            raise InferenceUnavailable("backpressure")
        failed = True
        status = None
        error = None
        try:
            for attempt in range(attempts):
                remaining = until - time.monotonic()
                if remaining <= 0:
                    raise error or InferenceDeadlineExceeded()
                if model is not None:
                    from .budget import estimate_tokens
                    try:
                        self.reliability.budget.acquire(model, estimate_tokens(model, payload), until=until)
                    except (InferenceAdmissionDeferred, InferenceDeadlineExceeded):
                        # A later admission wait must not hide an actual
                        # provider/connection failure from an earlier attempt.
                        if error is not None:
                            raise error from None
                        raise
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        raise error or InferenceDeadlineExceeded()
                headers = {"X-Request-ID": key, "Idempotency-Key": key}
                if operational and internal_proxy:
                    headers['X-Mandol-Deadline-Seconds'] = format(remaining, '.6f')
                attempt_timeout = remaining if operational and internal_proxy else min(remaining, self.timeout)
                decoded_stream = None
                try:
                    with self.client.stream(method, path, json=payload, timeout=attempt_timeout,
                            extensions={'trace': trace} if self._trace_path else None,
                            headers=headers) as response:
                        # Bound decoded bytes, including compressed responses.
                        # Even status-only health must drain the bounded body;
                        # closing an unread HTTP/1.1 stream discards its socket.
                        body = bytearray()
                        # Only bounded code/request identifiers are extracted
                        # from errors; no body or provider message is logged.
                        bound = self.max_response_bytes if response.status_code == 200 else 65536
                        if response.status_code == 200 and stream_decoder is not None:
                            status = response.status_code
                            def bounded_chunks():
                                size = 0
                                for chunk in response.iter_bytes():
                                    if time.monotonic() >= until:
                                        raise InferenceUnavailable('deadline', unknown_outcome=True)
                                    size += len(chunk)
                                    if size > bound:
                                        raise InferenceSchemaError('response_too_large')
                                    yield chunk
                            try:
                                chunks = bounded_chunks()
                                decoded_stream = stream_decoder(chunks)
                                if not isinstance(decoded_stream, dict):
                                    raise InferenceSchemaError('object_required')
                                # A successful terminal event still drains the
                                # bounded HTTP body for connection reuse. An
                                # abort above skips this and closes immediately.
                                for _ in chunks:
                                    pass
                            except Exception as exc:
                                # A received 200 stream may already be billed.
                                # Callback aborts and malformed streams never
                                # enter a second paid transport attempt.
                                exc.vision_paid_output_failure = True
                                if isinstance(exc, InferenceUnavailable):
                                    exc.unknown_outcome = True
                                raise
                        try:
                            for chunk in (() if decoded_stream is not None else response.iter_bytes()):
                                if time.monotonic() >= until:
                                    if response.status_code != 200:
                                        break  # Preserve the known rejection status/headers.
                                    raise InferenceUnavailable("deadline", unknown_outcome=method != "GET")
                                if len(body) + len(chunk) > bound:
                                    if response.status_code == 200:
                                        raise InferenceSchemaError("response_too_large")
                                    break
                                body.extend(chunk)
                        except httpx.RequestError:
                            if response.status_code == 200:
                                raise
                            # An incomplete diagnostic body cannot erase a
                            # received HTTP error, notably 429 / Retry-After.
                            body.clear()
                        if response.status_code == 200 and decoded_stream is None:
                            response._content = bytes(body)
                except httpx.RequestError as exc:
                    # A connect/pool failure precedes dispatch. Reads/writes or
                    # resets can follow a paid dispatch and remain unknown.
                    unknown = method != "GET" and (getattr(exc, 'vision_paid_output_failure', False)
                        or not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout, httpx.ProxyError)))
                    error = InferenceUnavailable(type(exc).__name__, unknown_outcome=unknown)
                    if getattr(exc, 'vision_paid_output_failure', False):
                        error.vision_paid_output_failure = True
                    if (not retry_unknown and unknown) or attempt + 1 == attempts:
                        raise error from None
                    delay = self._jitter(0., min(.1 * 2**attempt, 1.))
                else:
                    status = response.status_code
                    accepted_error = False
                    value = decoded_stream
                    if status == 200 and (path == '/chat/completions' or is_openrouter_endpoint(self.endpoint)):
                        try:
                            if value is None:
                                value = response.json()
                        except ValueError:
                            raise InferenceSchemaError('invalid_json') from None
                        if not isinstance(value, dict):
                            raise InferenceSchemaError('object_required')
                        if 'error' in value:
                            nested = value['error']
                            code = nested.get('code') if isinstance(nested, dict) else None
                            if isinstance(code, str) and re.fullmatch(r'[45][0-9]{2}', code):
                                code = int(code)
                            if type(code) is not int or not 400 <= code <= 599:
                                raise InferenceUnavailable('generation_error_unverified',
                                    status_code=200, unknown_outcome=True)
                            status, accepted_error = code, True
                    with self._stats_lock:
                        self._stats['http_429'] += status == 429
                        self._stats['http_5xx'] += status >= 500
                    if status == 200:
                        if empty:
                            failed = False
                            return {}
                        if value is None:
                            try:
                                value = response.json()
                            except ValueError:
                                raise InferenceSchemaError("invalid_json") from None
                        if not isinstance(value, dict):
                            raise InferenceSchemaError("object_required")
                        failed = False
                        return value
                    header_delay = retry_after_seconds(response.headers.get('Retry-After'))
                    metadata = provider_error_metadata(response, body)
                    if (internal_proxy and status in {400, 413, 422}
                            and response.headers.get('X-Mandol-Input-Rejected') == '1'):
                        error = InferenceInputRejected(status_code=status, **metadata)
                    elif status == 429:
                        error = InferenceRateLimited(retry_after=header_delay, **metadata)
                        if accepted_error:
                            # HTTP 200 means upstream accepted the call. Billing
                            # is uncertain even when a later error says 429.
                            error.unknown_outcome = True
                            error.vision_paid_output_failure = True
                        header_delay = error.retry_after
                        if model is not None:
                            try:
                                self.reliability.budget.defer(model, header_delay)
                            except InferenceUnavailable:
                                # Preserve an actual provider 429. A later
                                # admission still fails closed on store errors.
                                raise error from None
                    else:
                        error = InferenceUnavailable("http_" + str(status), status_code=status,
                            unknown_outcome=accepted_error or status in {500, 502, 503, 504},
                            retry_after=header_delay, **metadata)
                    if (status==400 and isinstance(payload,dict)
                            and payload.get('model')=='qwen3.7-text-rerank'):
                        error.input_length_rejected=_known_input_length_rejection(body)
                    if (status not in {408, 425, 429, 500, 502, 503, 504} or attempt + 1 == attempts
                            or status == 429 and operational and internal_proxy
                            or not retry_unknown and error.unknown_outcome):
                        raise error
                    delay = self._jitter(0., min(.1 * 2**attempt, 1.))
                    if header_delay is not None:
                        delay = max(delay, header_delay)
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
