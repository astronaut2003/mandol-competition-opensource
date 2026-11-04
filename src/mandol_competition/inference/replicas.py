"""Whole-rerank least-inflight scheduling for identically admitted replicas.

Scores from a failed attempt are discarded. A retry scores the complete original
candidate list on one different replica. This module never changes model math,
candidate order, prompts, evidence policy or a track's failure fallback.
"""
from dataclasses import dataclass
import math
import os
import threading
import time

from mandol_substrate.state import ProfileMismatch, StateUnavailable

from .errors import InferenceAdmissionDeferred, InferenceDeadlineExceeded, InferenceRateLimited, InferenceSchemaError, InferenceUnavailable
from .http import deadline_budget, endpoint_origin, remaining_budget


def replica_endpoints(track, primary, configured=None):
    raw = os.getenv('MANDOL_' + track.upper() + '_RERANKER_REPLICAS')
    endpoints = ([endpoint_origin(s) for s in configured] if configured is not None else
                 [endpoint_origin(s.strip()) for s in raw.split(',')] if raw else [endpoint_origin(primary)])
    if not 1 <= len(endpoints) <= 3 or len(set(endpoints)) != len(endpoints):
        raise ValueError('one to three distinct reranker replicas required')
    return endpoints


POOL_OPTIONS = frozenset({'max_inflight', 'max_queue', 'queue_timeout', 'cooldown',
                         'probe_interval', 'probe_timeout', 'warmup_timeout', 'request_timeout'})


def pool_configuration(value=None):
    """Validate operational bounds without introducing any scoring settings."""
    value = dict(value or {})
    if set(value) - (POOL_OPTIONS | {'endpoints'}):
        raise ValueError('unknown reranker pool setting')
    for key in ('max_inflight', 'max_queue'):
        if key in value and (type(value[key]) is not int or not (key == 'max_queue' and value[key] == 0 or 1 <= value[key] <= 1024)):
            raise ValueError('bounded integer pool capacity required')
    for key in POOL_OPTIONS - {'max_inflight', 'max_queue'}:
        if key in value and (type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not 0 < value[key] <= 1800):
            raise ValueError('finite bounded pool timeout required')
    if 'endpoints' in value:
        endpoints = value['endpoints']
        if not isinstance(endpoints, (list, tuple)) or not 1 <= len(endpoints) <= 3:
            raise ValueError('one to three distinct reranker replicas required')
        value['endpoints'] = [endpoint_origin(s) for s in endpoints]
        if len(set(value['endpoints'])) != len(endpoints):
            raise ValueError('duplicate reranker replicas')
    return value


def create_pool(clients, settings=None):
    """One scheduler for one, two or three replicas, including Local mode."""
    try:
        options = pool_configuration(settings)
        from .budget import from_environment
        reliability = from_environment()
        if reliability is not None:
            options['queue_timeout'] = reliability.seconds
        return ReplicaReranker(clients, **{k: v for k, v in options.items() if k != 'endpoints'})
    except BaseException:
        # Construction owns these clients even if admission configuration fails.
        # Do not leak connections or mask the original identity/config error.
        for client in clients:
            try: client.close()
            except Exception: pass
        raise


@dataclass
class _Replica:
    client: object
    ready: bool = False
    probing: bool = False
    quarantined: bool = False
    retry_at: float = 0.
    last_probe: float = 0.
    inflight: int = 0
    calls: int = 0
    successes: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    seconds: float = 0.
    error_class: str = 'none'
    last_http_status: int | None = None
    provider_error_code: str | None = None
    provider_request_id: str | None = None
    admission_deferrals: int = 0


class ReplicaReranker:
    def __init__(self, clients, *, max_inflight=4, max_queue=8, queue_timeout=5.,
                 cooldown=5., probe_interval=10., probe_timeout=3., warmup_timeout=60.,
                 request_timeout=1800., monitor=True):
        if not 1 <= len(clients) <= 3 or max_inflight < 1 or max_queue < 0:
            raise ValueError('bounded replica capacity required')
        for seconds in (queue_timeout, cooldown, probe_interval, probe_timeout, warmup_timeout, request_timeout):
            if not math.isfinite(seconds) or not 0 < seconds <= 1800:
                raise ValueError('finite bounded scheduler timeouts required')
        # Endpoint differences are physical serving identity. Recipes and the
        # original Querit preparation policy must be equal across this pool.
        for attr in ('recipe', 'expected_profile'):
            if hasattr(clients[0], attr):
                identity = getattr(clients[0], attr)
                if any(getattr(c, attr, None) != identity for c in clients):
                    raise ProfileMismatch('reranker pool contains different frozen recipes')
                setattr(self, attr, identity)
        self._replicas = [_Replica(c) for c in clients]
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._monitor_enabled = monitor
        self._closed = False
        self._cursor = 0
        self._waiting = self._rejected = self._failovers = 0
        self._queue_seconds = 0.
        self._created = time.monotonic()
        self.max_inflight, self.max_queue = max_inflight, max_queue
        self.queue_timeout, self.cooldown = queue_timeout, cooldown
        self.probe_interval, self.probe_timeout = probe_interval, probe_timeout
        self.warmup_timeout, self.request_timeout = warmup_timeout, request_timeout

    def _failure(self, replica, exc):
        replica.ready = False
        replica.errors += 1
        replica.consecutive_errors += 1
        replica.error_class = type(exc).__name__
        replica.quarantined = isinstance(exc, ProfileMismatch)
        wait = min(60., self.cooldown * 2 ** min(replica.consecutive_errors - 1, 4))
        replica.last_http_status = getattr(exc, 'status_code', None)
        replica.provider_error_code = getattr(exc, 'provider_error_code', None)
        replica.provider_request_id = getattr(exc, 'provider_request_id', None)
        if replica.last_http_status == 429:
            delay = getattr(exc, 'retry_after', None)
            wait = max(wait, delay if type(delay) in (int, float) and math.isfinite(delay) and delay >= 0 else 60.)
        replica.retry_at = time.monotonic() + wait

    def _probe(self, index, *, startup=False):
        with self._condition:
            r = self._replicas[index]
            if self._closed or r.probing or r.inflight or r.quarantined:
                return
            recovering = not r.ready
            r.probing = True
        try:
            with deadline_budget(self.warmup_timeout if startup else self.probe_timeout):
                if recovering:
                    r.client.validate_profile()
                    # Re-entry requires a real finite complete-pair warmup,
                    # not just an alive TCP socket or recipe declaration.
                    values = r.client.rerank('memory', ['memory health probe'])
                    self._check_scores(values, 1)
                else:
                    r.client.health()
        except (InferenceAdmissionDeferred, InferenceDeadlineExceeded) as exc:
            with self._condition:
                # A probe which could not dispatch provides no new evidence
                # about provider health. Keep a healthy replica admitted, and
                # never admit a previously failed one without a real warmup.
                r.admission_deferrals += 1
                delay = getattr(exc, 'retry_after', None)
                if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                    delay = self.probe_interval
                if not r.ready:
                    r.retry_at = max(r.retry_at, time.monotonic()+max(.1, delay))
        except Exception as exc:
            with self._condition:
                self._failure(r, exc)
        else:
            with self._condition:
                r.ready = True
                r.consecutive_errors = 0
                r.error_class = 'none'
                r.last_http_status = r.provider_error_code = r.provider_request_id = None
        finally:
            with self._condition:
                r.probing = False
                r.last_probe = time.monotonic()
                self._condition.notify_all()

    def validate_profile(self):
        for index in range(len(self._replicas)):
            self._probe(index, startup=True)
        self.health()
        if self._monitor_enabled and self._thread is None:
            self._thread = threading.Thread(target=self._monitor, name='reranker-readiness', daemon=True)
            self._thread.start()
        return True

    def _monitor(self):
        while not self._stop.wait(min(1., self.probe_interval)):
            for index, r in enumerate(self._replicas):
                now = time.monotonic()
                with self._condition:
                    due = (not r.quarantined and not r.probing and not r.inflight
                           and (now >= r.retry_at if not r.ready else now - r.last_probe >= self.probe_interval))
                if due:
                    self._probe(index)

    def health(self):
        with self._condition:
            if self._closed or not any(r.ready and not r.quarantined for r in self._replicas):
                raise InferenceUnavailable('all_reranker_replicas_unavailable')
        return True

    def _select(self, excluded):
        started = time.monotonic()
        until = started + remaining_budget(self.queue_timeout)
        queued = False
        with self._condition:
            try:
                while True:
                    if self._closed:
                        raise InferenceUnavailable('reranker_pool_closed')
                    eligible = [i for i, r in enumerate(self._replicas)
                                if i not in excluded and r.ready and not r.quarantined]
                    available = [i for i in eligible if not self._replicas[i].probing
                                 and self._replicas[i].inflight < self.max_inflight]
                    if available:
                        index = min(available, key=lambda i: (self._replicas[i].inflight,
                                    (i - self._cursor) % len(self._replicas)))
                        self._cursor = (index + 1) % len(self._replicas)
                        r = self._replicas[index]
                        r.inflight += 1
                        r.calls += 1
                        return index, r
                    if not eligible:
                        limited = [r for i, r in enumerate(self._replicas)
                                   if i not in excluded and not r.quarantined and r.last_http_status == 429]
                        if limited:
                            r = min(limited, key=lambda item: item.retry_at)
                            raise InferenceRateLimited('reranker_rate_limited',
                                retry_after=max(0., r.retry_at-time.monotonic()),
                                provider_error_code=r.provider_error_code,
                                provider_request_id=r.provider_request_id)
                        raise InferenceUnavailable('all_reranker_replicas_unavailable')
                    if not queued:
                        if self._waiting >= self.max_queue:
                            self._rejected += 1
                            raise InferenceUnavailable('reranker_backpressure')
                        self._waiting += 1
                        queued = True
                    remaining = until - time.monotonic()
                    if remaining <= 0:
                        self._rejected += 1
                        raise InferenceUnavailable('reranker_queue_deadline')
                    self._condition.wait(remaining)
            finally:
                if queued:
                    self._waiting -= 1
                self._queue_seconds += time.monotonic() - started

    @staticmethod
    def _check_scores(values, count):
        if (not isinstance(values, list) or len(values) != count
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise InferenceSchemaError('invalid_complete_rerank')

    def rerank(self, query, documents):
        # Materialize once: retries preserve the complete original list/order.
        documents = list(documents)
        excluded = set()
        last_error = None
        with deadline_budget(self.request_timeout):
            for _ in self._replicas:
                try:
                    index, r = self._select(excluded)
                except InferenceUnavailable:
                    if last_error is not None:
                        raise last_error from None
                    raise
                excluded.add(index)
                started = time.monotonic()
                try:
                    values = r.client.rerank(query, documents)
                    self._check_scores(values, len(documents))
                except (InferenceAdmissionDeferred, InferenceDeadlineExceeded):
                    # Shared account admission cannot be improved by trying a
                    # second replica, and must not poison readiness.
                    with self._condition:
                        r.admission_deferrals += 1
                    raise
                except ProfileMismatch as exc:
                    with self._condition:
                        self._failure(r, exc)
                    # Never hide a model/recipe identity violation with retry.
                    raise
                except StateUnavailable as exc:
                    with self._condition:
                        self._failure(r, exc)
                        self._failovers += 1
                    last_error = exc
                else:
                    with self._condition:
                        r.successes += 1
                    return values
                finally:
                    with self._condition:
                        r.inflight -= 1
                        r.seconds += time.monotonic() - started
                        self._condition.notify_all()
        raise last_error or InferenceUnavailable('all_reranker_replicas_unavailable')

    score = rerank

    def snapshot(self):
        with self._condition:
            ready = sum(r.ready and not r.quarantined for r in self._replicas)
            return {'uptime_seconds': time.monotonic() - self._created,
                    'ready_replicas': ready, 'configured_replicas': len(self._replicas),
                    'capacity_degraded': ready < len(self._replicas), 'queue_depth': self._waiting,
                    'queue_seconds': self._queue_seconds, 'rejected': self._rejected, 'failovers': self._failovers,
                    'replicas': [{'replica': chr(65 + i), 'ready': r.ready, 'probing': r.probing,
                        'quarantined': r.quarantined, 'inflight': r.inflight, 'requests': r.calls,
                        'successes': r.successes, 'errors': r.errors, 'seconds': r.seconds,
                        'error_class': r.error_class, 'admission_deferrals': r.admission_deferrals}
                        for i, r in enumerate(self._replicas)]}

    def attestation(self):
        return {'contract': 'whole-rerank-least-inflight-v1',
                'routing': 'one-complete-rerank-attempt-per-replica',
                'replicas': [r.client.attestation() for r in self._replicas],
                'capacity': self.snapshot()}

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.probe_timeout + 1.)
            if self._thread.is_alive():
                raise StateUnavailable('reranker readiness worker did not drain')
        for r in self._replicas:
            r.client.close()
