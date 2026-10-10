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

from .errors import InferenceSchemaError, InferenceUnavailable
from .http import deadline_budget, endpoint_origin


def replica_endpoints(track, primary):
    raw = os.getenv('MANDOL_' + track.upper() + '_RERANKER_REPLICAS')
    endpoints = [endpoint_origin(s.strip()) for s in raw.split(',')] if raw else [primary]
    if not 1 <= len(endpoints) <= 3 or len(set(endpoints)) != len(endpoints):
        raise ValueError('one to three distinct reranker replicas required')
    return endpoints


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
        replica.retry_at = time.monotonic() + min(60., self.cooldown * 2 ** min(replica.consecutive_errors - 1, 4))

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
        except Exception as exc:
            with self._condition:
                self._failure(r, exc)
        else:
            with self._condition:
                r.ready = True
                r.consecutive_errors = 0
                r.error_class = 'none'
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
        until = started + self.queue_timeout
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
                        'error_class': r.error_class} for i, r in enumerate(self._replicas)]}

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
