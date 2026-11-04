"""Bounded Competition API deadline executor; timed-out writes retain ownership.

A timeout is unknown completion, never an assertion that a write rolled back.
Running work retains its slot and storage resources until it finishes. A caller
must drain this executor before shutting down its canonical/projection clients.
"""

import math
import hashlib
import json
import os
import threading
import time
from contextvars import copy_context
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

from mandol_substrate.state.models import StateUnavailable


def add_work_key(track, payload):
    """Coalesce exact concurrent Add bodies; durable identity stays in the ledger.

    Include the complete parsed body, including user/session/request identity.
    Different bodies never share results: their existing durable conflict and
    normalization semantics remain authoritative. Nothing is cached after work.
    """
    value = payload.model_dump(mode='json')
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'),
                               ensure_ascii=False, allow_nan=False)
    for chunk in encoder.iterencode([track, value]):
        digest.update(chunk.encode('utf-8'))
    return digest.hexdigest()


class ApiDeadlineExecutor:
    def __init__(self, *, capacity=None, admission_timeout=None, max_queue=None):
        capacity = int(os.getenv('MANDOL_API_WORKER_CAPACITY', '8')) if capacity is None else capacity
        admission_timeout = float(os.getenv('MANDOL_API_QUEUE_TIMEOUT_SECONDS', '5')) if admission_timeout is None else admission_timeout
        max_queue = int(os.getenv('MANDOL_API_MAX_QUEUE', str(capacity))) if max_queue is None else max_queue
        if type(capacity) is not int or not 1 <= capacity <= 64:
            raise ValueError("Invalid bounded API worker capacity")
        if not math.isfinite(admission_timeout) or admission_timeout <= 0:
            raise ValueError("Invalid API admission timeout")
        if type(max_queue) is not int or not 0 <= max_queue <= 64:
            raise ValueError('Invalid bounded API waiting capacity')
        self.capacity, self.max_queue = capacity, max_queue
        self._slots = threading.BoundedSemaphore(capacity)
        self._pool = ThreadPoolExecutor(
            max_workers=capacity, thread_name_prefix="aml-deadline"
        )
        # Metadata only, never held during model/publication execution. RLock
        # permits an already-completed Future's inline completion callback.
        self._lock = threading.RLock()
        self._closed = False
        self._active = set()
        self._inflight = {}
        self._waiting = self._rejected = 0
        self._queue_seconds = 0.
        self.admission_timeout = admission_timeout

    def run(self, function, *, seconds, key=None):
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("Finite positive API deadline required")
        end = time.monotonic() + seconds
        proxy = Future()
        joined = False
        with self._lock:
            if self._closed:
                raise StateUnavailable("API worker admission is closed")
            if key is not None:
                if key in self._inflight:
                    if self._waiting >= self.max_queue:
                        raise StateUnavailable('API waiting queue is full; no work was submitted')
                    proxy = self._inflight[key]
                    self._waiting += 1
                    joined = True
                else:
                    self._inflight[key] = proxy
        if joined:
            started = time.monotonic()
            try:
                return self._result(proxy, end)
            finally:
                with self._lock:
                    self._waiting -= 1
                    self._queue_seconds += time.monotonic()-started
        try:
            return self._submit(function, seconds=seconds, end=end, key=key, proxy=proxy)
        except BaseException as exc:
            with self._lock:
                if key is not None and self._inflight.get(key) is proxy and not proxy.running():
                    # A submitted Future still owns this mapping after an HTTP
                    # timeout. Only failed admission has an unfinished proxy.
                    if not getattr(proxy, '_mandol_submitted', False):
                        self._inflight.pop(key, None)
                        if not proxy.done():
                            proxy.set_exception(exc)
            raise

    def _submit(self, function, *, seconds, end, key, proxy):
        started = time.monotonic()
        admitted = self._slots.acquire(blocking=False)
        if not admitted:
            with self._lock:
                if self._waiting >= self.max_queue:
                    self._rejected += 1
                    raise StateUnavailable('API waiting queue is full; no work was submitted')
                self._waiting += 1
            try:
                admitted = self._slots.acquire(timeout=min(seconds, self.admission_timeout))
            finally:
                with self._lock:
                    self._waiting -= 1
                    self._queue_seconds += time.monotonic() - started
            if not admitted:
                with self._lock:
                    self._rejected += 1
                raise StateUnavailable("API backpressure; no work was submitted")
        future = None
        try:
            with self._lock:
                if self._closed:
                    raise StateUnavailable("API worker admission is closed")
                if time.monotonic() >= end:
                    raise StateUnavailable(
                        "API deadline expired before work submission"
                    )
                # AnyIO propagates request context to its worker; a plain
                # ThreadPoolExecutor does not. Capture a distinct context per
                # submission, including timed-out work that continues owning
                # its slot. Never let a reused worker retain another request's
                # correlation, trace or scope-local context.
                context = copy_context()
                def bounded():
                    from mandol_substrate.deadlines import request_budget
                    remaining = end-time.monotonic()
                    if remaining <= 0:
                        raise StateUnavailable('API deadline expired before work execution')
                    with request_budget(remaining):
                        return function()
                future = self._pool.submit(context.run, bounded)
                self._active.add(future)
                proxy._mandol_submitted = True
                # Register before close can observe the submitted Future. This
                # avoids a completed-worker / not-yet-registered callback race.
                future.add_done_callback(lambda completed: self._finished(completed, key, proxy))
        except BaseException:
            if future is None:
                self._slots.release()
            raise
        return self._result(proxy, end)

    @staticmethod
    def _result(future, end):
        try:
            return future.result(timeout=max(0.0, end-time.monotonic()))
        except TimeoutError:
            # A task may itself raise TimeoutError; neither case permits a fake
            # success or a statement that publication did not occur.
            raise StateUnavailable(
                "API deadline exceeded; completion may be pending; retry the same Add request_id"
            ) from None

    def _finished(self, future, key, proxy):
        with self._lock:
            self._active.remove(future)
            if key is not None and self._inflight.get(key) is proxy:
                self._inflight.pop(key, None)
            # Completion must not become visible before its bounded slot is
            # available, otherwise an immediate next Add can be rejected idle.
            self._slots.release()
            try:
                proxy.set_result(future.result())
            except BaseException as exc:
                proxy.set_exception(exc)

    @property
    def active(self):
        with self._lock:
            return len(self._active)

    def snapshot(self):
        with self._lock:
            return {'active': len(self._active), 'queue_depth': self._waiting,
                    'queue_seconds': self._queue_seconds, 'rejected': self._rejected,
                    'capacity': self.capacity, 'max_queue': self.max_queue}

    def close(self):
        with self._lock:
            self._closed = True
        # Never close canonical/projection underneath an acknowledged/ongoing write.
        # Process SIGKILL still relies on the existing durable recovery protocol.
        self._pool.shutdown(wait=True, cancel_futures=False)
        if self.active:
            raise StateUnavailable("API worker drain incomplete")
