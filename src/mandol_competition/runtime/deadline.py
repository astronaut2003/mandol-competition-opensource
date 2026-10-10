"""Bounded Competition API deadline executor; timed-out writes retain ownership.

A timeout is unknown completion, never an assertion that a write rolled back.
Running work retains its slot and storage resources until it finishes. A caller
must drain this executor before shutting down its canonical/projection clients.
"""

import math
import os
import threading
import time
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from mandol_substrate.state.models import StateUnavailable


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
        self._waiting = self._rejected = 0
        self._queue_seconds = 0.
        self.admission_timeout = admission_timeout

    def run(self, function, *, seconds):
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("Finite positive API deadline required")
        end = time.monotonic() + seconds
        with self._lock:
            if self._closed:
                raise StateUnavailable("API worker admission is closed")
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
                future = self._pool.submit(copy_context().run, function)
                self._active.add(future)
                # Register before close can observe the submitted Future. This
                # avoids a completed-worker / not-yet-registered callback race.
                future.add_done_callback(self._finished)
        except BaseException:
            if future is None:
                self._slots.release()
            raise
        try:
            return future.result(timeout=max(0.0, end - time.monotonic()))
        except TimeoutError:
            # A task may itself raise TimeoutError; neither case permits a fake
            # success or a statement that publication did not occur.
            raise StateUnavailable(
                "API deadline exceeded; completion may be pending; retry the same Add request_id"
            ) from None

    def _finished(self, future):
        with self._lock:
            self._active.remove(future)
        self._slots.release()

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
