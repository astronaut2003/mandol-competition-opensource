from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from mandol_competition.runtime.deadline import ApiDeadlineExecutor
from mandol_substrate.state import StateUnavailable


def test_bounded_waiting_queue_rejects_unsubmitted_work_and_observes_drain():
    executor = ApiDeadlineExecutor(capacity=1, max_queue=1, admission_timeout=3)
    entered, release = threading.Event(), threading.Event()
    def active():
        entered.set()
        assert release.wait(5)
        return 'completed'
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(executor.run, active, seconds=5)
            assert entered.wait(3)
            second = callers.submit(executor.run, lambda: 'queued', seconds=5)
            until = time.monotonic() + 2
            while executor.snapshot()['queue_depth'] != 1 and time.monotonic() < until:
                time.sleep(.001)
            assert executor.snapshot()['queue_depth'] == 1
            with pytest.raises(StateUnavailable, match='queue is full'):
                executor.run(lambda: pytest.fail('rejected work must not run'), seconds=5)
            release.set()
            assert first.result(timeout=5) == 'completed'
            assert second.result(timeout=5) == 'queued'
        executor.close()
        snapshot = executor.snapshot()
        assert snapshot['active'] == snapshot['queue_depth'] == 0
        assert snapshot['rejected'] == 1
    finally:
        release.set()
        executor.close()


def test_production_capacity_is_explicit_and_environment_bounds_are_checked(monkeypatch):
    monkeypatch.setenv('MANDOL_API_WORKER_CAPACITY', '16')
    executor = ApiDeadlineExecutor()
    try:
        assert executor.capacity == 16 and executor.max_queue == 16
    finally:
        executor.close()
    monkeypatch.setenv('MANDOL_API_WORKER_CAPACITY', '65')
    with pytest.raises(ValueError):
        ApiDeadlineExecutor()
