from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mandol_competition.runtime.deadline import ApiDeadlineExecutor
from mandol_substrate.state.models import StateUnavailable


def test_deadline_copies_http_context_and_reused_worker_does_not_leak():
    correlation = ContextVar("deadline-request-correlation", default=None)
    worker = ApiDeadlineExecutor(capacity=1)
    app = FastAPI()

    @app.middleware("http")
    async def correlate(request, next_handler):
        token = correlation.set(request.headers.get("X-Test-Correlation"))
        try:
            return await next_handler(request)
        finally:
            correlation.reset(token)

    @app.get("/probe")
    def probe():
        def task():
            value = correlation.get()
            correlation.set("worker-local-change")
            return value
        return {"worker": worker.run(task, seconds=5), "caller": correlation.get()}

    try:
        with TestClient(app) as client:
            for value in ("first", "second", None):
                response = client.get("/probe", headers={"X-Test-Correlation": value} if value else {})
                assert response.json() == {"worker": value, "caller": value}
        assert correlation.get() is None
    finally:
        worker.close()


def test_timed_out_work_keeps_its_context_and_slot_until_drained():
    correlation = ContextVar("timed-out-correlation", default=None)
    worker = ApiDeadlineExecutor(capacity=1)
    entered, release, finished = (threading.Event() for _ in range(3))
    seen = []
    def task():
        entered.set()
        assert release.wait(5)
        seen.append(correlation.get())
        finished.set()
    token = correlation.set("original-request")
    try:
        with pytest.raises(StateUnavailable, match="completion may be pending"):
            worker.run(task, seconds=.03)
        assert entered.is_set() and worker.active == 1
        correlation.set("next-request")
        release.set()
        assert finished.wait(5)
        worker.close()
        assert seen == ["original-request"] and worker.active == 0
        assert correlation.get() == "next-request"
    finally:
        release.set()
        worker.close()
        correlation.reset(token)


def test_simultaneous_submissions_have_separate_contexts():
    correlation = ContextVar("parallel-request", default=None)
    worker = ApiDeadlineExecutor(capacity=2)
    rendezvous = threading.Barrier(2)
    def submit(value):
        token = correlation.set(value)
        try:
            def task():
                rendezvous.wait(timeout=5)
                return correlation.get()
            return worker.run(task, seconds=6)
        finally:
            correlation.reset(token)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures = [callers.submit(submit, value) for value in ("user-A", "user-B")]
            assert [f.result(timeout=8) for f in futures] == ["user-A", "user-B"]
        assert correlation.get() is None
    finally:
        worker.close()
