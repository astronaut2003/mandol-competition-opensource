from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from mandol_competition.code.api.app import create_app
from mandol_competition.code.api.config import Settings


def test_shutdown_waits_for_timed_out_write_before_closing_storage():
    entered, release, finished, closed = (threading.Event() for _ in range(4))
    def add(**kwargs):
        entered.set()
        assert release.wait(5)
        assert not closed.is_set()
        finished.set()
    def shutdown():
        assert finished.is_set()
        closed.set()
    memory = SimpleNamespace(ready=True, start=lambda: None, add_messages=add,
        shutdown=shutdown, purge_idle=lambda _: 0, store=SimpleNamespace(state=None))
    settings = replace(Settings.from_env(), memory_system_key='fixture', add_timeout_seconds=.03)
    app = create_app(settings, memory_service=memory)
    client = TestClient(app)
    client.__enter__()
    try:
        until = time.monotonic() + 3
        while client.get('/health').status_code != 200:
            assert time.monotonic() < until
            time.sleep(.01)
        response = client.post('/add', headers={'X-Api-Key': 'fixture'}, json={
            'request_id': 'drain', 'user_id': 'u', 'session_id': 's',
            'messages': [{'role': 'user', 'content': 'durable write'}]})
        assert response.status_code == 503 and entered.is_set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            closing = pool.submit(client.__exit__, None, None, None)
            assert not closed.wait(.05)
            release.set()
            closing.result(timeout=5)
        assert finished.is_set() and closed.is_set()
    finally:
        release.set()
