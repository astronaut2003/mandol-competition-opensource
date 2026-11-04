"""Cross-user progress without changing C0 evidence or resource lifetime."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

import pytest

from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.config import Settings


@pytest.fixture
def service(tmp_path):
    memory = CodingMemoryService(Settings(coding_memory=True,
        coding_db_path=str(tmp_path / "code.sqlite3")))
    memory.start()
    for user in ("warm", "cold"):
        memory.add_messages(user, "session", "request", [dict(role="user",
            content="  JWT issuer fix in auth/middleware.py; regression test passes.\n")])
    yield memory
    if memory.ready:
        memory.shutdown()


def test_cold_user_does_not_block_another_warm_user_or_stats(service, monkeypatch):
    expected = service.search("warm", "JWT issuer fix", 100)
    entered, release = threading.Event(), threading.Event()
    projection = service._projection
    def delayed(user, state, rows):
        if user == "cold":
            entered.set()
            assert release.wait(5)
        return projection(user, state, rows)
    monkeypatch.setattr(service, "_projection", delayed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        cold = pool.submit(service.search, "cold", "JWT issuer fix", 100)
        assert entered.wait(2)
        try:
            warm = pool.submit(service.search, "warm", "JWT issuer fix", 100)
            stats = pool.submit(service.stats)
            assert warm.result(timeout=.5) == expected
            assert stats.result(timeout=.5)["stored_units"] == 2
        finally:
            release.set()
            assert cold.result(timeout=3)


def test_shutdown_drains_search_and_does_not_close_its_storage(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    search = service._search
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        assert service.store.state is not None
        return search(*args)
    monkeypatch.setattr(service, "_search", delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(service.search, "warm", "JWT issuer fix", 100)
        assert entered.wait(2)
        closing = pool.submit(service.shutdown)
        try:
            with pytest.raises(TimeoutError):
                closing.result(timeout=.05)
        finally:
            release.set()
        assert request.result(timeout=3)
        closing.result(timeout=3)
    assert service.store.state is None
    with pytest.raises(RuntimeError, match="not ready"):
        service.search("warm", "JWT issuer fix", 100)


def test_retention_waits_for_inflight_search_and_then_erases_cached_sources(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = service._search
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    monkeypatch.setattr(service, "_search", delayed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        request = pool.submit(service.search, "warm", "JWT issuer fix", 100)
        assert entered.wait(2)
        purging = pool.submit(service.purge_idle, 30)
        try:
            with pytest.raises(TimeoutError):
                purging.result(timeout=.05)
        finally:
            release.set()
        assert request.result(timeout=3)
        assert purging.result(timeout=3) == 0
    assert not service._cache


def test_search_gate_registry_is_released_and_users_remain_isolated(service):
    for i in range(30):
        assert service.search("unknown-" + str(i), "JWT issuer fix", 100) == []
    assert service.search("warm", "JWT issuer fix", 100)
    assert service._searches.active_scopes == 0
    assert service._lifecycle.active_scopes == 0
    assert len(service._cache) == 1


def test_health_remains_available_while_retention_waits_for_slow_model_io(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = service._projection
    def delayed(user, state, rows):
        entered.set()
        assert release.wait(5)
        return original(user, state, rows)
    monkeypatch.setattr(service, "_projection", delayed)
    with ThreadPoolExecutor(max_workers=3) as pool:
        request = pool.submit(service.search, "cold", "JWT issuer fix", 100)
        assert entered.wait(2)
        purging = pool.submit(service.purge_idle, 30)
        try:
            assert pool.submit(service.health).result(timeout=.5)
        finally:
            release.set()
        assert request.result(timeout=3)
        assert purging.result(timeout=3) == 0


def test_same_user_add_waits_for_its_snapshot_but_another_user_add_progresses(service, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = service._projection
    def delayed(user, state, rows):
        if user == "cold":
            entered.set()
            assert release.wait(5)
        return original(user, state, rows)
    monkeypatch.setattr(service, "_projection", delayed)
    message = [dict(role="user", content="Redis socket timeout configuration")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        request = pool.submit(service.search, "cold", "JWT issuer fix", 100)
        assert entered.wait(2)
        same = pool.submit(service.add_messages, "cold", "session", "next", message)
        try:
            other = pool.submit(service.add_messages, "warm", "session", "next", message)
            assert other.result(timeout=.5) == 1
            with pytest.raises(TimeoutError):
                same.result(timeout=.05)
        finally:
            release.set()
        assert request.result(timeout=3)
        assert same.result(timeout=3) == 1
    assert service.search("cold", "Redis socket timeout", 100)
