"""First-ingestion TTL is independent of subsequent Add/Search and restarts."""

from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.config import Settings


def test_thirty_day_window_does_not_slide_and_expired_cache_is_unreachable(tmp_path):
    now = [1000.0]
    cfg = Settings(coding_memory=True, coding_db_path=str(tmp_path / "memory.sqlite3"))
    service = CodingMemoryService(cfg)
    service.store.clock = lambda: now[0]
    service.start()
    service.add_messages("u", "s", "first", [{"role": "user", "content": "timeout 30 seconds"}])
    original = service.store.state.scope_attributes(service.store.scope("u"))
    assert original["expires_at"] - original["created_at"] == 30 * 86400
    now[0] += 29 * 86400
    service.add_messages("u", "s", "later", [{"role": "user", "content": "retry three times"}])
    assert service.search("u", "timeout", 8)
    later = service.store.state.scope_attributes(service.store.scope("u"))
    assert (later["created_at"], later["expires_at"]) == (original["created_at"], original["expires_at"])
    service.shutdown()
    restored = CodingMemoryService(cfg)
    restored.store.clock = lambda: now[0]
    restored.start()
    assert restored.search("u", "retry", 8)
    now[0] = original["expires_at"]  # inclusive expiry; no real waiting
    assert restored.search("u", "retry", 8) == []
    assert not restored._cache
    assert restored.purge_idle(30) == 1
    assert restored.store.stats()["users"] == 0
    assert all(restored.store.state._rows(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] == 0
               for table in ("scope_attributes", "requests", "memories"))
    restored.add_messages("u", "s", "first", [{"role": "user", "content": "new window"}])
    fresh = restored.store.state.scope_attributes(restored.store.scope("u"))
    assert fresh["generation"] != original["generation"]
    assert fresh["created_at"] == now[0] and fresh["expires_at"] == now[0] + 30 * 86400
    restored.shutdown()
