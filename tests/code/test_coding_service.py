import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from mandol_competition.code.api.app import create_app
from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.coding_store import CapacityExceeded, CodingEvidenceStore, RequestConflict, StorageBusy
from mandol_competition.code.api.config import Settings


def settings(tmp_path, **kwargs):
    return Settings(coding_memory=True, coding_db_path=str(tmp_path / "memory.sqlite3"),
                    warmup_on_start=False, **kwargs)


def message(text="  TypeError: token refresh failed\n"):
    return {"role": "user", "content": text, "timestamp": 1700000000000}


def test_configured_card_cap_retains_eight_long_sessions_and_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("AML_CODING_PER_CARD_CHARS", "1500")
    assert Settings.from_env().coding_per_card_chars == 1500
    cfg = settings(tmp_path, coding_per_card_chars=1500, coding_evidence_chars=12000)
    service = CodingMemoryService(cfg)
    service.start()
    raw = "get_user_profile regression " + "historical source context " * 90
    for i in range(8):
        service.add_messages("packing", str(i), str(i), [message(raw)])
    cards = service.search("packing", "get_user_profile", 8)
    assert len(cards) == 8 and all(len(card["content"]) <= 1500 for card in cards)
    assert sum(len(card["content"]) for card in cards) <= 12000
    assert all(json.loads(r["payload"])["content"] == raw for r in service.store.snapshot("packing")[1])
    service.shutdown()
    restored = CodingMemoryService(cfg)
    restored.start()
    assert restored.search("packing", "get_user_profile", 8) == cards
    restored.shutdown()


def test_durable_exact_raw_idempotence_conflict_session_and_user_isolation(tmp_path):
    cfg = settings(tmp_path)
    service = CodingMemoryService(cfg)
    service.start()
    service.add_messages("alice", "session", "request", [message()])
    service.add_messages("alice", "session", "request", [message()])
    with pytest.raises(RequestConflict):
        service.add_messages("alice", "session", "request", [message("changed")])
    with pytest.raises(RequestConflict):
        service.add_messages("alice", "different-session", "request", [message()])
    first = service.search("alice", "token refresh TypeError", 8)
    service.add_messages("alice", "session", "next", [message("JWTInvalidError in refresh_token")])
    after = service.search("alice", "JWTInvalidError", 8)
    assert len(first) == len(after) == 1 and first[0]["id"] == after[0]["id"]
    assert "  TypeError: token refresh failed\n" in after[0]["content"]
    service.add_messages("bob", "session", "request", [message("bob_private_key_identifier")])
    assert service.search("nobody", "TypeError", 8) == []
    assert service.search("bob", "TypeError", 8) == []
    state, rows = service.store.snapshot("alice")
    assert len(rows) == 2 and [r["sequence"] for r in rows] == [0, 1]
    assert json.loads(rows[0]["payload"])["content"] == message()["content"]
    service.shutdown()
    restored = CodingMemoryService(cfg)
    restored.start()
    assert restored.search("alice", "JWTInvalidError", 8) == after
    restored.shutdown()


def test_concurrent_add_search_across_connections_is_atomic(tmp_path):
    a = CodingEvidenceStore(tmp_path / "journal.sqlite3")
    a.initialize()
    # Unified deployment has one SQLite owner; concurrent operations share it.
    b = a

    def write(index):
        store = a if index % 2 else b
        return store.add("u", "s", f"req-{index // 2}", [message("first"), message("second")])

    def read(_):
        _, rows = b.snapshot("u")
        assert len(rows) % 2 == 0
        assert [r["sequence"] for r in rows] == list(range(len(rows)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = [executor.submit(write, index) if index % 3 else executor.submit(read, index)
                for index in range(60)]
        for job in jobs:
            job.result()
    _, rows = a.snapshot("u")
    assert len(rows) == 60
    assert len({row["uid"] for row in rows}) == 60


def test_lock_capacity_expiry_and_schema_rejection(tmp_path):
    current = [100.0]
    store = CodingEvidenceStore(tmp_path / "journal.sqlite3", ttl_days=1, busy_timeout_ms=30,
                                max_units_per_user=1, clock=lambda: current[0])
    store.initialize()
    store.add("u", "s", "r", [message()])
    with pytest.raises(CapacityExceeded):
        store.add("u", "s", "r2", [message()])
    assert len(store.snapshot("u")[1]) == 1
    with sqlite3.connect(store.path) as writer:
        writer.execute("BEGIN IMMEDIATE")
        start = time.monotonic()
        with pytest.raises(StorageBusy):
            store.add("v", "s", "r", [message()])
        assert time.monotonic() - start < 2
        assert store.snapshot("u")[1]  # WAL readers still work
    current[0] += 86401
    assert store.snapshot("u") == (None, [])
    assert store.purge_expired() == ["u"]
    assert store.stats()["stored_units"] == 0
    store.close()
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(RuntimeError, match="schema"):
        store.initialize()


def test_projection_embedding_failure_preserves_add_and_lexical_search(tmp_path,monkeypatch):
    service = CodingMemoryService(settings(tmp_path))
    service.start()

    class BrokenEncoder:
        def index(self, documents):
            raise RuntimeError("embedding unavailable")

    service._encoder = BrokenEncoder()
    # This is a fault-injection unit test, not a Qdrant integration test. Fail
    # at the projection boundary without connecting to the default live port.
    from mandol_competition.code import projection
    def unavailable(*args,**kwargs):
        raise RuntimeError('embedding unavailable')
    monkeypatch.setattr(projection,'prepare_channel',unavailable)
    service.add_messages("u", "s", "r", [message()])
    assert service.search("u", "token refresh", 1)
    assert service.stats()["projection_fallbacks"] == 1
    assert len(service.store.snapshot("u")[1]) == 1
    service.shutdown()


def wait_ready(client):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            return
        time.sleep(.01)
    pytest.fail("Coding service did not become ready")


def test_http_contract_auth_invalid_body_topk_conflict_and_restart(tmp_path):
    cfg = settings(tmp_path, memory_system_key="test-key", max_request_bytes=1024)
    # The platform sends a memory system key in a Bearer header.
    headers = {"Authorization": "Bearer test-key"}
    payload = {"user_id": "u", "session_id": "s", "request_id": "r", "messages": [message()]}
    for run in range(2):
        with TestClient(create_app(cfg)) as client:
            wait_ready(client)
            if run == 0:
                assert client.post("/add", json=payload).status_code == 401
                response = client.post("/add", json=payload, headers=headers)
                assert response.status_code == 200 and response.json()["request_id"] == "r"
                assert client.post("/add", json=payload, headers=headers).status_code == 200
                changed = {**payload, "messages": [message("different")]}
                assert client.post("/add", json=changed, headers=headers).status_code == 409
                assert client.post("/add", json={"user_id": "u"}, headers=headers).status_code == 422
                assert client.post("/add", content=(b"x" * 600 for _ in range(2)), headers=headers).status_code == 413
                assert client.post("/search", json={"user_id": "u", "query": " ", "top_k": 1}, headers=headers).status_code == 422
            result = client.post("/search", json={"user_id": "u", "query": "TypeError", "top_k": 1}, headers=headers)
            assert result.status_code == 200 and len(result.json()["data"]) == 1
            empty = client.post("/search", json={"user_id": "missing", "query": "TypeError", "top_k": 1}, headers=headers)
            assert empty.json() == {"data": []}
