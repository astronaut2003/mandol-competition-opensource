from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mandol_substrate.representation import MemoryUnit
from mandol_competition.multimodal.persistence import BeginOutcome, CanonicalStore, canonical_body_hash


def unit(uid: str) -> MemoryUnit:
    return MemoryUnit(uid, {"text_content": uid}, {"user_id": "alice"})


def test_stage_is_atomic_and_invisible_until_indexed(tmp_path):
    store = CanonicalStore(tmp_path / "canonical.db")
    body_hash = canonical_body_hash({"b": 2, "a": 1})
    assert body_hash == canonical_body_hash({"a": 1, "b": 2})
    with store.hold("request", body_hash, "alice") as begin:
        assert begin.outcome is BeginOutcome.ACQUIRED
        store.stage("request", "alice", body_hash, [unit("one"), unit("two")])
        assert store.succeeded_units("alice") == []
        assert store.status_of("request") == "prepared"
        store.mark_succeeded("request", body_hash, {"success": True})
    assert [item.uid for item in store.succeeded_units("alice")] == ["one", "two"]
    with store.hold("request", body_hash) as begin:
        assert begin.outcome is BeginOutcome.REPLAY
        assert begin.response == {"success": True}
    with store.hold("request", canonical_body_hash({"a": 9})) as begin:
        assert begin.outcome is BeginOutcome.CONFLICT
    store.close()


def test_crash_retries_and_restart_only_reads_success(tmp_path):
    path = tmp_path / "canonical.db"
    store = CanonicalStore(path)
    assert store.journal_mode() == "wal"
    with store.hold("request", "hash", "alice"):
        store.stage("request", "alice", "hash", [unit("one")])
    store.close()
    store = CanonicalStore(path)
    assert store.succeeded_units("alice") == []
    with store.hold("request", "hash", "alice") as begin:
        assert begin.outcome is BeginOutcome.ACQUIRED
        store.stage("request", "alice", "hash", [unit("one")])
        store.mark_succeeded("request", "hash", {"success": True})
    assert [item.uid for item in store.succeeded_units("alice")] == ["one"]
    store.delete_user("alice")
    assert store.succeeded_units("alice") == []
    assert store.request_count() == 0
    store.close()


def test_stage_rolls_back_all_units(tmp_path):
    store = CanonicalStore(tmp_path / "canonical.db")
    with store.hold("request", "hash", "alice"):
        with pytest.raises(Exception):
            store.stage("request", "alice", "hash", [unit("one"), unit("one")])
        assert store.status_of("request") == "prepared"
        store.stage("request", "alice", "hash", [unit("one")])
        store.mark_succeeded("request", "hash", {"success": True})
    assert [item.uid for item in store.succeeded_units("alice")] == ["one"]
    store.close()


def test_failed_request_is_retryable_without_exposing_staged_units(tmp_path):
    store = CanonicalStore(tmp_path / "canonical.db")
    with store.hold("request", "hash", "alice"):
        store.stage("request", "alice", "hash", [unit("one")])
        store.fail("request", "hash", "index failure")
    assert store.succeeded_units("alice") == []
    with store.hold("request", "different", "alice") as begin:
        assert begin.outcome is BeginOutcome.CONFLICT
    with store.hold("request", "hash", "alice") as begin:
        assert begin.outcome is BeginOutcome.ACQUIRED
        store.stage("request", "alice", "hash", [unit("one")])
        store.mark_succeeded("request", "hash", {"success": True})
    assert [item.uid for item in store.succeeded_units("alice")] == ["one"]
    store.close()


def test_same_request_waits_for_first_writer(tmp_path):
    store = CanonicalStore(tmp_path / "canonical.db")
    entered = Event()
    release = Event()

    def first():
        with store.hold("request", "hash", "alice") as begin:
            assert begin.outcome is BeginOutcome.ACQUIRED
            entered.set()
            assert release.wait(5)
            store.stage("request", "alice", "hash", [unit("one")])
            store.mark_succeeded("request", "hash", {"success": True})

    def second():
        with store.hold("request", "hash", "alice") as begin:
            return begin

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_job = executor.submit(first)
        assert entered.wait(5)
        second_job = executor.submit(second)
        release.set()
        first_job.result(timeout=5)
        assert second_job.result(timeout=5).outcome is BeginOutcome.REPLAY
    store.close()
