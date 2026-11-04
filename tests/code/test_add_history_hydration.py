"""Small durable Adds must not deserialize unrelated historical payloads."""
import json

import pytest

from mandol_competition.code.api.coding_store import CapacityExceeded, CodingEvidenceStore


def test_add_keeps_sequence_capacity_replay_without_reading_historical_bodies(tmp_path, monkeypatch):
    store = CodingEvidenceStore(tmp_path / "code.sqlite3", max_units_per_user=102)
    store.initialize()
    try:
        store.add("u", "earlier", "history", [dict(role="user", content="large historical source " * 50)] * 100)
        raw = "  exact whitespace\n\ttrailing  "
        payload = [dict(role="user", content=raw)]
        original = store.state.iter_memories
        def unexpected(*args, **kwargs):
            raise AssertionError("An ordinary small Add must not hydrate the earlier corpus")
        monkeypatch.setattr(store.state, "iter_memories", unexpected)
        store.add("u", "current", "first", payload)
        store.add("u", "current", "second", payload)
        assert store.add("u", "current", "first", payload) == 1
        with pytest.raises(CapacityExceeded): store.add("u", "next", "exceeds", payload)
        store.add("v", "current", "first", payload)
        monkeypatch.setattr(store.state, "iter_memories", original)
        _, rows = store.snapshot("u")
        current = [r for r in rows if r["session_id"] == "current"]
        assert [r["sequence"] for r in current] == [0, 1]
        assert all(json.loads(r["payload"])["content"] == raw for r in current)
        assert len(rows) == 102
    finally: store.close()


def test_derived_builder_still_receives_complete_prior_sources(tmp_path):
    store = CodingEvidenceStore(tmp_path / "code.sqlite3")
    store.initialize()
    from mandol_substrate.state import Mutation
    seen = []
    class Builder:
        def prepare(self, scope, session, request, payload_hash, rows):
            seen.append([m.raw_data["text_content"] for m in rows])
            return Mutation()
    try:
        store.add("u", "earlier", "first", [dict(role="user", content="old")])
        store.experience_builder = Builder()
        store.add("u", "new", "second", [dict(role="user", content="new")])
        assert seen == [["old", "new"]]
    finally: store.close()
