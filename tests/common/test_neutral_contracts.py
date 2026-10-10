from dataclasses import asdict, replace

import pytest

from mandol_substrate.contracts import ResourceIdentity, SpacePolicy
from mandol_substrate.state import Conflict, ProfileMismatch, SQLiteStateStore


def test_manifest_binding_and_explicit_empty_spaces(tmp_path, profile, scope):
    path = tmp_path / "state.sqlite3"
    identity = ResourceIdentity("test", scope.namespace, "test_v1")
    with_store = SQLiteStateStore(path, profile, space_policy=SpacePolicy(), identity=identity)
    with_store.ensure_scope(scope)
    assert with_store._rows("SELECT * FROM memory_spaces") == []
    with_store.close()
    with pytest.raises(ProfileMismatch):
        SQLiteStateStore(path, profile, space_policy=SpacePolicy(),
                         identity=replace(identity, track="other"))
    reopened = SQLiteStateStore(path, profile, space_policy=SpacePolicy(), identity=identity)
    reopened.close()


def test_preparation_replay_survives_restart_and_conflicts(tmp_path, profile, scope):
    path = tmp_path / "state.sqlite3"
    state = SQLiteStateStore(path, profile)
    assert state.ledger.reserve(scope, "request", "digest") == {}
    state.ledger.checkpoint(scope, "request", "digest", {"prepared": ["model output"]})
    state.close()
    state = SQLiteStateStore(path, profile)
    assert state.ledger.reserve(scope, "request", "digest") == {"prepared": ["model output"]}
    with pytest.raises(Conflict):
        state.ledger.reserve(scope, "request", "different")
    with pytest.raises(ValueError):
        state.ledger.checkpoint(scope, "request", "digest", {"image": "data:image/png;base64,AAAA"})
    state.begin_purge(scope, {})
    state.finish_purge(scope)
    assert state._rows("SELECT * FROM request_preparations") == []
    state.close()


def test_degradation_is_durable_without_payloads(store, scope):
    store.record_event(scope, "search", "dense", "Timeout")
    store.record_event(scope, "search", "dense", "Timeout")
    assert store._rows("SELECT count FROM runtime_events")[0]["count"] == 2
    with pytest.raises(ValueError):
        store.record_event(scope, "search", "dense", "raw response contains secret")


def test_contracts_do_not_change_frozen_semantic_serialization(profile):
    from mandol_substrate.state.models import digest

    assert profile.hash == digest(asdict(profile))
    assert "track" not in asdict(profile)
