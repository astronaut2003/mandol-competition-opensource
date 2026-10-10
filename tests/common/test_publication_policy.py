from dataclasses import replace

import pytest

from mandol_substrate.contracts import PublicationPolicy
from mandol_substrate.state import Conflict, Mutation, ProfileMismatch, StateUnavailable
from .test_publication import coordinator


def optional():
    return PublicationPolicy("canonical-lexical-optional-dense-v1", False, False)


def test_lexical_ack_does_not_call_model_or_projection(store, profile, tmp_path, scope, memory):
    c, projection = coordinator(store, profile, tmp_path, policy=optional())
    class Unavailable:
        def encode_documents(self, texts):
            raise AssertionError("Add must not call inference")
    c.encoders = Unavailable()
    result = {"success": True, "request_id": "r", "user_id": scope.user_id, "session_id": "s"}
    assert c.apply(scope, "r", "same", Mutation(memories=(memory,)), result) == result
    assert projection.writes == 0
    assert store.lexical.search(scope, "alpha")[0].logical_uid == memory.logical_uid
    assert store._rows("SELECT projection_status FROM mutations")[0]["projection_status"] == "deferred"
    assert c.apply(scope, "r", "same", Mutation(), {}) == result
    with pytest.raises(Conflict):
        c.apply(scope, "r", "different", Mutation(), {})
    assert c.project_optional(scope) is False
    assert not store.projection_ready(scope, "dense")
    assert store._rows("SELECT count FROM runtime_events")[0]["count"] == 1


def test_optional_corpus_frontier_invalidated_by_new_add(store, profile, tmp_path, scope, memory):
    c, projection = coordinator(store, profile, tmp_path, policy=optional())
    c.apply(scope, "r", "d", Mutation(memories=(memory,)), {})
    assert c.project_optional(scope)
    assert store.projection_ready(scope, "dense")
    c.apply(scope, "r2", "d2", Mutation(memories=(replace(memory, logical_uid="new"),)), {})
    assert not store.projection_ready(scope, "dense")
    assert c.project_optional(scope)
    assert store.projection_ready(scope, "dense")
    with pytest.raises(Conflict):
        store.set_projection_frontier(scope, "dense", 1)


def test_policy_binding_cannot_switch_ack_semantics(store, profile, tmp_path):
    coordinator(store, profile, tmp_path, policy=optional())
    with pytest.raises(ProfileMismatch):
        coordinator(store, profile, tmp_path, policy=PublicationPolicy("strict-v1"))


def test_optional_unknown_after_t1_recovers_without_inference(store, profile, tmp_path, scope, memory):
    def crash(point):
        if point == "after_T1":
            raise TimeoutError("unknown HTTP outcome")
    c, projection = coordinator(store, profile, tmp_path, policy=optional(), fault=crash)
    with pytest.raises(TimeoutError):
        c.apply(scope, "r", "d", Mutation(memories=(memory,)), {"success": True})
    assert store.request(scope, "r")["status"] == "pending"
    c.fault = lambda _: None
    c.recover()
    assert c.apply(scope, "r", "d", Mutation(), {}) == {"success": True}
    assert projection.writes == 0
