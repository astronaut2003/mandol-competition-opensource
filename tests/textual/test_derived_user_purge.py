import threading
from types import SimpleNamespace

import pytest

from mandol_competition.textual.api.competition import CompetitionMemoryService
from mandol_substrate.contracts import DerivedNamespaceBinding, ResourceIdentity
from mandol_substrate.state import CanonicalMemory, Mutation, Scope, SQLiteStateStore, StateUnavailable
from mandol_substrate.state.models import digest
from tests.common.test_publication import coordinator


@pytest.mark.parametrize("cleanup_error", [RuntimeError, StateUnavailable])
def test_user_purge_resumes_all_retained_recipes_without_touching_another_user(tmp_path, profile, cleanup_error):
    identity = ResourceIdentity("textual", "textual", "textual_test")
    state = SQLiteStateStore(tmp_path / "state.sqlite3", profile, identity=identity)
    c, projection = coordinator(state, profile, tmp_path)
    projection.gc_scope = lambda scope: projection._write(scope, {})
    projection.iter_metadata = lambda scope: iter(projection._read(scope).values())
    raw_a, raw_b = Scope("textual", "A"), Scope("textual", "B")
    derived = []
    for recipe in ("old-recipe", "new-recipe"):
        namespace = "derived:" + digest(recipe)
        state.authorize_derived_namespace(DerivedNamespaceBinding(namespace, "textual", identity.hash,
                                             digest(recipe), "summary"))
        derived.append(Scope(namespace, "A"))
    for scope in (raw_a, raw_b, *derived):
        c.apply(scope, "r", "body", Mutation(memories=(CanonicalMemory("m", "User-specific evidence"),)), {})
    service = object.__new__(CompetitionMemoryService)
    service.settings = SimpleNamespace(evaluation_namespace="textual")
    service._ready = True; service._admission = threading.BoundedSemaphore(1)
    service.hlm_builder = None
    service.state, service.coordinator, service.projection = state, c, projection
    failure = []
    def cleanup(scope):
        if scope == raw_a and not failure:
            failure.append(True)
            raise cleanup_error("interrupted after raw projection deletion")
    service.graphs = SimpleNamespace(evict=cleanup)
    try:
        # A fault-injected cleanup exception propagates unchanged. Recovery
        # must preserve the same durable intent for either exception category.
        with pytest.raises(cleanup_error, match="interrupted after raw projection deletion"):
            service.purge("A")
        assert state.get_memories(raw_a, ["m"]) == {}
        assert all(state.get_memories(s, ["m"]) for s in derived)
        assert state.purge_state(raw_a)["status"] == "pending"
        # TTL recovery must include scopes for old recipes even when no HLM
        # builder is enabled in this process. Another user's data stays intact.
        assert service.purge_idle(30) == 1
        assert all(state.get_memories(s, ["m"]) == {} for s in (raw_a, *derived))
        assert all(list(projection.iter_metadata(s)) == [] for s in (raw_a, *derived))
        assert state.get_memories(raw_b, ["m"])
        assert list(projection.iter_metadata(raw_b))
        assert service.purge_idle(30) == 0
    finally:
        state.close()
