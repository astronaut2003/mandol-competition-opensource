"""Real disposable Qdrant/SQLite; fake inference is explicitly storage-only."""
import os
import uuid

import pytest

from mandol_competition.textual.core.hlm_builder import HighLevelMemoryBuilder
from mandol_competition.textual.core.hlm_production import HighLevelMemoryPolicy
from mandol_competition.textual.core.heterogeneous_worker import typed_scope
from mandol_substrate.contracts import ResourceIdentity
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.state import CanonicalMemory, Mutation, ProfileMismatch, RetrievalProfile, Scope, SQLiteStateStore
from mandol_substrate.text import MandolTextProcessor
from tests.common.fakes import Encoders
from tests.textual.test_canonical_hlm import _provider


def test_bound_hlm_publication_restart_isolation_and_purge(tmp_path):
    endpoint = os.environ["MANDOL_TEST_QDRANT"]
    profile = RetrievalProfile("test/dense", "fixed-rev", 3, "fixed-tokenizer",
                               "test/sparse", "fixed-rev", MandolTextProcessor().fingerprint)
    identity = ResourceIdentity("textual", "textual", "derived_" + uuid.uuid4().hex)
    path = tmp_path / "state.sqlite3"
    calls = []
    state = projection = facade = None

    def open_runtime():
        state = SQLiteStateStore(path, profile, identity=identity)
        projection = QdrantProjectionBackend(endpoint, profile, state=state, storage_id=state.db_uuid,
            identity=identity, collection_name=identity.collection)
        coordinator = PublicationCoordinator(state, projection, projection.codec, Encoders())
        facade = HighLevelMemoryBuilder.from_competition(coordinator, _provider(calls),
                policy=HighLevelMemoryPolicy(hierarchical_enabled=True))
        return state, projection, coordinator, facade

    try:
        state, projection, coordinator, facade = open_runtime()
        raw = Scope("textual", "same-user")
        memory = CanonicalMemory("raw-1", "Review the regression tests before releasing.",
                    role="user", turn_index=0, metadata={"aml": True, "user_id": raw.user_id})
        coordinator.apply(raw, "raw", "raw", Mutation(memories=(memory,)), {})
        built = facade.build_session(raw, [memory.logical_uid])
        uids = built[0]["output_uids"]["hierarchical"]
        partition = typed_scope(raw, "hierarchical", facade.canonical.recipe)
        assert projection.get_projection_metadata(partition, uids)
        assert projection.collection_metadata["namespace"] == "textual"
        assert state.lexical.inspect_stats(raw)["n_docs"] == 1
        assert facade.canonical.check_consistency(raw)["status"] == "PASS"
        with pytest.raises(ProfileMismatch):
            projection.ensure_schema(Scope("code", "same-user"))
        assert projection.search_dense(Scope(partition.namespace, "other-user"), [1, 0, .5],
                                       limit=5, published_seq=0) == []
        facade.close(); projection.close(); state.close()
        state, projection, coordinator, facade = open_runtime()
        assert facade.build_session(raw, [memory.logical_uid]) == built
        assert calls == ["hierarchical"]
        assert projection.get_projection_metadata(partition, uids)
        from mandol_substrate.maintenance import ResourcePurger
        for scope in [raw, partition]:
            state.begin_purge(scope, {"purged": True})
            ResourcePurger(state, projection).complete(scope)
        assert state.get_memories(partition, uids) == {}
        assert list(projection.iter_metadata(partition)) == []
        assert state._rows("SELECT * FROM build_jobs") == []
    finally:
        if facade is not None:
            facade.close()
        if projection is not None:
            projection.client.delete_collection(identity.collection)
            projection.close()
        if state is not None:
            state.close()
