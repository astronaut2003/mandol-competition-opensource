"""Run explicitly with MANDOL_TEST_QDRANT set to a disposable 1.19.1 server."""

from dataclasses import replace
import os
import uuid

import httpx
import pytest

from mandol_substrate.contracts import ResourceIdentity, SpacePolicy, PublicationPolicy
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.external import ExternalQdrantService
from mandol_substrate.state import CanonicalMemory, Mutation, ProfileMismatch, RetrievalProfile, Scope, SQLiteStateStore, StateUnavailable
from mandol_substrate.text import MandolTextProcessor
from tests.common.fakes import Encoders


def test_three_collections_isolate_identity_revision_and_drop(tmp_path):
    endpoint = os.environ["MANDOL_TEST_QDRANT"]
    suffix = uuid.uuid4().hex
    profile = RetrievalProfile("test/dense", "fixed-rev", 3, "fixed-tokenizer",
                               "test/sparse", "fixed-rev", MandolTextProcessor().fingerprint)
    resources = []
    service = ExternalQdrantService(endpoint)
    service.start()
    try:
        for track in ("textual", "code", "multimodal"):
            identity = ResourceIdentity(track, track, f"{track}_{suffix}")
            state = SQLiteStateStore(tmp_path / track / "state.sqlite3", profile,
                                     identity=identity, space_policy=SpacePolicy(("raw",)))
            projection = QdrantProjectionBackend(endpoint, profile, state=state,
                storage_id=state.db_uuid, collection_name=identity.collection, identity=identity,
                optional_projection=track == "code")
            policy = (PublicationPolicy("optional-v1", False, False) if track == "code"
                      else PublicationPolicy("strict-v1", True, track != "multimodal"))
            coordinator = PublicationCoordinator(state, projection, projection.codec, Encoders(), policy=policy)
            scope = Scope(track, "same-user")
            memory = CanonicalMemory("same-uid", track + " evidence", spaces=("raw",))
            coordinator.apply(scope, "same-request", "digest", Mutation(memories=(memory,)), {"ok": True})
            if track == "code":
                assert coordinator.project_optional(scope)
            rows = projection.get_projection_metadata(scope, ["same-uid"])
            assert set(rows) == {"same-uid"}
            assert "text_content" not in rows["same-uid"]
            resources.append((state, projection, coordinator, scope))
            with pytest.raises(ProfileMismatch):
                QdrantProjectionBackend(endpoint, profile, state=state, storage_id=state.db_uuid,
                    collection_name="wrong_collection", identity=identity)
            with pytest.raises(ProfileMismatch):
                projection.search_dense(Scope("another-track", "same-user"), [1., 0., .5], limit=5, published_seq=1)
            assert projection.search_dense(Scope(track, "other-user"), [1., 0., .5], limit=5, published_seq=0) == []
        state, projection, coordinator, scope = resources[0]
        old = state.get_memories(scope, ["same-uid"])["same-uid"]
        stale = projection.codec.encode(projection.codec.metadata(scope, old), [1., 2., .5], {7: 1.})
        coordinator.apply(scope, "edit", "edit-digest", Mutation(memories=(replace(old, version=2, text_content="new"),)), {})
        projection.upsert_projection(scope, [stale])
        hits = projection.search_dense(scope, [1., 2., .5], limit=5, published_seq=2)
        assert len(hits) == 1 and hits[0].metadata["version"] == 2
        # A different storage owner with the correct name is still rejected.
        wrong = QdrantProjectionBackend(endpoint, profile, state=state, storage_id="wrong-storage",
            collection_name=projection.collection_name, identity=state.identity)
        try:
            with pytest.raises(ProfileMismatch):
                wrong.ensure_schema(scope)
        finally:
            wrong.close()
        projection.client.delete_collection(projection.collection_name)
        with pytest.raises(StateUnavailable):
            projection.ensure_schema(scope, create=True)
        for other_state, other_projection, _, other_scope in resources[1:]:
            assert other_projection.get_projection_metadata(other_scope, ["same-uid"])
            assert other_state.lexical.search(other_scope, "evidence")
        service.close()
        # Closing an API's connection cannot stop its shared Qdrant process.
        assert httpx.get(endpoint + "/", trust_env=False).status_code == 200
    finally:
        service.close()
        for state, projection, _, _ in resources:
            projection.client.delete_collection(projection.collection_name)
            projection.close()
            state.close()


@pytest.mark.parametrize('bad_field', ['dimension', 'sparse', 'profile'])
def test_collection_contract_fails_closed_without_reset(tmp_path, bad_field):
    from qdrant_client import models
    endpoint = os.environ['MANDOL_TEST_QDRANT']
    profile = RetrievalProfile('test/dense', 'fixed', 3, 'fixed', 'test/sparse', 'fixed', MandolTextProcessor().fingerprint)
    identity = ResourceIdentity('textual', 'contract-check', 'contract_' + uuid.uuid4().hex)
    state = SQLiteStateStore(tmp_path / 'state.sqlite3', profile, identity=identity)
    backend = QdrantProjectionBackend(endpoint, profile, state=state, storage_id=state.db_uuid,
                                     identity=identity, collection_name=identity.collection)
    try:
        metadata = backend.collection_metadata | ({'semantic_profile_hash': 'wrong'} if bad_field == 'profile' else {})
        backend.client.create_collection(identity.collection,
            vectors_config={'dense': models.VectorParams(size=2 if bad_field == 'dimension' else 3,
                distance=models.Distance.COSINE, datatype=models.Datatype.FLOAT32, on_disk=True)},
            sparse_vectors_config={'splade': models.SparseVectorParams(modifier=models.Modifier.IDF if bad_field == 'sparse' else models.Modifier.NONE,
                index=models.SparseIndexParams(on_disk=True, datatype=models.Datatype.FLOAT32))},
            shard_number=1, replication_factor=1, write_consistency_factor=1, hnsw_config=models.HnswConfigDiff(m=0),
            on_disk_payload=True, metadata=metadata)
        with pytest.raises(ProfileMismatch): backend.ensure_schema(Scope(identity.namespace, 'u'), create=True)
        assert backend.client.get_collection(identity.collection).config.metadata == metadata
    finally:
        backend.client.delete_collection(identity.collection); backend.close(); state.close()
