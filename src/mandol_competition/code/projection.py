"""Optional Code Dense projection, preserving its frozen exact raw ranking.

Qdrant is the rebuildable vector owner. A disposable in-process matrix retains
the original float32 dot-product ordering after a full, verified corpus read.
The Dot collection avoids a second normalization by Qdrant changing those bits.
There is no vector data in canonical SQLite and no partial corpus is admitted.
"""

import numpy as np

from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.state import ProfileMismatch, StateUnavailable

POLICY = QdrantProjectionPolicy(schema="mandol-qdrant-projection-dot-v1", dense_metric="Dot")


class DenseEncoders:
    def __init__(self, encoder):
        self.encoder = encoder

    def encode_documents(self, texts):
        return [(vector.tolist(), {}) for vector in self.encoder.encode(texts)]


class CanonicalLexicalChannel:
    def __init__(self, store, user_id, uids):
        self.store, self.scope, self.uids = store, store.scope(user_id), sorted(uids)

    def search(self, query, *, top_k, candidate_uids=None):
        ids = self.uids if candidate_uids is None else sorted(set(candidate_uids).intersection(self.uids))
        if top_k <= 0 or not ids:
            return []
        return [(hit.logical_uid, hit.score) for hit in self.store.state.lexical.search(
            self.scope, query, limit=min(top_k, 16384), candidate_uids=ids)]


class QdrantDenseChannel:
    def __init__(self, store, scope, encoder, projection, ids):
        self.store, self.scope, self.encoder, self.projection = store, scope, encoder, projection
        if not store.state.projection_ready(scope, "dense"):
            raise StateUnavailable("Dense corpus is not fully published")
        self.seq = store.state.scope_state(scope)["published_seq"]
        self.ids = sorted(ids)
        values = projection.get_candidate_embeddings(scope, self.ids)
        if set(values) != set(self.ids):
            raise StateUnavailable("partial Dense corpus")
        self.vectors = np.asarray([values[uid] for uid in self.ids], dtype=np.float32)

    def search(self, query, *, top_k):
        self.projection.ensure_schema(self.scope)
        if (not self.store.state.projection_ready(self.scope, "dense")
                or self.store.state.scope_state(self.scope)["published_seq"] != self.seq):
            raise StateUnavailable("Dense corpus revision changed")
        if not self.ids:
            return []
        vector = (self.encoder.encode_query(query) if hasattr(self.encoder,"encode_query")
                  else self.encoder.encode([query])[0])
        scores = self.vectors @ vector
        return sorted(zip(self.ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))[:top_k]


def connect_projection(service):
    # Only construction/publication of the shared HTTP backend is global.
    # Per-user projection I/O happens after this short critical section.
    with service._lock:
        return _connect_projection(service)


def _connect_projection(service):
    store = service.store
    if service._vector is None:
        identity = store.state.identity
        from .native_sparse import POLICY as NATIVE_POLICY, NativeProjection
        native = service.settings.coding_raw_policy == NATIVE_POLICY
        factory = NativeProjection if native else QdrantProjectionBackend
        service._vector = factory(service.settings.qdrant_endpoint, store.profile,
            state=store.state, storage_id=store.state.db_uuid, identity=identity,
            collection_name=identity.collection, policy=POLICY, optional_projection=True,
            api_key=service.settings.qdrant_api_key)
        store.coordinator.projection = service._vector
        store.coordinator.codec = service._vector.codec
        if service._encoder is not None:
            store.coordinator.encoders = service._encoder if native else DenseEncoders(service._encoder)
    return service._vector


def prepare_channel(service, user_id, sources):
    store, scope = service.store, service.store.scope(user_id)
    connect_projection(service)
    from .native_sparse import POLICY as NATIVE_POLICY, FRONTIER, NativeChannel
    if service.settings.coding_raw_policy == NATIVE_POLICY:
        service._vector.ensure_schema(scope, create=True)
        # An old-source rollback may have GC'd the extension generation. Its
        # frontier alone cannot prove that all current physical rows still exist.
        actual = service._vector.get_projection_metadata(scope, sorted(sources))
        if not store.state.projection_ready(scope, FRONTIER) or set(actual) != set(sources):
            service._vector.gc_scope(scope)
            if not store.coordinator.project_optional(scope, channel=FRONTIER):
                raise StateUnavailable("optional native Dense/Sparse projection unavailable")
        return NativeChannel(store, scope, service._encoder, service._vector, sources)
    if not store.state.projection_ready(scope, "dense"):
        # Remove stale/orphan revisions left by completed canonical erasure.
        service._vector.ensure_schema(scope, create=True)
        service._vector.gc_scope(scope)
        if not store.coordinator.project_optional(scope):
            raise StateUnavailable("optional Dense projection unavailable")
    return QdrantDenseChannel(store, scope, service._encoder, service._vector, sources)
