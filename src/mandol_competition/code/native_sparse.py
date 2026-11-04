"""Versioned, optional v4 paired projection over unchanged C0 canonical data.

The storage profile remains C0. This explicit projection/search extension has
its own immutable point identities and complete frontier. Legacy Dense points
coexist in the same owned collection, so a rollback never mistakes a partial
paired projection for the old Dense corpus. No vectors are stored in SQLite.
"""
from dataclasses import asdict
import math

import numpy as np

from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.state import StateUnavailable
from mandol_substrate.state.models import digest
from mandol_substrate.vector.codec import MinimalProjectionCodec, point_id

BASELINE = "c0-original-v1"
POLICY = "code-native-sparse-rrf100-v1"
FRONTIER = "code-native-v4-pair-v1"


def descriptor(profile):
    from mandol_competition.inference.dashscope import official_embedding_recipes
    dense, sparse = official_embedding_recipes()
    if profile.options.get("dense_recipe_hash") != dense.hash:
        raise ValueError("Native Sparse requires the frozen named-v4 C0 storage profile")
    return dict(schema="code-optional-projection-extension-v1", profile_id=POLICY,
                storage_profile_hash=profile.hash, dense_recipe=asdict(dense),
                sparse_recipe=asdict(sparse), frontier=FRONTIER,
                fusion="equal-weight-rrf", rrf_k=60, dense_k=100, sparse_k=100,
                tail="unchanged-remaining-dense-order", episode_policy="unchanged-c0")


class NativeSparseCodec(MinimalProjectionCodec):
    def __init__(self, profile, policy_hash):
        super().__init__(profile, policy_hash)
        self.extension_hash = digest(descriptor(profile))

    def metadata(self, scope, memory):
        row = super().metadata(scope, memory)
        row["physical_id"] = digest([FRONTIER, self.extension_hash, row["physical_id"]])
        row["projection_hash"] = digest([self.extension_hash, row["projection_hash"]])
        return row

    def encode(self, metadata, dense, sparse):
        row = super().encode(metadata, dense, sparse)
        if not row["splade_sparse"]:
            raise StateUnavailable("Native projection requires a nonempty sparse vector")
        if abs(float(np.linalg.norm(row["dense_vector"])) - 1) > .01:
            raise StateUnavailable("Native projection requires provider-normalized Dense")
        return row


class NativeProjection(QdrantProjectionBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec = NativeSparseCodec(self.profile, self.policy.hash)

    def gc_scope(self, scope, *, only_uids=None):
        """Erase stale revisions, retaining both current rollback generations."""
        from qdrant_client import models
        self._assert_binding(scope)
        frontier = self.state.scope_state(scope)
        legacy = MinimalProjectionCodec(self.profile, self.policy.hash)
        removed = 0
        for rows in self.iter_metadata(scope):
            canonical = self.state.get_memories(scope, [r["logical_uid"] for r in rows])
            stale = []
            for row in rows:
                if only_uids is not None and row["logical_uid"] not in only_uids:
                    continue
                if row["commit_seq"] > frontier["canonical_seq"]:
                    raise StateUnavailable("GC refuses ahead-of-canonical projection")
                memory = canonical.get(row["logical_uid"])
                keep = (memory and memory.lifecycle != "deleted" and row["physical_id"] in {
                    self.codec.metadata(scope, memory)["physical_id"],
                    legacy.metadata(scope, memory)["physical_id"]})
                if not keep:
                    stale.append(point_id(row["physical_id"]))
            if stale:
                self.client.delete(self.collection_name, models.PointIdsList(points=stale),
                                   wait=True, ordering=models.WriteOrdering.STRONG, timeout=self.timeout)
                removed += len(stale)
        return removed


def fuse(dense, sparse):
    """Frozen experimental mechanism: RRF100/60, then complete Dense tail."""
    if not sparse:
        return list(dense)
    scores = {}
    for rows in (dense[:100], sparse[:100]):
        for rank, (uid, _) in enumerate(rows, 1):
            scores[uid] = scores.get(uid, 0.) + 1 / (60 + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])) + [
        (uid, 0.) for uid, _ in dense if uid not in scores]


class NativeChannel:
    def __init__(self, store, scope, encoder, projection, ids):
        self.store, self.scope, self.encoder, self.projection = store, scope, encoder, projection
        self.ids = sorted(ids)
        self.seq = store.state.scope_state(scope)["published_seq"]
        self._check_frontier()
        rows = projection.get_projection_metadata(scope, self.ids, vectors=True)
        memories = store.state.get_memories(scope, self.ids)
        if set(rows) != set(self.ids) or set(memories) != set(self.ids):
            raise StateUnavailable("partial native Dense/Sparse corpus")
        for uid in self.ids:
            if not projection.codec.matches(rows[uid], projection.codec.metadata(scope, memories[uid])):
                raise StateUnavailable("Native projection differs from canonical revision")
            projection.codec.encode(rows[uid], rows[uid]["dense_vector"], rows[uid]["splade_sparse"])
        self.vectors = np.asarray([rows[uid]["dense_vector"] for uid in self.ids], dtype=np.float32)
        self.sparse = {uid: rows[uid]["splade_sparse"] for uid in self.ids}
        self._check_frontier()

    def _check_frontier(self):
        if (not self.store.state.projection_ready(self.scope, FRONTIER)
                or self.store.state.scope_state(self.scope)["published_seq"] != self.seq):
            raise StateUnavailable("Native Dense/Sparse corpus is not completely published")

    def search(self, query, *, top_k):
        self.projection.ensure_schema(self.scope)
        self._check_frontier()
        if not self.ids or top_k <= 0:
            return []
        dense_query, sparse_query = self.encoder.encode_query_pair(query)
        dense = sorted(zip(self.ids, map(float, self.vectors @ dense_query)),
                       key=lambda pair: (-pair[1], pair[0]))
        sparse = [(uid, sum(weight * self.sparse[uid].get(index, 0.)
                            for index, weight in sparse_query.items())) for uid in self.ids]
        if any(not math.isfinite(value) for _, value in (*dense, *sparse)):
            raise StateUnavailable("nonfinite native retrieval score")
        sparse = sorted(((uid, value) for uid, value in sparse if value > 0),
                        key=lambda pair: (-pair[1], pair[0]))
        return fuse(dense, sparse)[:top_k]
