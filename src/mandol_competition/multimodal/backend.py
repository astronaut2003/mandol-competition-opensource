"""Multimodal adapters over the shared canonical/publication/retrieval substrate."""

from __future__ import annotations

import shutil
import threading
from contextlib import contextmanager, ExitStack

from mandol_substrate.gates import ScopeGates
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.retrieval import RetrievalBackend
from mandol_substrate.state import (
    CanonicalMemory,
    Conflict,
    Mutation,
    Scope,
    SQLiteStateStore,
    StateUnavailable,
)
from mandol_substrate.vector.verification import verify_projection

from mandol_substrate.representation import MemoryUnit

from .config import ConfigurationError
from .models import build_processor, load_profile
from .inference import remote_encoders, remote_reranker
from mandol_substrate.contracts import PublicationPolicy, ResourceIdentity, SpacePolicy


def _reject_inline_images(value):
    if isinstance(value, dict):
        if value.get("type") == "image_url":
            raise ValueError("canonical images must use MediaStore references")
        for item in value.values():
            _reject_inline_images(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_inline_images(item)
    elif isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        raise ValueError("inline image bytes cannot enter canonical state")


def canonical_memory(unit, user_id):
    if unit.metadata.get("user_id") != user_id:
        raise ValueError("MemoryUnit user_id differs from publication scope")
    _reject_inline_images(unit.raw_data)
    _reject_inline_images(unit.metadata)
    metadata = unit.metadata
    return CanonicalMemory(logical_uid=unit.uid, text_content=unit.raw_data["text_content"],
        raw_data=unit.raw_data, metadata=metadata, session_id=metadata.get("session_id", ""),
        request_id=metadata.get("request_id", ""), turn_index=metadata.get("message_index", 0),
        role=metadata.get("role", ""), spaces=("aml",))


class MultimodalBackend:
    """Own one SQLite database and one verified Qdrant collection.

    Per-user operation leases include VLM/media work and response restoration,
    so purge cannot race either. Shared publication leases protect canonical
    frontiers. A separate admission barrier drains all work on close/maintenance.
    SQLite's POSIX owner lock enforces a single API writer process.
    """

    def __init__(self, config, *, profile=None, encoders=None, projection_factory=None,
                 reranker=None, processor=None, fault=None, maintenance=False):
        self.config = config
        if not maintenance and (config.data_dir / "canonical.sqlite3").exists() and not (config.data_dir / "state.sqlite3").exists():
            raise ConfigurationError("legacy canonical.sqlite3 exists; run explicit import-legacy before serving")
        self.profile = profile or load_profile(config)
        from .episodes import quality_policy
        self.visual_policy = quality_policy(self.profile)
        self.fault = fault or (lambda point: None)
        self.encoders = encoders or remote_encoders(config, self.profile)
        self.reranker = reranker
        if self.reranker is None and config.rerank_method is not None:
            self.reranker = remote_reranker(config, self.profile)
        self.state = SQLiteStateStore(config.data_dir / "state.sqlite3", self.profile,
                                     processor=processor or build_processor(self.profile), fault=self.fault,
                                     space_policy=SpacePolicy(("aml",)),
                                     identity=ResourceIdentity("multimodal", config.namespace, config.qdrant_collection))
        self.projection = None
        try:
            self.projection = projection_factory(self.state, self.profile) if projection_factory else QdrantProjectionBackend(
                config.qdrant_url, self.profile, state=self.state, storage_id=self.state.db_uuid,
                collection_name=config.qdrant_collection, timeout=config.qdrant_timeout,
                identity=self.state.identity, api_key=config.qdrant_api_key)
            self.coordinator = PublicationCoordinator(self.state, self.projection, self.projection.codec,
                self.encoders, retries=config.publication_retries, verify_timeout=config.publication_timeout,
                batch_size=self.projection.policy.batch_size, fault=self.fault,
                policy=PublicationPolicy("multimodal-post-prepare-strict-v1", True, False))
            self.retrieval = RetrievalBackend(self.state, self.projection, self.coordinator,
                                              self.encoders, self.profile, reranker=self.reranker)
        except BaseException:
            if self.projection is not None:
                self.projection.close()
            self.state.close()
            raise
        self._operations, self._admission = ScopeGates(timeout=300), ScopeGates(timeout=300)
        self._run_scope = Scope("runtime", self.state.db_uuid)
        self._start_lock = threading.Lock()
        self._started = self._closed = False

    def scope(self, user_id):
        return Scope(self.config.namespace, user_id)

    def start(self):
        with self._start_lock:
            if self._closed:
                raise StateUnavailable("retrieval backend is closed")
            if self._started:
                return
            importing = self.state._rows("SELECT value FROM manifest WHERE key='legacy_import_status'")
            if importing and importing[0]["value"] != "complete":
                raise StateUnavailable("legacy import incomplete; resume import-legacy before serving")
            if (self.config.data_dir / "canonical.sqlite3").exists() and not importing:
                raise StateUnavailable("legacy data requires explicit import-legacy before serving")
            for client in (getattr(self.encoders, "dense", None), getattr(self.encoders, "sparse", None), self.reranker):
                if hasattr(client, "validate_profile"):
                    client.validate_profile()
            # Purge is roll-forward cleanup, never publication of its old outbox.
            for scope in self.state.iter_scopes():
                purge = self.state.purge_state(scope)
                if purge and purge["status"] == "pending":
                    if scope.namespace != self.config.namespace:
                        raise StateUnavailable("pending purge belongs to another modality resource owner")
                    self._complete_purge(scope)
            self.coordinator.recover()
            self.projection.ensure_schema(Scope(self.config.namespace, "__schema_probe__"), create=True)
            if not verify_projection(self.state, self.projection)["servable"]:
                raise StateUnavailable("SQLite/Qdrant divergence; explicit reconciliation required")
            self._started = True

    @contextmanager
    def user_operation(self, user_id, *, write=True):
        with self._admission.lease(self._run_scope):
            self.start()
            scope = self.scope(user_id)
            with ExitStack() as leases:
                from .profiling import span
                with span('user_scope_lease_wait'):
                    leases.enter_context(self._operations.lease(scope, write=write))
                self.state.ensure_scope(scope)
                self.state.assert_not_purging(scope)
                yield

    def replay(self, user_id, request_id, payload_digest):
        scope = self.scope(user_id)
        existing = self.state.request(scope, request_id)
        if existing is None:
            self.assert_searchable(user_id)
            return None
        if existing["payload_digest"] != payload_digest:
            raise Conflict("same request_id with different payload")
        # No repeated VLM/media/semantic generation after a committed T1.
        return self.coordinator.apply(scope, request_id, payload_digest, Mutation(), {})

    def publish(self, request, payload_digest, units, result):
        memories = tuple(canonical_memory(unit, request.user_id) for unit in units)
        mutation = Mutation(memories=memories, spaces=("aml",))
        if (self.visual_policy and self.visual_policy['episode']
                and len(memories) < self.coordinator.max_mutation_memories):
            from .episodes import prepare_navigation
            mutation = prepare_navigation(self.state, self.scope(request.user_id), request, memories)
        return self.coordinator.apply(self.scope(request.user_id), request.request_id, payload_digest,
                                      mutation, result)

    def search(self, user_id, query, *, top_k):
        retrieval = self.retrieval
        route = None
        if self.visual_policy:
            from .episodes import adaptive_profile, route as navigate
            scope = self.scope(user_id)
            if self.visual_policy['episode']:
                route = lambda view: navigate(self.state, scope, query, view, enabled=True)
            if self.visual_policy['adaptive']:
                retrieval = RetrievalBackend(self.state, self.projection, self.coordinator,
                    self.encoders, adaptive_profile(self.profile, query), reranker=self.reranker,
                    observer=self.retrieval.observer)
        results = retrieval.search(self.scope(user_id), query, top_k=top_k, candidate_route=route)
        return [(MemoryUnit(m.logical_uid, m.raw_data, m.metadata), score) for m, score in results]

    def assert_searchable(self, user_id):
        # Reject pending scopes before spending work on another VLM request.
        with self.coordinator.search_view(self.scope(user_id)):
            pass

    def _complete_purge(self, scope):
        from mandol_substrate.maintenance import ResourcePurger
        def cleanup(scope):
            root = self.config.user_root(scope.user_id)
            if root.exists():
                shutil.rmtree(root)
        ResourcePurger(self.state, self.projection, cleanup=cleanup,
            retries=self.config.publication_retries, fault=self.fault).complete(scope)

    def purge(self, user_id):
        # Also usable to clear a pending Add or resume an interrupted purge.
        with self._admission.lease(self._run_scope):
            if self._closed:
                raise StateUnavailable("retrieval backend is closed")
            scope = self.scope(user_id)
            with self._operations.lease(scope, write=True), self.coordinator.gates.lease(scope, write=True):
                self.state.ensure_scope(scope)
                count = self.state._rows("SELECT count(*) AS n FROM memories WHERE scope=? AND lifecycle<>'deleted'",
                                         (scope.key,))[0]["n"]
                report = {"user_id": user_id, "graph_evicted": False, "units_removed": count,
                          "media_dir_removed": self.config.user_root(user_id).is_dir()}
                report = self.state.begin_purge(scope, report)
                self.fault("after_purge_canonical")
                self._complete_purge(scope)
                return report

    @contextmanager
    def maintenance(self):
        with self._admission.lease(self._run_scope, write=True):
            if self._closed:
                raise StateUnavailable("retrieval backend is closed")
            yield

    def verify(self):
        with self.maintenance():
            return verify_projection(self.state, self.projection)

    def reconcile(self, *, rebuild=False, gc=False):
        """Explicit operator recovery. Published rebuild never runs on restart."""
        with self.maintenance():
            self.projection.explicit_rebuild = rebuild
            try:
                if rebuild:
                    self.projection.ensure_schema(Scope(self.config.namespace, "__rebuild__"), create=True)
                for scope in self.state.iter_scopes():
                    purge = self.state.purge_state(scope)
                    if purge and purge["status"] == "pending":
                        if scope.namespace != self.config.namespace:
                            raise StateUnavailable("purge resource owner differs")
                        self._complete_purge(scope)
                    if rebuild:
                        self.projection.ensure_schema(scope, create=True)
                self.coordinator.recover()
                for scope in self.state.iter_scopes():
                    with self.coordinator.gates.lease(scope, write=True):
                        self.coordinator.audit_scope(scope)
                        if gc:
                            self.projection.gc_scope(scope)
                report = verify_projection(self.state, self.projection)
                if not report["servable"]:
                    raise StateUnavailable("projection recovery verification failed")
                self._started = True
                return report
            finally:
                self.projection.explicit_rebuild = False

    def purge_idle(self, ttl_days):
        import time
        if not self._started or self._closed:
            return 0
        cutoff = time.time_ns() // 1_000_000 - int(ttl_days * 86400 * 1000)
        count = 0
        for scope in self.state.iter_scopes():
            intent = self.state.purge_state(scope)
            if intent and intent['status'] == 'done':
                continue
            if intent or self.state.scope_state(scope)['updated_at_ms'] < cutoff:
                self.purge(scope.user_id)
                count += 1
        return count

    def health(self):
        if self._closed or not self._started or not self.state.health() or self.state.pending_count():
            raise StateUnavailable("backend publication unavailable")
        if self.state._rows("SELECT 1 FROM scope_purges WHERE status='pending' LIMIT 1"):
            raise StateUnavailable("purge cleanup unavailable")
        self.projection.ensure_schema(self.scope("__health_probe__"))
        for client in (getattr(self.encoders, "dense", None), getattr(self.encoders, "sparse", None), self.reranker):
            if hasattr(client, "http"):
                client.http.health()
            elif hasattr(client, "health"):
                client.health()
        return {"status": "ok"}

    def close(self):
        with self._admission.lease(self._run_scope, write=True):
            if not self._closed:
                self._closed = True
                try:
                    self.projection.close()
                finally:
                    self.state.close()
                    for model in (self.encoders, self.reranker):
                        if hasattr(model, "close"):
                            model.close()
