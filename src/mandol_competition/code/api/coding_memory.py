"""Opt-in persistent coding memory using Mandol's source-backed episode types."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict, defaultdict

from .coding_store import CodingEvidenceStore
from mandol_substrate.state import ProfileMismatch

logger = logging.getLogger(__name__)


class CodingMemoryService:
    def __init__(self, settings):
        self.settings = settings
        from ..experience import load_candidate
        self.quality_candidate = load_candidate()
        self.experience_builder = None
        self.store = CodingEvidenceStore(settings.coding_db_path, ttl_days=settings.data_ttl_days,
                                         busy_timeout_ms=settings.coding_busy_timeout_ms,
                                         max_units_per_user=settings.max_units_per_user,
                                         code_aware=settings.coding_code_aware, namespace=settings.namespace,
                                         collection=settings.qdrant_collection,
                                         inference_contract=settings.inference_contract)
        self._ready, self._init_error = False, None
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._encoder = None
        self._vector = None
        self._projection_failures = 0
        self._search_counts = {"DENSE": 0, "LEXICAL": 0, "LEXICAL_FALLBACK": 0, "EMPTY": 0}
        self._last_retrieval_path = None

    @property
    def ready(self):
        return self._ready

    @property
    def init_error(self):
        return self._init_error

    def start(self):
        try:
            self.store.initialize()
            if self.quality_candidate and self.quality_candidate['distill']:
                from ..experience import ExperienceBuilder
                self.experience_builder = ExperienceBuilder(self)
                self.store.experience_builder = self.experience_builder
            # Import the real software types now; no model is implicitly loaded.
            from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
            _ = EngineeringEpisodeBuilder
            from mandol_competition.code.software.experience_retrieval import EpisodeRetrievalConfig, EvidencePacker
            EpisodeRetrievalConfig(mode=self.settings.coding_retrieval_mode,
                                   anchor_weight=self.settings.coding_anchor_weight,
                                   outcome_weight=self.settings.coding_outcome_weight,
                                   route_weight=self.settings.coding_route_weight)
            EvidencePacker(max_cards=self.settings.coding_max_cards, max_chars=self.settings.coding_evidence_chars,
                           per_card_chars=self.settings.coding_per_card_chars)
            if self.settings.coding_cached_users < 1:
                raise ValueError("Coding cache capacity must be positive")
            if self.settings.coding_dense_base_url:
                if self.settings.coding_dense_model_path or self.settings.coding_retrieval_mode != "raw":
                    raise ValueError("Remote coding Dense requires raw mode and no local model path")
                from .remote_embeddings import RemoteDenseEncoder,RecipeDenseEncoder
                if self.settings.inference_contract=="aml-official-eligible-v1":
                    self._encoder=RecipeDenseEncoder(self.settings.coding_dense_base_url,
                        timeout=self.settings.coding_dense_timeout_seconds)
                else:
                    self._encoder = RemoteDenseEncoder(self.settings.coding_dense_base_url,
                                                       model=self.settings.coding_dense_model,
                                                       timeout=self.settings.coding_dense_timeout_seconds)
                try:
                    self._encoder.encode(["Coding memory embedding preflight"])
                    logger.info("coding_dense backend=remote available=true dimension=1024")
                except ProfileMismatch:
                    raise
                except Exception as exc:
                    self._projection_failures += 1
                    logger.warning("coding_dense backend=remote available=false path=LEXICAL_FALLBACK error=%s",
                                   type(exc).__name__)
            elif self.settings.coding_dense_model_path:
                raise ValueError("Code API requires external HTTP Dense; local model loading is disabled")
            self._ready = True
        except Exception as exc:
            self._init_error = type(exc).__name__
            raise

    def _require_ready(self):
        if not self._ready:
            raise RuntimeError("Coding memory service is not ready")

    def add_messages(self, user_id, session_id, request_id, messages):
        self._require_ready()
        # Add success acknowledges a durable raw transaction. Projection/model
        # failures during Search cannot undo or overwrite this source of truth.
        return self.store.add(user_id, session_id, request_id, messages)

    def _projection(self, user_id, state, rows):
        from mandol_substrate.representation import MemoryUnit
        from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
        from mandol_competition.code.software.experience_retrieval import EpisodeRetriever, EpisodeRetrievalConfig

        key = (user_id, state)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            sources, sessions = {}, defaultdict(list)
            for row in rows:
                payload = json.loads(row["payload"])
                unit = MemoryUnit(uid=row["uid"], raw_data={"text_content": payload["content"]},
                                  metadata={"user_id": user_id, "session_id": row["session_id"],
                                            "request_id": row["request_id"], "role": payload["role"],
                                            "ts_ms": payload.get("timestamp"), "sequence": row["sequence"]})
                sources[unit.uid] = unit
                sessions[row["session_id"]].append(unit)
            episodes = [EngineeringEpisodeBuilder().build(units, user_id=user_id, session_id=session)
                        for session, units in sessions.items()]
            if self.experience_builder:
                episodes = self.experience_builder.overlay(user_id, episodes, sources)
            channel = None
            if self._encoder is not None:
                try:
                    from ..projection import prepare_channel
                    channel = prepare_channel(self, user_id, sources)
                except ProfileMismatch:
                    raise
                except Exception as exc:
                    self._projection_failures += 1
                    logger.warning("Coding index embedding failed (%s); path=LEXICAL_FALLBACK", type(exc).__name__)
            config = EpisodeRetrievalConfig(mode=self.settings.coding_retrieval_mode,
                                            code_aware=self.settings.coding_code_aware,
                                            router=self.settings.coding_router,
                                            evidence_needs=self.settings.coding_evidence_needs,
                                            anchor_weight=self.settings.coding_anchor_weight,
                                            outcome_weight=self.settings.coding_outcome_weight,
                                            route_weight=self.settings.coding_route_weight)
            if self.quality_candidate and self.quality_candidate.get('anchor_policy') == 'episode-anchor-idf-v1':
                from dataclasses import replace
                config = replace(config,anchor_weight=.25,distinctive_anchors=True)
            from ..projection import CanonicalLexicalChannel
            retriever = EpisodeRetriever(episodes, sources, user_id=user_id, config=config, raw_channel=channel,
                raw_index=CanonicalLexicalChannel(self.store, user_id, sources))
            for old in [old for old in self._cache if old[0] == user_id]:
                del self._cache[old]
            # A transient index failure must not pin this user to lexical until
            # another Add or process restart. The next Search retries projection.
            if channel is not None or self._encoder is None:
                self._cache[key] = (retriever, sources)
            while len(self._cache) > self.settings.coding_cached_users:
                self._cache.popitem(last=False)
            return retriever, sources

    def search(self, user_id, query, top_k):
        # Profiling showed named-v4 query I/O dominated the global cache lock.
        # Only an already built complete frozen Dense channel qualifies for
        # prefetch. The original serialized path handles cold projection,
        # publication changes, optional-channel errors and all experiments.
        encoder = None
        if (query.strip() and top_k > 0 and self._encoder is not None
                and self.settings.inference_contract == 'aml-official-eligible-v1'
                and not self.quality_candidate):
            with self._lock:
                state, _ = self.store.snapshot(user_id)
                cached = self._cache.get((user_id, state))
                if cached is not None and cached[0].raw_channel is not None:
                    encoder = self._encoder
        if encoder is not None:
            from .remote_embeddings import prepared_dense_query
            with prepared_dense_query(encoder, query):
                return self._serialized_search(user_id, query, top_k)
        return self._serialized_search(user_id, query, top_k)

    def _serialized_search(self, user_id, query, top_k):
        # Serialize projection/search diagnostics along with the existing cache.
        # Adds retain independent short SQLite transactions.
        started = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            try:
                return self._search(user_id, query, top_k)
            finally:
                logger.info("coding_lock wait_ms=%.3f held_ms=%.3f",
                            (acquired - started) * 1000, (time.perf_counter() - acquired) * 1000)

    def _record_search(self, path):
        self._last_retrieval_path = path
        self._search_counts[path] += 1
        logger.info("coding_retrieval path=%s", path)

    def _search(self, user_id, query, top_k):
        from mandol_competition.code.software.experience_retrieval import EvidencePacker

        self._require_ready()
        if not query.strip() or top_k <= 0:
            return []
        started = time.perf_counter()
        # Revision-aware derived caches must also observe canonical maintenance
        # mutations that do not increment the legacy Add attribute revision.
        state, rows = self.store.snapshot(user_id, include_frontier=bool(self.experience_builder))
        snapshot_done = time.perf_counter()
        if state is None:
            # Do not keep expired raw evidence in a previous in-process index.
            with self._lock:
                for key in [key for key in self._cache if key[0] == user_id]:
                    del self._cache[key]
            self._record_search("EMPTY")
            return []
        retriever, sources = self._projection(user_id, state, rows)
        if self.quality_candidate and self.quality_candidate['route'] == 'gated_hybrid':
            # A query-only engineering intent gate; no generation or target
            # solution features. The raw lane and best-raw-rank rule stay intact.
            import re
            from dataclasses import replace
            engineering = bool(re.search(r'\b(?:fix|bug|error|exception|fail|implement|feature|support|refactor|regression|why|how)\w*\b|修复|报错|实现', query, re.I))
            retriever.config = replace(retriever.config, mode='hybrid' if engineering else 'raw')
        projection_done = time.perf_counter()
        before = retriever.fallback_count
        hits = retriever.search(query, top_k=min(top_k, self.settings.coding_max_cards))
        retrieval_done = time.perf_counter()
        dense_requested = bool(self._encoder is not None or self.settings.coding_dense_model_path
                               or self.settings.coding_dense_base_url)
        path = ("DENSE" if retriever.raw_channel is not None and retriever.fallback_count == before
                else "LEXICAL_FALLBACK" if dense_requested else "LEXICAL")
        self._record_search(path)
        if path == "LEXICAL_FALLBACK":
            self.store.state.record_event(self.store.scope(user_id), "search", "dense", "Unavailable")
        result = EvidencePacker(max_cards=self.settings.coding_max_cards,
                              max_chars=self.settings.coding_evidence_chars,
                              per_card_chars=self.settings.coding_per_card_chars,
                              include_derived=bool(self.experience_builder)).pack(hits, sources, query=query, top_k=top_k)
        logger.info("coding_timing rows=%d snapshot_ms=%.3f projection_ms=%.3f retrieval_ms=%.3f packing_ms=%.3f",
                    len(rows), (snapshot_done - started) * 1000, (projection_done - snapshot_done) * 1000,
                    (retrieval_done - projection_done) * 1000, (time.perf_counter() - retrieval_done) * 1000)
        return result

    def purge_idle(self, ttl_days):
        # Expiry is bounded from initial ingestion, never extended by Search.
        # Clear transient evidence even if a WAL checkpoint needs retry.
        with self._lock:
            self._cache.clear()
            count = len(self.store.purge_expired())
            self._retry_purge_projection()
            return count

    def _retry_purge_projection(self):
        state = self.store.state
        pending = {row['scope'] for row in state._rows("SELECT scope FROM projection_frontiers WHERE error_class='PurgePending'")}
        if not pending:
            return
        initialized = bool(state._rows("SELECT 1 FROM manifest WHERE key='projection_collection_initialized'"))
        for scope in state.iter_scopes():
            if scope.key not in pending:
                continue
            with self.store.operations.lease(scope, write=True), self.store.coordinator.gates.lease(scope, write=True):
                try:
                    if initialized:
                        from ..projection import connect_projection
                        projection = connect_projection(self)
                        projection.ensure_schema(scope)
                        projection.gc_scope(scope)
                        for rows in projection.iter_metadata(scope):
                            memories = state.get_memories(scope, [row['logical_uid'] for row in rows])
                            for row in rows:
                                memory = memories.get(row['logical_uid'])
                                if memory is None or not projection.codec.matches(row, projection.codec.metadata(scope, memory)):
                                    raise RuntimeError('purge projection cleanup remains unverified')
                    state.mark_projection_degraded(scope, 'dense', 'ProjectionNotBuilt')
                except ProfileMismatch:
                    raise
                except Exception as exc:
                    state.record_event(scope, 'purge', 'dense', type(exc).__name__)

    def health(self):
        self._require_ready()
        if not self.store.state.health() or self.store.state.pending_count():
            from mandol_substrate.state import StateUnavailable
            raise StateUnavailable("canonical publication unavailable")
        return True  # Dense is optional by policy.

    def stats(self):
        with self._lock:
            remote = self._encoder.stats() if hasattr(self._encoder, "stats") else None
            return {"backend": "coding-sqlite", "retrieval_mode": self.settings.coding_retrieval_mode,
                    "model_ready": remote["available"] if remote is not None else self._encoder is not None,
                    "projection_fallbacks": self._projection_failures,
                    "last_retrieval_path": self._last_retrieval_path,
                    "search_counts": dict(self._search_counts), "remote_embedding": remote,
                    **self.store.stats()}

    def shutdown(self):
        with self._lock:
            self._ready = False
            self._cache.clear()
            if hasattr(self._encoder, "close"):
                self._encoder.close()
            if self.experience_builder:
                self.experience_builder.close()
            self._encoder = None
            if self._vector is not None:
                self._vector.close()
                self._vector = None
            self.store.close()
        # Canonical data survives restart; only retention deletes it.
