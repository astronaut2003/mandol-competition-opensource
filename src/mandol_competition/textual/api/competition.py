"""Competition orchestration. Never instantiates SemanticMap or legacy indexes."""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime

from mandol_competition.textual.core.context_policy import ContextPolicy
from mandol_competition.textual.core.graph import semantic_neighbors
from mandol_competition.textual.core.models import RetrievalEncoders, ServingProfile, VllmModelClient
from mandol_competition.textual.core.profile import load_profile
from mandol_competition.textual.core.query_policy import QueryPolicy
from mandol_competition.textual.core.reranker_client import create_reranker_client
from mandol_competition.textual.core.reranker_execution import RerankerExecutionPolicy
from mandol_competition.textual.core.retrieval import CompetitionRetriever
from mandol_competition.textual.core.return_policy import ReturnPolicy
from mandol_competition.textual.core.splade import MandolSpladeEncoder
from mandol_substrate.state import (
    CanonicalMemory,
    Mutation,
    Scope,
    StateUnavailable,
)
from mandol_substrate.state.models import digest
from mandol_competition.observability.diagnostics import diagnostic_call, diagnostic_stage


def _generation_selector(settings):
    from mandol_competition.inference.evidence_selector import EvidenceSelector
    from mandol_competition.inference.generation import generation_route
    route = generation_route(settings.selector_endpoint, 'gpt-4o-mini',
                             credential_env=settings.selector_api_key_env)
    if not route.endpoint:
        raise StateUnavailable('Explicit mini selector endpoint/credential required')
    try:
        key = route.key()
    except ValueError:
        raise StateUnavailable('Explicit mini selector endpoint/credential required') from None
    return EvidenceSelector(route.endpoint, api_key=key, timeout=90)


class CompetitionMemoryService:
    def __init__(self, settings):
        settings.validate()
        self.settings = settings
        self._ready = False
        self._init_error = None
        self._resources = []
        self._admission = threading.BoundedSemaphore(8)
        self._maintenance_stop = threading.Event()
        self._maintenance_thread = None
        self._maintenance_error = None
        self.query_policy = None
        self.return_policy = None
        self.reranker_execution_policy = None
        self.neighbor_policy = None
        self.context_policy = None
        self.run = None
        # Explicit dependency injection only. Frozen packaged profiles leave
        # derived navigation disabled; no generative Search or implicit builder.
        self.navigation = None
        self.hlm_builder = None
        self.answer_utility_policy = None
        self.evidence_selector = None
        self.observation_order_policy = None

    @property
    def ready(self):
        return self._ready

    @property
    def init_error(self):
        return self._init_error

    def _remote_reranker_replicas(self, endpoints, *, endpoint_policy):
        """Default frozen Querit clients; explicit test compositions may inject a
        different, independently bound model without impersonating Querit.
        """
        from mandol_competition.inference.dashscope_rerank import BACKEND, from_environment
        settings = getattr(self, 'settings', None)
        backend = getattr(settings, 'reranker_backend', 'profile')
        if backend == BACKEND:
            return [from_environment(endpoint, credential_env=settings.reranker_credential_env,
                                     timeout=self.profile.options['native_reranker_request_timeout'])
                    for endpoint in endpoints]
        if backend != 'profile':
            raise ValueError('unknown Textual reranker backend')
        from mandol_competition.textual.core.remote_inference import RemoteQueritClient
        return [RemoteQueritClient(endpoint, self.reranker_runtime_options()["reranker_serving"],
            timeout=self.profile.options["native_reranker_request_timeout"],
            endpoint_policy=endpoint_policy) for endpoint in endpoints]

    def start(self):
        try:
            self.settings.bind_run()
            self.observation_order_policy = None
            metadata = getattr(self.settings, "quality_candidate", None) or {}
            if metadata.get("observation_order_policy"):
                from ..core.observation_order import ObservationOrderPolicy
                self.observation_order_policy = ObservationOrderPolicy(**metadata["observation_order_policy"])
            self.profile = load_profile(self.settings.competition_profile)
            if self.settings.competition_return_policy:
                self.return_policy = ReturnPolicy.load(
                    self.settings.competition_return_policy
                )
                self.return_policy.validate(self.profile)
            if self.settings.competition_context_policy:
                self.context_policy = ContextPolicy.load(
                    self.settings.competition_context_policy
                )
                self.context_policy.validate(self.profile)
            if self.settings.competition_query_policy:
                self.query_policy = QueryPolicy.load(
                    self.settings.competition_query_policy
                )
                self.query_policy.effective_profile(self.profile)
            options = self.profile.options
            if self.settings.competition_reranker_execution_policy:
                self.reranker_execution_policy = RerankerExecutionPolicy.load(
                    self.settings.competition_reranker_execution_policy
                )
                self.reranker_execution_policy.effective_options(self.profile)
            if self.settings.qdrant_mode == "external":
                from ..runtime import ExternalCompetitionRun
                self.run = ExternalCompetitionRun(self.settings)
            else:
                from mandol_competition.textual.core.run_storage import CompetitionRun
                self.run = CompetitionRun(self.settings.competition_run_root, encoders=None,
                    binary=self.settings.qdrant_binary, defer_recovery=True)
            self._resources.append(self.run)
            if self.run.profile.hash != self.profile.hash:
                raise StateUnavailable("API profile differs from dataset/run profile")
            self.state, self.projection, self.graphs = (
                self.run.state,
                self.run.vector,
                self.run.graphs,
            )
            official = self.settings.inference_contract == "aml-official-eligible-v1"
            dense_profile = None if official else ServingProfile(**options["embedding_serving"])
            if not official and (
                dense_profile.model,
                dense_profile.revision,
                dense_profile.dimension,
            ) != (
                self.profile.embedding_model,
                self.profile.embedding_revision,
                self.profile.embedding_dimension,
            ):
                raise StateUnavailable("embedding profile mismatch")
            if self.settings.inference_contract in {"vpc-vllm-029-v1", "vllm-http-v1", "aml-official-eligible-v1"}:
                from mandol_competition.textual.core.remote_inference import remote_encoders

                if official:
                    from mandol_competition.inference.clients import HTTPJointEncoders, HTTPEmbeddingClient, HTTPSparseEncoderClient
                    from mandol_competition.inference.dashscope import official_embedding_recipes
                    dense_recipe, sparse_recipe = official_embedding_recipes()
                    dense = HTTPEmbeddingClient(self.settings.embedding_url,dense_recipe,timeout=60)
                    self._resources.append(dense)
                    sparse = HTTPSparseEncoderClient(self.settings.splade_url,sparse_recipe,timeout=60)
                    self._resources.append(sparse)
                    dense.validate_profile(); sparse.validate_profile()
                    self.encoders = HTTPJointEncoders(dense,sparse)
                else:
                    self.encoders = remote_encoders(
                        self.profile, self.settings.embedding_url, self.settings.splade_url,
                        endpoint_policy="vpc" if self.settings.inference_contract == "vpc-vllm-029-v1" else "http",
                    )
                self.dense = self.encoders.dense
                self.splade = self.encoders.sparse if official else self.encoders.splade
                if not official:
                    self._resources.extend((self.dense, self.splade))
                from mandol_competition.inference.replicas import replica_endpoints, create_pool
                endpoints = replica_endpoints('textual', self.settings.reranker_url,
                                             self.settings.reranker_pool.get('endpoints'))
                replicas = self._remote_reranker_replicas(endpoints,
                    endpoint_policy="vpc" if self.settings.inference_contract == "vpc-vllm-029-v1" else "http")
                self.reranker = create_pool(replicas, self.settings.reranker_pool)
                self._resources.append(self.reranker)
                self.reranker.validate_profile()
            else:
                self.dense = VllmModelClient(self.settings.embedding_url, dense_profile)
                self._resources.append(self.dense)
                self.reranker = create_reranker_client(
                    self.settings.reranker_url, self.reranker_runtime_options()
                )
                self._resources.append(self.reranker)
                from mandol_competition.textual.core.local_models import load_snapshot_tokenizer

                for client in (self.dense, self.reranker):
                    tokenizer = load_snapshot_tokenizer(
                        client.profile.model,
                        client.profile.tokenizer_revision,
                        cache_dir=self.settings.model_cache,
                    )
                    client.validate_profile(tokenizer)
                self.splade = MandolSpladeEncoder(
                    self.profile.splade_model,
                    self.profile.splade_revision,
                    cache_dir=self.settings.model_cache,
                    device="cpu",
                )
                self.encoders = RetrievalEncoders(self.dense, self.splade)
            self.run.finish_startup(self.encoders)
            self.coordinator = self.run.coordinator
            self.retriever = CompetitionRetriever(
                self.state,
                self.projection,
                self.profile,
                query_policy=self.query_policy,
            )
            self.dense.health()
            self.reranker.health()
            self.splade.health()
            self._ready = True
            utility=(getattr(self.settings,"quality_candidate",None) or {}).get("answer_utility_policy")
            if utility:
                from ..core.answer_utility import AnswerUtilityPolicy
                self.answer_utility_policy=AnswerUtilityPolicy(**utility)
                if self.answer_utility_policy.selector!='none':
                    self.evidence_selector=_generation_selector(self.settings)
                    self._resources.append(self.evidence_selector)
            if self.settings.experimental_hlm_policy:
                import json
                from pathlib import Path

                from mandol_competition.textual.core.derivation_provider import DerivationModelBridge
                from mandol_competition.textual.core.hlm_production import (
                    HighLevelMemoryPolicy,
                    ProductionHLMProvider,
                    ProductionHLMRecipe,
                )

                value = json.loads(
                    Path(self.settings.experimental_hlm_policy).read_text()
                )
                if value.get("schema") in {"textual-procedure-navigation-v1", "textual-procedure-user-bound-navigation-v4"}:
                    from ..core.procedure_memory import (ProcedureProvider, ProcedureRecipe,
                        UserBoundProcedureProvider, UserBoundProcedureRecipe)
                    if (set(value) != {"schema", "base_storage_profile_hash", "recipe_fingerprint"}
                        or value["base_storage_profile_hash"] != self.profile.hash):
                        raise ValueError("Explicit procedure binding differs from run")
                    bound_user = value["schema"] == "textual-procedure-user-bound-navigation-v4"
                    recipe = (UserBoundProcedureRecipe if bound_user else ProcedureRecipe)(self.profile.hash)
                    if value["recipe_fingerprint"] != recipe.fingerprint:
                        raise ValueError("Procedure recipe fingerprint differs")
                    bridge = DerivationModelBridge(self.run.root / "logs/procedure-provider.jsonl",
                        provider_factory=UserBoundProcedureProvider if bound_user else ProcedureProvider, max_tokens=4096)
                    self._resources.append(bridge)
                    self.configure_procedure(recipe, bridge)
                elif value.get("schema") == "textual-event-state-navigation-v3":
                    from ..core.event_state_memory import EventStateRecipe
                    if (set(value) != {"schema", "base_storage_profile_hash", "policy", "recipe_fingerprint"}
                            or value["base_storage_profile_hash"] != self.profile.hash):
                        raise ValueError("Explicit event-state binding differs from run")
                    policy = HighLevelMemoryPolicy(**value["policy"])
                    recipe = EventStateRecipe(self.profile.hash, policy=policy)
                    if value["recipe_fingerprint"] != recipe.fingerprint:
                        raise ValueError("Event-state recipe fingerprint differs")
                    bridge = DerivationModelBridge(self.run.root / "logs/event-state-provider.jsonl",
                        provider_factory=ProductionHLMProvider)
                    self._resources.append(bridge)
                    self.configure_hlm(policy, bridge, recipe=recipe)
                elif value.get("schema") == "memory-architecture-v3":
                    from mandol_competition.textual.core.contextual_base import (
                        ArchitectureV3Provider,
                    )
                    from mandol_competition.textual.core.memory_architecture_v3 import (
                        load_architecture_binding,
                    )

                    policy = load_architecture_binding(value, self.profile.hash)
                    bridge = DerivationModelBridge(
                        self.run.root / "logs/memory-v3-provider.jsonl",
                        provider_factory=ArchitectureV3Provider,
                    )
                    self._resources.append(bridge)
                    self.configure_memory_v3(policy, bridge)
                else:
                    if (
                        set(value)
                        != {"base_storage_profile_hash", "policy", "recipe_fingerprint"}
                        or value["base_storage_profile_hash"] != self.profile.hash
                    ):
                        raise ValueError(
                            "Explicit experimental HLM binding differs from run"
                        )
                    policy = HighLevelMemoryPolicy(**value["policy"])
                    recipe = ProductionHLMRecipe(self.profile.hash, policy=policy)
                    if value["recipe_fingerprint"] != recipe.fingerprint:
                        raise ValueError("Experimental HLM recipe fingerprint mismatch")
                    bridge = DerivationModelBridge(
                        self.run.root / "logs/hlm-provider.jsonl",
                        provider_factory=ProductionHLMProvider,
                    )
                    self._resources.append(bridge)
                    self.configure_hlm(policy, bridge)
            self._maintenance_stop.clear()
            if (getattr(self.settings, "quality_candidate", None) or {}).get("evidence_packaging_policy"):
                if self.navigation is None or not hasattr(self.navigation, "semantic_mode"):
                    raise StateUnavailable("Evidence packaging requires an explicitly bound HLM navigation recipe")
            self._maintenance_thread = threading.Thread(
                target=self._maintain, name="competition-projection-gc", daemon=True
            )
            self._maintenance_thread.start()
        except Exception as exc:
            self._init_error = f"Initialization failed ({type(exc).__name__})"
            self._ready = False
            self.shutdown()
            raise

    @contextmanager
    def admission(self):
        if not self._ready:
            raise StateUnavailable("competition service not ready")
        if not self._admission.acquire(timeout=5):
            raise StateUnavailable("API backpressure")
        try:
            yield
        finally:
            self._admission.release()

    def scope(self, user_id):
        return Scope(self.settings.evaluation_namespace, user_id)

    def configure_hlm(self, policy, provider, *, recipe=None):
        """Explicit canonical opt-in; packaged source_b1 never invokes this."""
        from mandol_competition.textual.core.hlm_builder import HighLevelMemoryBuilder
        from mandol_competition.textual.core.evidence_navigation import NavigationPolicy
        from mandol_competition.textual.core.gates import ScopeGates
        from mandol_competition.textual.core.heterogeneous_worker import typed_scope
        from mandol_competition.textual.core.navigation_runtime import DerivedRoute
        from mandol_competition.textual.core.semantic_evidence import SemanticEvidenceNavigation

        if not self.ready:
            raise StateUnavailable("Configure HLM only after run/model preflight")
        if self.hlm_builder is not None:
            self.hlm_builder.close()
        if recipe is not None and recipe.policy != policy:
            raise ValueError("Query navigation and bound Add recipe policy differ")
        self.hlm_builder = HighLevelMemoryBuilder.from_competition(
            self.coordinator, provider, **({"recipe": recipe} if recipe is not None else {"policy": policy})
        )
        self._hlm_gates = ScopeGates(timeout=600, max_inflight_scopes=16)
        self._resources.append(self.hlm_builder)
        runtime = self.hlm_builder.canonical
        self.navigation = None
        packaging = (getattr(self.settings, "quality_candidate", None) or {}).get("evidence_packaging_policy")
        if packaging:
            from ..core.evidence_packaging import EvidencePackagingPolicy
            packaging = EvidencePackagingPolicy(**packaging)
        if policy.navigation_enabled or policy.semantic_evidence_enabled:
            routes = [
                DerivedRoute(
                    k,
                    typed_scope(
                        self.scope("binding-only"), k, runtime.recipe
                    ).namespace,
                    runtime.eligible,
                    historical_eligible=runtime.eligible_history
                        if k == "profile" and self.settings.experimental_quality_profile else None,
                )
                for k in runtime.kinds()
            ]
            self.navigation = SemanticEvidenceNavigation(
                self,
                NavigationPolicy(runtime.kinds(), version="canonical_raw_navigation_candidate_v2"
                    if self.settings.experimental_quality_profile else "canonical_raw_navigation_v1"),
                routes,
                semantic_evidence_enabled=policy.semantic_evidence_enabled or packaging is not None,
                selected_k=packaging.selected_k if packaging else 10,
                max_context_chars=packaging.max_context_chars if packaging else 24000,
            )
        return self.hlm_builder

    def configure_procedure(self, recipe, provider):
        """Explicit extractive-rule research arm; default production is unchanged."""
        from ..core.hlm_builder import HighLevelMemoryBuilder
        from ..core.evidence_navigation import NavigationPolicy
        from ..core.gates import ScopeGates
        from ..core.heterogeneous_worker import typed_scope
        from ..core.navigation_runtime import DerivedRoute
        from ..core.procedure_memory import ProcedureNavigation
        if not self.ready or recipe.storage_profile_hash != self.profile.hash:
            raise StateUnavailable("Procedure requires ready matching canonical runtime")
        if self.hlm_builder is not None:
            self.hlm_builder.close()
        self.hlm_builder = HighLevelMemoryBuilder.from_procedure(self.coordinator, provider, recipe=recipe)
        self._resources.append(self.hlm_builder)
        self._hlm_gates = ScopeGates(timeout=600, max_inflight_scopes=16)
        runtime = self.hlm_builder.canonical
        routes = [DerivedRoute("procedure", typed_scope(self.scope("binding-only"), "procedure", recipe).namespace,
                               runtime.eligible)]
        self.navigation = ProcedureNavigation(self, NavigationPolicy(("procedure",),
            version="canonical_raw_navigation_candidate_v2"), routes)
        return self.hlm_builder

    def _hlm_lease(self, user_id, *, write):
        return (
            self._hlm_gates.lease(self.scope(user_id), write=write)
            if getattr(self, "hlm_builder", None) is not None
            else nullcontext()
        )

    def configure_memory_v3(self, policy, provider):
        """Explicit production API opt-in; frozen packaged baseline stays OFF."""
        from mandol_competition.textual.core.hlm_builder import HighLevelMemoryBuilder
        from mandol_competition.textual.core.contextual_retrieval import ArchitectureV3Navigation
        from mandol_competition.textual.core.evidence_navigation import NavigationPolicy
        from mandol_competition.textual.core.gates import ScopeGates
        from mandol_competition.textual.core.heterogeneous_worker import typed_scope
        from mandol_competition.textual.core.navigation_runtime import DerivedRoute

        if not self.ready:
            raise StateUnavailable("Configure V3 only after preflight")
        if self.hlm_builder is not None:
            self.hlm_builder.close()
        self.hlm_builder = HighLevelMemoryBuilder.from_competition_v3(
            self.coordinator, provider, policy=policy
        )
        self._resources.append(self.hlm_builder)
        self._hlm_gates = ScopeGates(timeout=600, max_inflight_scopes=16)
        runtime = self.hlm_builder.canonical
        routes = [DerivedRoute(k, typed_scope(self.scope("binding-only"), k,
                  runtime.recipe).namespace, runtime.eligible) for k in runtime.kinds()]
        self.navigation = None
        if policy.contextual_enabled or policy.navigation_enabled or policy.semantic_evidence_enabled:
            self.navigation = ArchitectureV3Navigation(
                self, NavigationPolicy(runtime.kinds()), routes,
                contextual=self.hlm_builder.contextual, architecture_policy=policy,
            )
        return self.hlm_builder

    def add_messages(self, user_id, session_id, request_id, messages):
        with self.admission(), self._hlm_lease(user_id, write=True):
            if not 0 < len(messages) <= 1024:
                raise ValueError("invalid bounded Add size")
            scope = self.scope(user_id)
            payload_digest = digest([user_id, session_id, request_id, messages])
            memories = []
            for ordinal, msg in enumerate(messages):
                content = msg["content"].strip()
                if not content:
                    raise ValueError("empty Add content")
                memories.append(
                    CanonicalMemory(
                        scope.memory_uid(session_id, request_id, ordinal),
                        content,
                        raw_data={"text_content": content},
                        metadata={
                            "aml": True,
                            "user_id": user_id,
                            "session_id": session_id,
                            "request_id": request_id,
                            "role": msg["role"],
                            "turn_index": ordinal,
                            "ts_ms": msg.get("timestamp"),
                        },
                        session_id=session_id,
                        request_id=request_id,
                        turn_index=ordinal,
                        role=msg["role"],
                        business_timestamp_ms=msg.get("timestamp"),
                    )
                )
            result = self.coordinator.apply(
                scope,
                request_id,
                payload_digest,
                Mutation(memories=tuple(memories)),
                {"count": len(memories)},
            )
            if self.hlm_builder is not None:
                # Raw is already durable; HTTP success waits for explicit derived
                # preparation/publication. Retry reuses the raw ledger + receipts.
                self.hlm_builder.build_session(scope, [m.logical_uid for m in memories])
            return result["count"]

    def _live_candidates(self, scope, uids, spaces=None):
        memories = self.state.get_memories(scope, uids)
        return [
            uid
            for uid in uids
            if uid in memories
            and memories[uid].active
            and memories[uid].lifecycle != "deleted"
            and (spaces is None or set(memories[uid].spaces).intersection(spaces))
        ]

    def search(self, user_id, query, top_k, *, choices=None):
        with self.admission(), self._hlm_lease(user_id, write=False):
            scope = self.scope(user_id)
            # Unknown users use the same run collection and an empty exact scope.
            diagnostic_call('scope', self.run.ensure_scope, scope)
            from ..core.query_inputs import candidate_query
            retrieval_query = diagnostic_call('query_input', candidate_query, getattr(self.settings, "quality_candidate", None), query, choices,
                                               sparse_client=getattr(self.encoders, "splade", None))
            if getattr(self.settings, 'inference_contract', None) == 'aml-official-eligible-v1':
                from mandol_competition.inference.dashscope_rerank import BACKEND
                if getattr(self.settings, 'reranker_backend', 'profile') == BACKEND:
                    from ..core.rerank_query_views import plan_query as plan_rerank_query
                    diagnostic_call('query_input', plan_rerank_query, query)
            dense, sparse = diagnostic_call('embedding', self.encoders.encode_query, retrieval_query)
            utility=getattr(self,"answer_utility_policy",None)
            returned_pool=max(top_k,utility.candidate_k) if utility else top_k
            with self.coordinator.search_view(scope) as view:
                policy = self.query_policy
                options = (
                    policy.effective_profile(self.profile).options
                    if policy
                    else self.profile.options
                )
                depth = (
                    policy.fused_k
                    if policy
                    else max(top_k, int(options.get("rerank_k", 100)))
                )
                fused, _audit = diagnostic_call('retrieval', self.retriever.retrieve,
                    view, retrieval_query, dense, sparse, top_k=depth
                )
                seeds = [u.uid for u, s in fused]
                expanded = diagnostic_call('graph', view.graph.traverse,
                    seeds,
                    hops=options["graph_hops"],
                    per_node=options["graph_neighbors"],
                    limit=options["graph_candidate_limit"],
                    allowed=lambda ids: self._live_candidates(scope, ids),
                )
                semantic_seeds = (
                    seeds[: min(10, len(seeds))]
                    if options["semantic_neighbors"]
                    else []
                )
                for seed in semantic_seeds:
                    expanded.extend(
                        h.logical_uid
                        for h in semantic_neighbors(
                            self.projection,
                            scope,
                            view.published_seq,
                            seed,
                            limit=options["semantic_neighbors"],
                            threshold=options["semantic_threshold"],
                        )
                    )
                uids = list(dict.fromkeys(seeds + expanded))
                canonical = diagnostic_call('candidate_hydration', self.state.get_memories, scope, uids)
                if set(canonical) != set(uids):
                    raise StateUnavailable("graph hydration missing canonical node")
                uids = self._live_candidates(scope, uids)
                if getattr(self, "navigation", None) is not None:
                    uids, canonical, _navigation_audit = self.navigation.expand(
                        scope, view, query, dense, sparse, uids, canonical
                    )
                finalize = getattr(getattr(self, "navigation", None), "finalize", None)
                if finalize is not None:
                    semantic_hits = finalize(scope, view, query, uids, canonical, top_k)
                    if semantic_hits is not None:
                        return semantic_hits
                if policy is not None and not policy.rerank:
                    # OFF is an explicit experiment: preserve pre-rerank order.
                    # Unscored graph-only evidence has no fabricated neural score.
                    fusion_scores = {unit.uid: score for unit, score in fused}
                    ordered = [(uid, fusion_scores.get(uid)) for uid in uids[:returned_pool]]
                else:
                    with diagnostic_stage('reranker'):
                        from mandol_competition.inference.dashscope_rerank import BACKEND
                        documents = [canonical[u].text_content for u in uids]
                        if getattr(self.settings, 'reranker_backend', 'profile') == BACKEND:
                            from ..core.rerank_query_views import rerank_query_views
                            scores = rerank_query_views(self.reranker, query, documents)
                        else:
                            scores = self.reranker.rerank(query, documents)
                        ordered=sorted(zip(uids,scores,strict=True),key=lambda item:(-item[1],item[0]))[:returned_pool]
                with diagnostic_stage('result_construction'):
                    hits = [
                        {
                            "id": uid,
                            "content": canonical[uid].text_content,
                            "score": score,
                            "created_at": datetime.fromtimestamp(
                                canonical[uid].created_at_ms / 1000, UTC
                            )
                            .isoformat()
                            .replace("+00:00", "Z"),
                        }
                        for uid, score in ordered
                    ]
                if self.context_policy is not None:
                    hits = self.context_policy.render(
                        scope, hits, {uid: canonical[uid] for uid, _ in ordered}
                    )
                if utility:
                    from ..core.answer_utility import select_existing
                    hits,fallback=select_existing(utility,hits,query=query,options=choices,
                        requested=top_k,selector=self.evidence_selector,canonical=canonical)
                    budget=ReturnPolicy(self.profile.hash,final_k=10 if fallback else utility.limit(query,top_k),
                        max_context_chars=24000 if fallback else utility.max_context_chars)
                    hits,_budget_audit=budget.materialize(hits[:top_k],top_k)
                elif self.return_policy is not None:
                    hits, _budget_audit = diagnostic_call('evidence_budget', self.return_policy.materialize, hits, top_k)
                if getattr(self, "observation_order_policy", None) is not None:
                    hits, _order_audit = self.observation_order_policy.render(
                        scope, hits, canonical, query=query, choices=choices,
                        published_seq=view.published_seq)
                return hits

    def health(self):
        with self.admission():
            if not self.state.health() or self.state.pending_count():
                raise StateUnavailable("canonical publication not ready")
            self.projection.ensure_schema(self.scope("__health_probe__"))
            for client in (self.dense, self.splade, self.reranker):
                if hasattr(client, "http"):
                    client.http.health()
                else:
                    client.health()  # explicit owned/reference reproduction
            if self._maintenance_error:
                raise StateUnavailable("projection maintenance unavailable")
            return True

    def _maintain(self):
        # Streaming GC runs without a scope write lease. It deletes only immutable
        # physical IDs that can never become current again (versions monotonic).
        # Any uncertain GC timeout is harmless; health reports the failure and a
        # later pass retries. It never recreates a collection or semantic output.
        while not self._maintenance_stop.wait(60):
            try:
                from mandol_substrate.state.models import ScopeBusy
                skipped_busy = False
                for scope in self.state.iter_scopes():
                    if self._maintenance_stop.is_set():
                        return
                    entered_gc_scope = False
                    try:
                        lease = self.run.maintenance_view(scope)
                        with lease:
                            entered_gc_scope = True
                            current = self.state.scope_state(scope)
                            if current["canonical_seq"] != current["published_seq"]:
                                continue
                            self.projection.gc_scope(scope)
                    except ScopeBusy:
                        if entered_gc_scope:
                            raise  # A failure inside GC must remain a real failure.
                        # No GC operation began. Busy readers/writers are not
                        # evidence of projection corruption or an unavailable
                        # dependency; retry this scope on the next pass.
                        skipped_busy = True
                        continue
                if not skipped_busy:
                    # Preserve an earlier real GC failure until a complete
                    # pass verifies all scopes, including any busy ones.
                    self._maintenance_error = None
            except Exception as exc:  # noqa: BLE001 -- retain failure as not-ready; no fallback
                self._maintenance_error = (
                    f"projection maintenance failed: {type(exc).__name__}"
                )

    def reranker_runtime_options(self):
        if self.reranker_execution_policy is None:
            return self.profile.options
        return self.reranker_execution_policy.effective_options(self.profile)

    def inference_attestation(self):
        if self.settings.inference_contract not in {"vpc-vllm-029-v1", "vllm-http-v1", "aml-official-eligible-v1"}:
            return {"contract": "reference-v1"}
        return {"contract": self.settings.inference_contract,
                "models": {name: getattr(self, name).attestation()
                           for name in ("dense", "splade", "reranker")
                           if hasattr(self, name)}}

    def stats(self):
        from mandol_competition.inference.dashscope_rerank import BACKEND, recipe
        from ..core.rerank_query_views import descriptor as query_view_descriptor
        from mandol_competition.inference.embedding_query_views import descriptor as embedding_view_descriptor
        backend = getattr(self.settings, 'reranker_backend', 'profile')
        external = backend == BACKEND
        bounded_embedding = self.settings.inference_contract == 'aml-official-eligible-v1'
        return {
            "backend": "qdrant",
            "retrieval_mode": "hybrid",
            "lexical_backend": "sqlite_exact",
            "inference": self.inference_attestation(),
            "reranker_execution": (
                self.reranker.attestation()
                if self.settings.inference_contract in {"vpc-vllm-029-v1", "vllm-http-v1", "aml-official-eligible-v1"} and self._ready
                else
                self.reranker_execution_policy.attestation(self.profile)
                if self.reranker_execution_policy is not None
                else None
            ),
            "graph": self.graphs.stats() if self._ready else {},
            "ready": self._ready,
            "quality_candidate": getattr(self.settings, "quality_candidate", None),
            "evidence_profile": getattr(self.settings, "selected_evidence_profile", None),
            "search_profile": {
                "id": ("textual-v4-token-and-qwen37-token-guard-v5" if external else "textual-v4-query-token-split-v2")
                    if bounded_embedding else ("textual-qwen37-query-split-v2" if external else "textual-profile-reranker-v1"),
                "storage_profile_hash": self.profile.hash if self._ready else None,
                "reranker_backend": backend,
                "reranker_recipe_hash": recipe().hash if external else None,
                "reranker_query_policy": query_view_descriptor() if external else None,
                "embedding_query_policy": embedding_view_descriptor() if bounded_embedding else None,
                "storage_and_document_encoding_changed": False,
                "long_query_representation_changed": bounded_embedding,
            },
        }

    def purge(self, user_id):
        from mandol_substrate.maintenance import ResourcePurger
        raw_scope = self.scope(user_id)
        with self.admission(), self._hlm_lease(user_id, write=True):
            report = {"purged": True}
            scopes = [raw_scope, *(s for s in self.state.owned_user_scopes(user_id) if s != raw_scope)]
            # Erase raw support first. Even an interrupted cleanup cannot expose
            # derived evidence whose sources or completed BuildJobs disappeared.
            for scope in scopes:
                with self.coordinator.gates.lease(scope, write=True):
                    self.state.begin_purge(scope, report)
                    ResourcePurger(self.state, self.projection, cleanup=self.graphs.evict).complete(scope)
            return report

    def purge_idle(self, ttl_days):
        if not self._ready:
            return 0
        cutoff = int((datetime.now(UTC).timestamp() - ttl_days * 86400) * 1000)
        deleted = 0
        for scope in self.state.iter_scopes():
            if scope.namespace != self.settings.evaluation_namespace:
                continue
            purge = self.state.purge_state(scope)
            if purge and purge["status"] == "done":
                # A crash after raw erasure may leave old derived recipes to
                # clean. Namespace authorizations remain durable across restart.
                if all(self.state.purge_state(s) and self.state.purge_state(s)["status"] == "done"
                       for s in self.state.owned_user_scopes(scope.user_id)):
                    continue
            if not purge and self.state.scope_state(scope)["updated_at_ms"] >= cutoff:
                continue
            self.purge(scope.user_id)
            deleted += 1
        return deleted

    def shutdown(self):
        self._ready = False
        self._maintenance_stop.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=30)
            if self._maintenance_thread.is_alive():
                raise StateUnavailable(
                    "projection maintenance did not stop; refusing to close its database"
                )
        for resource in reversed(self._resources):
            resource.close()
        self._resources.clear()
