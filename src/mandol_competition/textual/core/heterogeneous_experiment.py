"""PUBLIC development overlay; formal Competition service is not modified."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from mandol_competition.textual.api.competition import CompetitionMemoryService

from mandol_substrate.state.models import StateUnavailable, digest
from .heterogeneous_admission import ItemAdmissionRecipe
from .heterogeneous_memory import HeterogeneousProvider, HeterogeneousRecipe, render
from .heterogeneous_replay import DurableConstructionReplay
from .heterogeneous_retrieval import late_rrf, leg_overlap, plan_query, select_context
from .heterogeneous_worker import HeterogeneousRuntime, reference_uid, typed_scope
from .paired_quality import append_call
from .profile_experiment import ModelBridge
from .query_policy import QueryPolicy
from .retrieval import CompetitionRetriever
from .run_storage import atomic_json

ARMS = {
    "raw": {
        "types": (),
        "planner": "off",
        "graph": 0,
        "provenance": False,
        "reserve": 10,
        "dedup": False,
    },
    "episode": {
        "types": ("episode",),
        "planner": "uniform",
        "graph": 0,
        "provenance": False,
        "reserve": 6,
        "dedup": False,
    },
    "episode_late": {
        "types": ("episode",),
        "planner": "uniform",
        "graph": 0,
        "provenance": False,
        "reserve": 6,
        "dedup": True,
    },
    "episode_plan": {
        "types": ("episode",),
        "planner": "intent",
        "graph": 0,
        "provenance": False,
        "reserve": 6,
        "dedup": True,
    },
    "episode_quantified": {
        "types": ("episode",),
        "planner": "quantified",
        "graph": 0,
        "provenance": False,
        "reserve": 6,
        "dedup": True,
    },
    "episode_source": {
        "types": ("episode",),
        "planner": "intent",
        "graph": 0,
        "provenance": True,
        "reserve": 6,
        "dedup": True,
    },
    "episode_graph1": {
        "types": ("episode",),
        "planner": "intent",
        "graph": 1,
        "provenance": True,
        "reserve": 6,
        "dedup": True,
    },
    "episode_graph2": {
        "types": ("episode",),
        "planner": "intent",
        "graph": 2,
        "provenance": True,
        "reserve": 6,
        "dedup": True,
    },
    "episode_reserve8": {
        "types": ("episode",),
        "planner": "intent",
        "graph": 1,
        "provenance": True,
        "reserve": 8,
        "dedup": True,
    },
}


class HeterogeneousExperimentService(CompetitionMemoryService):
    def __init__(self, settings, *, gate_path, replay_ledger=None, replay_sqlite=None):
        super().__init__(settings)
        gate = json.loads(Path(gate_path).read_text())
        from .heterogeneous_grounding import GroundedRecipe

        self.recipe_class = (
            GroundedRecipe
            if gate.get("admission") == "world_grounded_v9"
            else ItemAdmissionRecipe
            if gate.get("admission") == "individual_strict_v1"
            else HeterogeneousRecipe
        )
        self.replay_ledger, self.replay_sqlite = replay_ledger, replay_sqlite
        if bool(replay_ledger) != bool(replay_sqlite):
            raise ValueError("Replay requires both source ledger and canonical SQLite")
        if (
            gate.get("status") != "PASS_FOR_SMALL_PUBLIC_PANEL_ONLY"
            or gate["recipe_hash"] != self.recipe_class("0" * 64).fingerprint
            or gate.get("episode_precision", 0) < 0.8
            or gate.get("episode_coverage", 0) < 0.75
        ):
            raise StateUnavailable("Matching small-panel semantic gate required")
        self.allowed_types = set(gate["allowed_types"])
        self.bridge = None

    def start(self):
        super().start()
        self._ready = False
        root = Path(self.settings.competition_run_root)
        self.recipe = self.recipe_class(self.profile.hash)
        binding = {
            "recipe_hash": self.recipe.fingerprint,
            "base_profile_hash": self.profile.hash,
            "base": "source_b1_public_v1",
            "formal": False,
            "arms": ARMS,
            "publication": "explicit separate complete BuildJob frontier; raw Add is unchanged",
        }
        path = root / "heterogeneous-experiment.json"
        if path.exists() and json.loads(path.read_text()) != json.loads(
            json.dumps(binding)
        ):
            raise StateUnavailable("Immutable experiment run binding changed")
        atomic_json(path, binding)
        self.bridge = (
            DurableConstructionReplay(
                self.replay_ledger,
                self.replay_sqlite,
                root / "artifacts/hlm-calls.jsonl",
            )
            if self.replay_ledger
            else ModelBridge(
                root / "artifacts/hlm-calls.jsonl",
                provider_factory=HeterogeneousProvider,
            )
        )
        self.hlm = HeterogeneousRuntime(self.coordinator, self.recipe, self.bridge)
        self._ready = True

    def build_sources(self, user_id, source_uids):
        with self.admission():
            started = time.monotonic()
            result = self.hlm.build(self.scope(user_id), source_uids)
            append_call(
                Path(self.settings.competition_run_root) / "artifacts/hlm-builds.jsonl",
                {
                    "user_id": user_id,
                    "source_uids": source_uids,
                    "result": result,
                    "seconds": time.monotonic() - started,
                },
            )
            return result

    def _typed(self, raw_scope, kind, query, dense, sparse, limit):
        if kind not in self.allowed_types or limit == 0:
            return [], {}, {}
        scope = typed_scope(raw_scope, kind, self.recipe)
        if not self.state._rows("SELECT 1 FROM scopes WHERE scope=?", (scope.key,)):
            return [], {}, {}
        policy = QueryPolicy(
            self.profile.hash,
            self.profile.methods,
            tuple((method, limit) for method in self.profile.methods),
            fused_k=limit,
            graph_hops=0,
            semantic_neighbors=0,
        )
        retriever = CompetitionRetriever(
            self.state, self.projection, self.profile, query_policy=policy
        )
        with self.coordinator.search_view(scope) as view:
            ranked, audit = retriever.retrieve(view, query, dense, sparse, top_k=limit)
            canonical = self.state.get_memories(scope, [m.uid for m, _ in ranked])
            valid = self.hlm.eligible(raw_scope, canonical)
            return [m.uid for m, _ in ranked if m.uid in valid], valid, audit

    def _graph(self, raw_scope, seeds, raw_seeds, kind, depth, cap):
        if not depth:
            return []
        scope = typed_scope(raw_scope, kind, self.recipe)
        if not self.state._rows("SELECT 1 FROM scopes WHERE scope=?", (scope.key,)):
            return []
        entity_scope = typed_scope(raw_scope, "entity", self.recipe)
        starts = [reference_uid(entity_scope, u, 1) for u in seeds[:4]]
        raw = self.state.get_memories(raw_scope, raw_seeds[:4])
        starts.extend(reference_uid(raw_scope, u, m.version) for u, m in raw.items())
        with self.coordinator.search_view(scope) as view:
            ids = view.graph.traverse(
                starts,
                hops=depth,
                limit=32,
                per_node=cap,
                relations={
                    "HAS_PARTICIPANT",
                    "PARTICIPATES_IN",
                    "DERIVED_FROM",
                    "SUPPORTS_EPISODE",
                    "SUBJECT",
                    "OBJECT",
                    "RELATION_SUBJECT",
                    "RELATION_OBJECT",
                },
            )
            canonical = self.state.get_memories(scope, ids)
            return list(self.hlm.eligible(raw_scope, canonical))

    def search_arm(self, user_id, query, top_k, arm):
        if arm not in ARMS:
            raise ValueError("Unknown frozen experimental arm")
        options = ARMS[arm]
        scope = self.scope(user_id)
        started = time.monotonic()
        times = {}
        if not self.state._rows("SELECT 1 FROM scopes WHERE scope=?", (scope.key,)):
            return [], {"arm": arm, "unknown_user": True}
        with (
            self.admission(),
            self.hlm.gates.lease(scope),
            self.coordinator.search_view(scope) as view,
        ):
            tick = time.monotonic()
            dense, sparse = self.encoders.encode_query(query)
            times["encode"] = time.monotonic() - tick
            tick = time.monotonic()
            raw_ranked, audit = self.retriever.retrieve(
                view, query, dense, sparse, top_k=100
            )
            times["raw"] = time.monotonic() - tick
            raw_ids = [m.uid for m, _ in raw_ranked]
            canonical = self.state.get_memories(scope, raw_ids)
            ranked = {"raw": raw_ids}
            typed_audit = {}
            anchors = []
            entity_map = {}
            failures = []
            if options["types"]:
                tick = time.monotonic()
                _, entity_map, _ = self._typed(scope, "entity", query, dense, sparse, 4)
                for u, m in entity_map.items():
                    names = [
                        m.metadata["hlm_item"]["name"],
                        *m.metadata["hlm_item"]["aliases"],
                    ]
                    if any(
                        re.search(
                            r"(?<!\w)" + re.escape(name) + r"(?!\w)",
                            query,
                            re.IGNORECASE,
                        )
                        for name in names
                    ):
                        anchors.append(u)
                times["entity"] = time.monotonic() - tick
            tick = time.monotonic()
            plan = plan_query(
                query,
                planner=options["planner"],
                entity_anchors=anchors,
                overlap=leg_overlap(audit["legs"]),
                graph_depth=options["graph"],
                raw_reserve=options["reserve"],
            )
            times["planner"] = time.monotonic() - tick

            # Raw is unconditional. Typed retrieval cannot change raw inputs,
            # canonical corpus, raw K100 or raw RRF60.
            def retrieve(kind):
                t = time.monotonic()
                try:
                    value = self._typed(
                        scope,
                        kind,
                        query,
                        dense,
                        sparse,
                        dict(plan.candidate_budgets)[kind],
                    )
                except StateUnavailable as exc:
                    failures.append(
                        {
                            "kind": kind,
                            "failure": type(exc).__name__,
                            "policy": "EXPLICIT_RAW_FALLBACK",
                        }
                    )
                    value = ([], {}, {})
                return kind, value, time.monotonic() - t

            with ThreadPoolExecutor(max_workers=2) as pool:
                for kind, (ids, records, detail), elapsed in pool.map(
                    retrieve, options["types"]
                ):
                    ranked[kind] = ids
                    canonical.update(records)
                    typed_audit[kind] = detail
                    times[kind] = elapsed
            tick = time.monotonic()
            expanded = []
            for kind in options["types"]:
                ids = self._graph(
                    scope, anchors, raw_ids, kind, plan.graph_depth, plan.neighbor_cap
                )
                s = typed_scope(scope, kind, self.recipe)
                records = self.state.get_memories(s, ids)
                canonical.update(records)
                ranked[kind] = list(dict.fromkeys([*ranked[kind], *ids]))[:100]
                expanded.extend(ids)
            times["graph"] = time.monotonic() - tick
            tick = time.monotonic()
            if arm == "episode":
                union = list(dict.fromkeys(u for ids in ranked.values() for u in ids))
                contributions = {}
            else:
                union, contributions = late_rrf(ranked, limit=200)
            # Preserve baseline input order exactly when no HLM was admitted.
            if not any(ranked.get(k) for k in options["types"]):
                union = list(raw_ids)
            provenance = []
            if options["provenance"]:
                for uid in union:
                    m = canonical[uid]
                    if m.metadata.get("hlm_kind"):
                        ev = m.metadata["hlm_item"].get("evidence", [])
                        provenance.extend(e["source_uid"] for e in ev[:1])
                additional = self.state.get_memories(
                    scope, list(dict.fromkeys(provenance))
                )
                canonical.update(additional)
                union = list(dict.fromkeys([*union, *additional]))
            times["fusion_provenance"] = time.monotonic() - tick
            texts = [canonical[u].text_content for u in union]
            tick = time.monotonic()
            scores = self.reranker.rerank(query, texts) if texts else []
            times["querit"] = time.monotonic() - tick
            ordered = sorted(
                zip(union, scores, strict=True), key=lambda x: (-x[1], x[0])
            )
            tick = time.monotonic()
            selected = select_context(
                ordered,
                canonical,
                raw_reserve=options["reserve"],
                dedup=options["dedup"],
            )
            hits = []
            source_map = {}
            for uid, score in selected:
                m = canonical[uid]
                kind = m.metadata.get("hlm_kind")
                hit = {"id": uid, "content": m.text_content, "score": score}
                if not kind:
                    hit = self.context_policy.render(scope, [hit], {uid: m})[0]
                    source_map[uid] = [uid]
                else:
                    row = m.metadata["hlm_item"]
                    hit["content"] = render(kind, row, with_quotes=True)
                    source_map[uid] = [e["source_uid"] for e in row.get("evidence", [])]
                hits.append(hit)
            hits, budget = self.return_policy.materialize(hits, top_k)
            times["packing"] = time.monotonic() - tick
            times["total"] = time.monotonic() - started
            trace = {
                "arm": arm,
                "query": query,
                "plan": asdict(plan),
                "ranked": ranked,
                "expanded": expanded,
                "provenance_expanded": provenance,
                "fusion": contributions,
                "rerank_ids": union,
                "rerank_text_hash": digest(texts),
                "reranked": ordered,
                "final_ids": [h["id"] for h in hits],
                "source_map": source_map,
                "return_chars": sum(len(h["content"]) for h in hits),
                "times": times,
                "failures": failures,
                "raw_legs": {k: list(v) for k, v in audit["legs"].items()},
                "typed_legs": {
                    kind: {k: list(v) for k, v in a.get("legs", {}).items()}
                    for kind, a in typed_audit.items()
                },
                "context_hash": digest(hits),
                "budget": budget,
            }
            return hits, trace

    def shutdown(self):
        if self.bridge:
            self.bridge.close()
            self.bridge = None
        super().shutdown()
