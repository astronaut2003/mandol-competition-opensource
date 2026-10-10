"""Explicit AML-depth retrieval experiment; never changes the formal raw policy.

Published typed indexes are navigation inputs. Raw canonical sources remain the
default payload. No builder, provider, schema mutation, or query agent lives here.
"""

import math
import re
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from mandol_substrate.state import Scope, StateUnavailable
from .context_render import render_context_items
from .evidence_navigation import EvidenceFamilyPacker
from .heterogeneous_worker import reference_uid, typed_scope
from .query_policy import QueryPolicy
from .retrieval import CompetitionRetriever
from .semantic_evidence import SemanticEvidenceNavigation


@dataclass(frozen=True)
class RetrievalControlPolicy:
    contextual: bool = False
    profile: bool = False
    entity: bool = False
    episode_hint: bool = False
    mixed_payload: bool = False
    profile_candidate_k: int = 10
    profile_activation: str = "intent"
    profile_soft_cap: int = 0
    profile_fixed_quota: bool = False
    graph_depth: int = 1
    schema: str = "mandol-aml-control-v1"

    def __post_init__(self):
        for field in (
            "contextual",
            "profile",
            "entity",
            "episode_hint",
            "mixed_payload",
            "profile_fixed_quota",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError("Control activation flags must be booleans")
        for field in ("profile_candidate_k", "profile_soft_cap", "graph_depth"):
            if type(getattr(self, field)) is not int:
                raise ValueError("Control budgets must be integers")
        if (
            self.schema != "mandol-aml-control-v1"
            or self.profile_candidate_k not in (10, 20)
            or self.profile_activation not in ("intent", "always")
            or self.profile_soft_cap not in (0, 1, 2)
            or self.graph_depth not in (0, 1)
        ):
            raise ValueError("Unregistered control policy")
        if self.profile_fixed_quota and (self.profile_soft_cap or self.mixed_payload):
            raise ValueError("Fixed quota is an independent diagnostic")
        if (self.profile_soft_cap or self.profile_fixed_quota) and not self.profile:
            raise ValueError("Profile payload requires its independent route")


def query_signals(query):
    """Small language-level signals, never benchmark IDs or labels."""
    patterns = {
        "preference": r"prefer|favorite|favourite|like|enjoy|recommend|personaliz|偏好|喜欢|推荐",
        "routine_goal": r"habit|routine|goal|dream|aspir|constraint|avoid|习惯|目标|约束",
        "identity": r"\b(work|job|career|live|name|age|family|partner|background|identity)\b|职业|居住|身份",
        "current": r"\b(current|currently|now|latest|still|anymore)\b|目前|现在|最新",
        "historical": r"\b(before|previous|previously|used to|history|historical|formerly|past)\b|以前|曾经|历史",
        "temporal": r"\b(when|after|during|date|year|month|happen)\b|何时|之后|发生",
    }
    return tuple(k for k, p in patterns.items() if re.search(p, query, re.IGNORECASE))


def profile_lifecycle(item, signals):
    """Interpret validated status without changing or collapsing its lineage."""
    p = item.get("profile_item", {})
    if p.get("subject") != "user":
        return False, "WRONG_SUBJECT"
    status = p.get("status", "unknown").lower()
    if status not in {"current", "historical"}:
        return False, "INACTIVE_OR_UNKNOWN_LIFECYCLE"
    if status == "historical" and "historical" not in signals:
        return False, "HISTORICAL_NOT_REQUESTED"
    if "current" in signals and "historical" not in signals and status != "current":
        return False, "CURRENT_ONLY"
    if p.get("lifetime") == "temporary" and not p.get("valid_to"):
        return False, "UNBOUNDED_TEMPORARY_APPLICABILITY"
    return True, "HISTORICAL_ALLOWED" if "historical" in signals else "CURRENT_ALLOWED"


def rank_routes(routes):
    scores, origins = {}, {}
    for route, ids in routes.items():
        for rank, uid in enumerate(dict.fromkeys(ids), 1):
            scores[uid] = scores.get(uid, 0.0) + 1 / (60 + rank)
            origins.setdefault(uid, {})[route] = rank
    return sorted(scores, key=lambda u: (-scores[u], u)), origins, scores


def select_payload(raw_order, family_order, families, *, top_k, policy):
    """Candidate budgets never entitle a final slot; fixed quota is diagnostic."""
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("AML depth must be 1..100")
    if any(not math.isfinite(s) for _, s in raw_order + family_order):
        raise ValueError("Nonfinite neural score")
    threshold = raw_order[min(9, len(raw_order) - 1)][1] if raw_order else math.inf
    chosen, eligible, reasons, seen_support = [], [], {}, set()
    for uid, score in family_order:
        f = families[uid]
        if seen_support.intersection(f.supporting_raw):
            reasons[uid] = "family_overlap"
            continue
        utility = len(f.supporting_raw) > 1 or f.raw_chars > len(f.content)
        high_relevance = score >= threshold
        allowed = (
            policy.mixed_payload
            or policy.profile_fixed_quota
            or (high_relevance and utility and f.derived_type == "profile")
        )
        if not allowed:
            reasons[uid] = "low_relevance" if not high_relevance else "redundant"
            continue
        eligible.append(uid)
        chosen.append((uid, score))
        seen_support.update(f.supporting_raw)
    reservation = math.ceil(top_k * 0.05) if policy.profile_fixed_quota else 0
    if policy.mixed_payload:
        selected = sorted(raw_order + chosen, key=lambda x: (-x[1], x[0]))[:top_k]
    elif reservation:
        chosen = chosen[:reservation]
        selected = sorted(
            raw_order[: top_k - len(chosen)] + chosen, key=lambda x: (-x[1], x[0])
        )
    else:
        # Soft cap limits opportunity; it does not force a lower-ranked family.
        chosen = chosen[: policy.profile_soft_cap]
        selected = sorted(raw_order + chosen, key=lambda x: (-x[1], x[0]))[:top_k]
    selected_ids = {u for u, _ in selected}
    final_families = [u for u, _ in selected if u in families]
    displaced = [u for u, _ in raw_order[:top_k] if u not in selected_ids]
    return selected, {
        "eligible_families": eligible,
        "family_rejections": reasons,
        "selected_families": {u: asdict(families[u]) for u in final_families},
        "raw_displaced": displaced,
        "reserved_slots": reservation,
        "profile_payload_count": sum(
            families[u].derived_type == "profile" for u in final_families
        ),
        "unused_soft_slots": max(0, policy.profile_soft_cap - len(final_families)),
        "raw_backfill": max(
            0, (reservation or policy.profile_soft_cap) - len(final_families)
        ),
        "below_raw10_relevance_families": [
            u for u, s in selected if u in families and s < threshold
        ],
    }


class RetrievalControlPlane(SemanticEvidenceNavigation):
    """Binds to CompetitionMemoryService.navigation under its publication lease."""

    def __init__(
        self, service, navigation_policy, routes, *, control_policy, contextual=None
    ):
        super().__init__(service, navigation_policy, routes)
        self.control_policy, self.contextual = control_policy, contextual
        if control_policy.contextual and contextual is None:
            raise ValueError("Contextual route requires a published recipe binding")
        self.request_state = ContextVar("retrieval_control_request", default=None)

    def _lookup(self, scope, route, query, dense, sparse, limit):
        partition = Scope(route.namespace, scope.user_id)
        if partition.key == scope.key:
            raise ValueError("Raw and derived lexical statistics must be isolated")
        if not self.service.state._rows(
            "SELECT 1 FROM scopes WHERE scope=?", (partition.key,)
        ):
            return [], {}
        policy = QueryPolicy(
            self.service.profile.hash,
            self.service.profile.methods,
            tuple((m, limit) for m in self.service.profile.methods),
            fused_k=limit,
            graph_hops=0,
            semantic_neighbors=0,
        )
        retriever = CompetitionRetriever(
            self.service.state,
            self.service.projection,
            self.service.profile,
            query_policy=policy,
        )
        with self.service.coordinator.search_view(partition) as view:
            ranked, audit = retriever.retrieve(view, query, dense, sparse, top_k=limit)
            ids = [m.uid for m, _ in ranked]
            eligibility = route.eligible
            if (
                route.kind == "profile"
                and "historical" in query_signals(query)
                and route.historical_eligible
            ):
                eligibility = route.historical_eligible
            admitted = eligibility(
                scope, self.service.state.get_memories(partition, ids)
            )
            result = []
            for uid in ids:
                m = admitted.get(uid)
                if m is not None:
                    if (
                        m.metadata.get("user_id") != scope.user_id
                        or m.commit_seq > view.published_seq
                    ):
                        raise StateUnavailable("Foreign or unpublished typed candidate")
                    result.append(m)
            return result, {k: v for k, v in audit.items() if k != "canonical"}

    def _support(self, scope, view, memory, canonical):
        meta = memory.metadata
        item = meta["hlm_item"]
        evidence = meta.get(
            "source_spans", item.get("evidence", item.get("mentions", []))
        )
        ids = list(dict.fromkeys(e["source_uid"] for e in evidence))
        canonical.update(self.service.state.get_memories(scope, ids))
        accepted = []
        for e in evidence:
            uid = e["source_uid"]
            raw = canonical.get(uid)
            if (
                raw is None
                or raw.logical_uid != uid
                or not EvidenceFamilyPacker._raw(scope, raw, view.published_seq)
                or raw.version != meta.get("source_versions", {}).get(uid)
                or not e.get("quote")
                or e["quote"] not in raw.text_content
            ):
                raise StateUnavailable("Invalid exact supporting source")
            if uid not in accepted:
                accepted.append(uid)
        return accepted

    def expand(self, scope, view, query, dense, sparse, raw_ids, canonical):
        tick = time.monotonic()
        if len(raw_ids) > 150 or len(set(raw_ids)) != len(raw_ids):
            raise StateUnavailable("Frozen raw candidate bound changed")
        if any(
            u not in canonical
            or not EvidenceFamilyPacker._raw(scope, canonical[u], view.published_seq)
            for u in raw_ids
        ):
            raise StateUnavailable("Invalid raw frontier")
        policy = self.control_policy
        signals = query_signals(query)
        trigger = policy.profile and (
            policy.profile_activation == "always" or bool(set(signals) - {"temporal"})
        )
        # Fixed quota diagnostic retrieves the small index even for unrelated queries.
        trigger = trigger or policy.profile_fixed_quota
        routes = {r.kind: r for r in self.routes}
        limits = {"profile": policy.profile_candidate_k, "entity": 20, "episode": 10}
        kinds = (["profile"] if trigger else []) + (["entity"] if policy.entity else [])
        if policy.episode_hint or policy.mixed_payload:
            kinds.append("episode")
        selected, times, audits, failures, rejected = {}, {}, {}, [], []
        if policy.contextual:
            from .navigation_runtime import DerivedRoute

            routes["contextual"] = DerivedRoute(
                "contextual",
                typed_scope(scope, "contextual", self.contextual.recipe).namespace,
                self.contextual.eligible,
            )
            kinds.append("contextual")
            limits["contextual"] = 100
        for kind in kinds:
            begin = time.monotonic()
            try:
                selected[kind], audits[kind] = self._lookup(
                    scope, routes[kind], query, dense, sparse, limits[kind]
                )
            except StateUnavailable as exc:
                selected[kind] = []
                failures.append(
                    {
                        "route": kind,
                        "failure": type(exc).__name__,
                        "policy": "RAW_FALLBACK",
                    }
                )
            times[kind] = time.monotonic() - begin
        filtered = []
        for m in selected.get("profile", []):
            ok, reason = profile_lifecycle(m.metadata["hlm_item"], signals)
            if ok:
                filtered.append(m)
            else:
                rejected.append({"uid": m.logical_uid, "reason": reason})
        selected["profile"] = filtered
        entity_seeds = []
        for m in selected.get("entity", []):
            item = m.metadata["hlm_item"]
            if any(
                n
                and re.search(
                    r"(?<!\w)" + re.escape(n) + r"(?!\w)", query, re.IGNORECASE
                )
                for n in [item.get("name", ""), *item.get("aliases", [])]
            ):
                entity_seeds.append(m)
        selected["entity"] = entity_seeds[:4]
        begin = time.monotonic()
        reached = []
        if policy.entity and policy.graph_depth and "relation" in routes:
            eroute, rroute = routes["entity"], routes["relation"]
            partition = Scope(rroute.namespace, scope.user_id)
            for m in selected["entity"]:
                ref = reference_uid(
                    Scope(eroute.namespace, scope.user_id), m.logical_uid, m.version
                )
                edges = self.service.state._rows(
                    "SELECT target_uid FROM graph_edges WHERE scope=? AND source_uid=? AND active=1 "
                    "AND relation_type IN ('RELATION_SUBJECT','RELATION_OBJECT') ORDER BY edge_id LIMIT 4",
                    (partition.key, ref),
                )
                rows = self.service.state.get_memories(
                    partition, [e["target_uid"] for e in edges]
                )
                admitted = rroute.eligible(scope, rows)
                for relation in admitted.values():
                    # Structural admission; semantic direction can still be wrong.
                    if relation.metadata.get("user_id") != scope.user_id:
                        raise StateUnavailable("Cross-user relation")
                    item = relation.metadata["hlm_item"]
                    if item.get("condition") or item.get("status") not in {
                        "CURRENT",
                        "HISTORICAL",
                    }:
                        continue
                    reached.append(relation)
            selected["relation"] = list({m.logical_uid: m for m in reached}.values())
        times["graph_expansion"] = time.monotonic() - begin
        caps = {"contextual": 40, "profile": 40, "entity_graph": 40, "episode_hint": 10}
        rank_lists = {"raw": list(raw_ids)}
        typed = []
        supports_by_hint = {}
        for kind, rows in selected.items():
            route = (
                "entity_graph"
                if kind in {"entity", "relation"}
                else "episode_hint"
                if kind == "episode"
                else kind
            )
            for m in rows:
                if kind == "episode" and not policy.mixed_payload:
                    item = m.metadata["hlm_item"]
                    if (
                        item.get("confidence", 0) < 0.99
                        or item.get("status") != "COMPLETED"
                        or item.get("assertion_kind") != "EVENT_OCCURRENCE"
                        or item.get("time_precision") == "UNKNOWN"
                    ):
                        rejected.append(
                            {
                                "uid": m.logical_uid,
                                "reason": "CONSERVATIVE_EPISODE_HINT",
                            }
                        )
                        continue
                try:
                    supports = self._support(scope, view, m, canonical)
                except StateUnavailable:
                    rejected.append(
                        {
                            "uid": m.logical_uid,
                            "reason": "INVALID_SOURCE_VERSION_OR_QUOTE",
                        }
                    )
                    continue
                typed.append((kind, m))
                supports_by_hint[m.logical_uid] = supports
                target = rank_lists.setdefault(route, [])
                for uid in supports[: 4 if kind != "contextual" else 1]:
                    if uid not in target and len(target) < caps[route]:
                        target.append(uid)
        begin = time.monotonic()
        ids, origins, fusion_scores = rank_routes(rank_lists)
        if len(ids) > 280 or not set(raw_ids) <= set(ids):
            raise StateUnavailable("Raw-preserving candidate union bound violated")
        times["fusion"] = time.monotonic() - begin
        audit = {
            "policy": asdict(policy),
            "signals": signals,
            "profile_triggered": bool(trigger),
            "selected_typed": {
                k: [m.logical_uid for m in v] for k, v in selected.items()
            },
            "route_lists": rank_lists,
            "origins": origins,
            "fusion_scores": fusion_scores,
            "typed_audits": audits,
            "supports_by_hint": supports_by_hint,
            "raw_ids": list(raw_ids),
            "failures": failures,
            "rejected": rejected,
            "timings": times,
            "expand_seconds": time.monotonic() - tick,
            "query_gpt_calls": 0,
            "pre_rerank_count": len(ids),
        }
        self.typed_candidates.set(tuple(typed))
        self.request_state.set(audit)
        self.trace.set(audit)
        return ids, canonical, audit

    def finalize(self, scope, view, query, raw_ids, canonical, requested_top_k):
        tick = time.monotonic()
        policy = self.control_policy
        audit = dict(self.request_state.get())
        families = {}
        if (
            policy.profile_soft_cap
            or policy.profile_fixed_quota
            or policy.mixed_payload
        ):
            for kind, m in self.typed_candidates.get():
                if kind != "profile" and not (
                    kind == "episode" and policy.mixed_payload
                ):
                    continue
                family = self._family(scope, view, kind, m, canonical)
                if family is not None and len(families) < 20:
                    families[family.family_id] = family
        texts = [canonical[u].text_content for u in raw_ids] + [
            f.content for f in families.values()
        ]
        if texts:
            scores = self.service.reranker.rerank(query, texts)
            if len(scores) != len(texts) or any(not math.isfinite(s) for s in scores):
                raise StateUnavailable("Invalid Querit scores")
        else:
            scores = []
        audit["timings"]["querit"] = time.monotonic() - tick
        raw_order = sorted(
            zip(raw_ids, scores[: len(raw_ids)], strict=True),
            key=lambda x: (-x[1], x[0]),
        )
        family_order = sorted(
            zip(families, scores[len(raw_ids) :], strict=True),
            key=lambda x: (-x[1], x[0]),
        )
        selected, packing = select_payload(
            raw_order, family_order, families, top_k=requested_top_k, policy=policy
        )
        raw_hits = [
            {
                "id": u,
                "content": canonical[u].text_content,
                "score": s,
                "created_at": datetime.fromtimestamp(
                    canonical[u].created_at_ms / 1000, UTC
                )
                .isoformat()
                .replace("+00:00", "Z"),
            }
            for u, s in selected
            if u in canonical
        ]
        rendered, _ = render_context_items(
            scope,
            raw_hits,
            {h["id"]: canonical[h["id"]] for h in raw_hits},
            mode=self.service.context_policy.mode,
            final_k=requested_top_k,
        )
        lookup = {h["id"]: h for h in rendered}
        hits = [
            lookup[u]
            if u in lookup
            else {"id": u, "content": families[u].content, "score": s}
            for u, s in selected
        ]
        audit.update(packing)
        audit.update(
            raw_order=raw_order,
            family_order=family_order,
            family_candidates={u: asdict(f) for u, f in families.items()},
            context_chars=len("\n\n".join(h["content"] for h in hits)),
            output_depth=requested_top_k,
            output_mode="AML_PUBLIC_CONTRACT",
            final_ids=[h["id"] for h in hits],
            finalization_seconds=time.monotonic() - tick,
        )
        self.trace.set(audit)
        return hits


def bind_retrieval_control(service, recipe, policy, *, contextual_recipe=None):
    """Attach an explicitly selected candidate to the existing service.

    This reads completed constructions; it does not enable generation on Add.
    The caller owns the immutable recipe identities and experimental selection.
    """
    from .contextual_base import ContextualBaseMemory
    from .evidence_navigation import NavigationPolicy
    from .hlm_production import CanonicalHighLevelMemory
    from .navigation_runtime import DerivedRoute

    if not service.ready:
        raise StateUnavailable("Control-plane binding requires ready raw runtime")
    runtime = CanonicalHighLevelMemory(service.coordinator, recipe, None)
    service._resources.append(runtime)
    routes = [
        DerivedRoute(
            k,
            typed_scope(service.scope("binding-only"), k, recipe).namespace,
            runtime.eligible,
            runtime.eligible_history if k == "profile" else None,
        )
        for k in ("profile", "episode", "entity", "relation")
    ]
    contextual = (
        ContextualBaseMemory(
            service.coordinator,
            contextual_recipe,
            None,
            publication_writer=runtime._writer,
        )
        if contextual_recipe
        else None
    )
    service.navigation = RetrievalControlPlane(
        service,
        NavigationPolicy(tuple(r.kind for r in routes)),
        routes,
        control_policy=policy,
        contextual=contextual,
    )
    return service.navigation
