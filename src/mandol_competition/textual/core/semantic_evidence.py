"""Opt-in semantic evidence families on the frozen raw Search/navigation path.

Per-type ranks allocate candidates, Querit scores one bounded union. Supporting
quotes remain exact canonical substrings. No entity occupies a final slot.
"""

import hashlib
import math
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from mandol_substrate.state import Scope
from mandol_substrate.state.models import digest
from .evidence_navigation import NavigationHint, RawSupport
from .heterogeneous_retrieval import late_rrf, plan_query
from .heterogeneous_worker import reference_uid
from .navigation_runtime import HeterogeneousNavigation


@dataclass(frozen=True)
class EvidenceFamily:
    family_id: str
    derived_type: str
    derived_uid: str
    semantic_text: str
    supporting_raw: tuple[str, ...]
    contextual_raw: tuple[str, ...]
    content: str
    raw_chars: int


class SemanticEvidenceNavigation(HeterogeneousNavigation):
    def __init__(
        self,
        *args,
        semantic_evidence_enabled=False,
        raw_reserve=6,
        profile_budget=2,
        selected_k=10,
        max_context_chars=24000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if raw_reserve not in (4, 6, 8, 10) or profile_budget not in (0, 1, 2):
            raise ValueError("Unqualified evidence budget")
        if (type(selected_k) is not int or not 1 <= selected_k <= 100
                or type(max_context_chars) is not int or not 1000 <= max_context_chars <= 100000):
            raise ValueError("Invalid evidence return budget")
        self.raw_reserve, self.profile_budget = raw_reserve, profile_budget
        self.selected_k, self.max_context_chars = selected_k, max_context_chars
        self.rerank_inputs = ContextVar("evidence_family_rerank_inputs", default=None)
        self.semantic_mode = ContextVar(
            "semantic_evidence_mode", default=semantic_evidence_enabled
        )
        self.expansion_mode = ContextVar("hlm_provenance_expansion", default=True)

    @contextmanager
    def select_mode(self, *, semantic, expansion=True):
        a = self.semantic_mode.set(semantic)
        b = self.expansion_mode.set(expansion)
        try:
            yield
        finally:
            self.semantic_mode.reset(a)
            self.expansion_mode.reset(b)

    def expand(self, scope, view, query, dense, sparse, raw_ids, canonical):
        ids, canonical, audit = super().expand(
            scope, view, query, dense, sparse, raw_ids, canonical
        )
        if self.expansion_mode.get() and "entity" in self._active_types.get():
            typed = list(self.typed_candidates.get())
            seeds = [(kind, m) for kind, m in typed if kind == "entity"][:4]
            routes = {
                r.kind: r for r in self.routes if r.kind in self._active_types.get()
            }
            reached = []
            for _, entity in seeds:
                names = [
                    entity.metadata["hlm_item"]["name"],
                    *entity.metadata["hlm_item"].get("aliases", []),
                ]
                if not any(
                    re.search(
                        r"(?<!\w)" + re.escape(n) + r"(?!\w)", query, re.IGNORECASE
                    )
                    for n in names
                ):
                    continue
                escope = Scope(routes["entity"].namespace, scope.user_id)
                ref = reference_uid(escope, entity.logical_uid, entity.version)
                for kind in ("relation", "episode"):
                    if kind not in routes:
                        continue
                    partition = Scope(routes[kind].namespace, scope.user_id)
                    rows = self.service.state._rows(
                        "SELECT target_uid FROM graph_edges WHERE scope=? AND source_uid=? AND active=1 AND relation_type IN ('RELATION_SUBJECT','RELATION_OBJECT','PARTICIPATES_IN') ORDER BY edge_id LIMIT 4",
                        (partition.key, ref),
                    )
                    memories = self.service.state.get_memories(
                        partition, [r["target_uid"] for r in rows]
                    )
                    allowed = routes[kind].eligible(scope, memories)
                    for uid, m in allowed.items():
                        if (kind, uid) not in {(k, v.logical_uid) for k, v in typed}:
                            typed.append((kind, m))
                            reached.append(uid)
            self.typed_candidates.set(tuple(typed))
            hints = []
            for kind, m in typed:
                ev = m.metadata["hlm_item"].get(
                    "evidence", m.metadata["hlm_item"].get("mentions", [])
                )
                sources = self.service.state.get_memories(
                    scope, [e["source_uid"] for e in ev]
                )
                canonical.update(sources)
                supports = tuple(
                    RawSupport(
                        scope.key,
                        e["source_uid"],
                        m.metadata["source_versions"][e["source_uid"]],
                        hashlib.sha256(
                            sources[e["source_uid"]].text_content.encode()
                        ).hexdigest(),
                        e["quote"],
                    )
                    for e in ev
                    if e["source_uid"] in sources
                )
                hints.append(
                    NavigationHint(m.logical_uid, kind, scope.user_id, supports, True)
                )
            ids, packing = self.packer.candidates(
                scope, raw_ids, hints, canonical, published_seq=view.published_seq
            )
            audit.update(packing)
            audit.update(graph_depth=1, neighbor_cap=4, graph_reached=reached)
        plan = plan_query(
            query, planner="intent", graph_depth=0, raw_reserve=self.raw_reserve
        )
        plan = replace(
            plan,
            candidate_budgets=(
                *plan.candidate_budgets,
                ("profile", 16),
                ("hierarchical", 16),
            ),
        )
        audit["heterogeneous_query_plan"] = vars(plan)
        if not self.expansion_mode.get():
            audit["expansion_disabled"] = True
            ids = list(raw_ids)
        return ids, canonical, audit

    def _family(self, scope, view, kind, memory, canonical):
        if kind == "entity":
            return None
        meta = memory.metadata
        item = meta["hlm_item"]
        if kind == "profile":
            p = item["profile_item"]
            # Unresolved temporary applicability is not silently declared current.
            if p["lifetime"] == "temporary" and not p.get("valid_to"):
                return None
        evidence = item.get("evidence", [])
        if not evidence:
            return None
        support = tuple(dict.fromkeys(e["source_uid"] for e in evidence))
        for e in evidence:
            raw = canonical.get(e["source_uid"])
            if (
                raw is None
                or not self.packer._raw(scope, raw, view.published_seq)
                or raw.version != meta["source_versions"][e["source_uid"]]
                or not e["quote"]
                or e["quote"] not in raw.text_content
            ):
                return None
        rendered = []
        for e in evidence:
            raw = canonical[e["source_uid"]]
            mention = (
                datetime.fromtimestamp(raw.business_timestamp_ms / 1000, UTC)
                .isoformat()
                .replace("+00:00", "Z")
                if raw.business_timestamp_ms is not None
                else "UNKNOWN"
            )
            rendered.append(
                f"[Supporting raw {raw.logical_uid}; source mention time={mention}; role={raw.role}]\n{raw.text_content if len(support) == 1 else e['quote']}"
            )
        context = tuple(item.get("context_source_uids", []))
        contexts = self.service.state.get_memories(scope, list(context))
        for uid in context:
            raw = contexts.get(uid)
            if (
                raw is None
                or not self.packer._raw(scope, raw, view.published_seq)
                or raw.version != meta["source_versions"][uid]
            ):
                return None
            rendered.append(
                f"[Context only {uid}; role={raw.role}]\n{raw.text_content}"
            )
        semantic_text = memory.text_content
        structured = item.get("profile_item", item)
        # Rendering cannot silently discard an accepted structured qualifier
        # merely because the model omitted it from its free-text statement.
        for key in ("scope", "condition", "valid_from", "valid_to", "time_qualifier"):
            value = structured.get(key)
            if value is not None and str(value) not in semantic_text:
                semantic_text += f"\n{key}: {value}"
        content = semantic_text + "\n" + "\n".join(rendered)
        # Never cut off supporting/context evidence to make a family fit.
        if len(content) > 12000:
            return None
        return EvidenceFamily(
            "family_" + digest([memory.logical_uid, memory.version]),
            kind,
            memory.logical_uid,
            semantic_text,
            support,
            context,
            content,
            sum(len(canonical[u].text_content) for u in support),
        )

    def finalize(self, scope, view, query, raw_ids, canonical, requested_top_k):
        if not self.semantic_mode.get() or not self._active_types.get():
            return None
        start = time.monotonic()
        families = {}
        ranks = {}
        for kind, memory in self.typed_candidates.get():
            family = self._family(scope, view, kind, memory, canonical)
            if family is None:
                continue
            if not self.expansion_mode.get() and not set(family.supporting_raw) <= set(
                raw_ids
            ):
                continue
            families[family.family_id] = family
            ranks.setdefault(kind, []).append(family.family_id)
        if not families:
            return None
        union, _ = late_rrf(ranks, limit=32)
        raw_texts = [canonical[u].text_content for u in raw_ids]
        documents = raw_texts + [families[u].content for u in union]
        source_families = {}
        for uid in union:
            family = families[uid]
            memory = next(m for kind, m in self.typed_candidates.get()
                          if kind == family.derived_type and m.logical_uid == family.derived_uid)
            refs = []
            for e in memory.metadata["hlm_item"]["evidence"]:
                raw, quote = canonical[e["source_uid"]], e["quote"]
                quote_start = raw.text_content.index(quote)
                refs.append({"raw_uid": raw.logical_uid, "source_version": raw.version,
                    "content_sha256": digest(raw.text_content),
                    "quote_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                    "quote_start": quote_start, "quote_end": quote_start + len(quote)})
            source_families[uid] = {"derived_uid": family.derived_uid,
                "derived_version": memory.version, "derived_type": family.derived_type,
                "source_refs": refs, "context_only_uids": list(family.contextual_raw)}
        identities = [*raw_ids, *union]
        token = self.rerank_inputs.set({"ids": identities,
            "hashes": dict(zip(identities, (digest(t) for t in documents), strict=True)),
            "source_families": source_families})
        try:
            scores = self.service.reranker.rerank(query, documents)
        finally:
            self.rerank_inputs.reset(token)
        if len(scores) != len(raw_ids) + len(union) or any(
            not math.isfinite(s) for s in scores
        ):
            raise ValueError("Incomplete/nonfinite real reranker scores")
        raw_scores = dict(zip(raw_ids, scores[: len(raw_ids)], strict=True))
        family_scores = dict(zip(union, scores[len(raw_ids) :], strict=True))
        raw_order = sorted(raw_ids, key=lambda u: (-raw_scores[u], u))
        k = min(self.selected_k, requested_top_k)
        # Reserve is a floor on intact raw records, never a forced derived quota.
        selected_raw = raw_order[: min(k, self.raw_reserve)]
        threshold = (
            raw_scores[raw_order[min(k, len(raw_order)) - 1]]
            if raw_order
            else -math.inf
        )
        protected = self.service.context_policy.render(
            scope,
            [
                {"id": u, "content": canonical[u].text_content, "score": raw_scores[u]}
                for u in selected_raw
            ],
            {u: canonical[u] for u in selected_raw},
        )
        family_room = max(
            0,
            self.max_context_chars
            - sum(len(h["content"]) for h in protected)
            - 2 * max(0, len(protected) - 1),
        )
        chosen = []
        seen = set()
        type_counts = {}
        decisions = {}
        for uid in sorted(union, key=lambda u: (-family_scores[u], u)):
            f = families[uid]
            if len(chosen) + len(selected_raw) >= k:
                for remaining in union:
                    if remaining not in decisions:
                        decisions[remaining] = "raw_reserve"
                break
            if family_scores[uid] < threshold:
                decisions[uid] = "rerank_low"
                continue
            if len(f.content) + 2 > family_room:
                decisions[uid] = "24k"
                continue
            if (
                f.derived_type == "profile"
                and type_counts.get("profile", 0) >= self.profile_budget
            ):
                decisions[uid] = "budget"
                continue
            # Same semantic interpretation/support family is redundant; same
            # source can still contain distinct facts, so source alone isn't ID.
            identity = (
                f.derived_type,
                f.semantic_text,
                tuple(sorted(f.supporting_raw)),
            )
            if identity in seen:
                decisions[uid] = "duplicate"
                continue
            seen.add(identity)
            decisions[uid] = "selected_provisionally"
            chosen.append(uid)
            family_room -= len(f.content) + 2
            type_counts[f.derived_type] = type_counts.get(f.derived_type, 0) + 1
        # A family containing an intact canonical source replaces that raw
        # packet, rather than consuming an additional duplicate evidence slot.
        intact_support = {
            u
            for fid in chosen
            for u in families[fid].supporting_raw
            if canonical[u].text_content in families[fid].content
        }
        selected_raw = [u for u in raw_order if u not in intact_support][
            : k - len(chosen)
        ]
        hits = [
            {"id": u, "content": canonical[u].text_content, "score": raw_scores[u]}
            for u in selected_raw
        ]
        hits = self.service.context_policy.render(
            scope, hits, {h["id"]: canonical[h["id"]] for h in hits}
        )
        hits.extend(
            {"id": u, "content": families[u].content, "score": family_scores[u]}
            for u in chosen
        )
        hits.sort(key=lambda h: (-h["score"], h["id"]))
        # Atomic families: no semantic statement detached from its provenance by
        # truncation. Oversize families fall back to original raw ranked records.
        packed = []
        used = 0
        returned = set()
        fallback_ids = raw_order[:100]
        fallback = self.service.context_policy.render(
            scope,
            [
                {"id": u, "content": canonical[u].text_content, "score": raw_scores[u]}
                for u in fallback_ids
            ],
            {u: canonical[u] for u in fallback_ids},
        )
        for hit in hits + fallback:
            if len(packed) >= k:
                break
            if hit["id"] in returned:
                continue
            room = self.max_context_chars - used - (2 if packed else 0)
            if len(hit["content"]) > room:
                if hit["id"] in families:
                    continue
                if room <= 0:
                    continue
                hit = {**hit, "content": hit["content"][:room]}
                if not hit["content"].strip():
                    continue
            packed.append(hit)
            returned.add(hit["id"])
            used += len(hit["content"]) + (2 if len(packed) > 1 else 0)
        packed.sort(key=lambda h: (-h["score"], h["id"]))
        actual = [u for u in chosen if u in returned]
        intact_support = {
            u
            for fid in actual
            for u in families[fid].supporting_raw
            if canonical[u].text_content in families[fid].content
        }
        audit = self.trace.get() or {}
        audit.update(
            semantic_mode=True,
            family_candidates=len(union),
            derived_selected=actual,
            selected_families={u: vars(families[u]) for u in actual},
            raw_retained=[h["id"] for h in packed if h["id"] not in families],
            raw_displaced=[
                u
                for u in raw_order[:k]
                if u not in returned and u not in intact_support
            ],
            intact_raw_in_families=sorted(intact_support),
            context_chars=len("\n\n".join(h["content"] for h in packed)),
            semantic_seconds=time.monotonic() - start,
            raw_reserve=self.raw_reserve,
            semantic_type_counts=type_counts,
            family_conversion={
                u: {
                    "after_late_fusion_rank": union.index(u) + 1,
                    "querit_score": family_scores[u],
                    "selected": u in actual,
                    "drop_reason": None
                    if u in actual
                    else "24k"
                    if decisions.get(u) == "selected_provisionally"
                    else decisions.get(u, "24k"),
                    "supporting_raw": families[u].supporting_raw,
                    "contained_raw_chars": families[u].raw_chars,
                    "family_chars": len(families[u].content),
                }
                for u in union
            },
        )
        self.trace.set(audit)
        return packed
