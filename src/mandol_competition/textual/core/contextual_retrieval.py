"""V3 optional contextual base route and utility-gated semantic families."""

import time
from contextlib import contextmanager
from contextvars import ContextVar

from mandol_substrate.state import StateUnavailable
from .conservative_dedup import representatives
from .heterogeneous_worker import typed_scope
from .query_policy import QueryPolicy
from .retrieval import CompetitionRetriever
from .semantic_evidence import SemanticEvidenceNavigation


class ArchitectureV3Navigation(SemanticEvidenceNavigation):
    def __init__(self, *args, contextual=None, architecture_policy, **kwargs):
        if any(r.kind == "hierarchical" for r in args[2]):
            raise ValueError("Hierarchical has no V3 semantic route")
        super().__init__(
            *args,
            raw_reserve=8 if architecture_policy.packing == "conservative" else 6,
            profile_budget=1 if architecture_policy.packing == "conservative" else 2,
            semantic_evidence_enabled=architecture_policy.semantic_evidence_enabled,
            **kwargs,
        )
        self.contextual = contextual
        self.architecture_policy = architecture_policy
        self.context_mode = ContextVar(
            "prefix_mode", default=architecture_policy.contextual_enabled
        )
        self.render_mode = ContextVar(
            "prefix_render", default=architecture_policy.prefix_rendering_enabled
        )
        self.current_views = ContextVar("current_prefixes", default=None)
        self.original_raw = ContextVar("original_raw", default=())
        self.utility_audit = ContextVar("family_utility", default=None)

    @contextmanager
    def select_context(self, *, enabled, render=False):
        a = self.context_mode.set(enabled)
        b = self.render_mode.set(render)
        try:
            yield
        finally:
            self.context_mode.reset(a)
            self.render_mode.reset(b)

    def expand(self, scope, view, query, dense, sparse, raw_ids, canonical):
        self.original_raw.set(tuple(raw_ids))
        self.utility_audit.set({})
        ids, canonical, audit = super().expand(
            scope, view, query, dense, sparse, raw_ids, canonical
        )
        # Pairwise structural decisions are immutable plans. Search never merges
        # canonical records or mutates lifecycle, and Profile isn't clustered.
        typed = list(self.typed_candidates.get())
        filtered = []
        audits = {}
        for kind in sorted({k for k, _ in typed}):
            current = {m.logical_uid: m for k, m in typed if k == kind}
            selected, detail = representatives(
                kind, current, mode=self.architecture_policy.dedup
            )
            filtered.extend(
                (kind, m) for k, m in typed if k == kind and m.logical_uid in selected
            )
            audits[kind] = detail
        self.typed_candidates.set(tuple(filtered))
        audit["dedup"] = audits
        self.current_views.set({})
        if not self.context_mode.get() or self.contextual is None:
            self.trace.set(audit)
            return ids, canonical, audit
        tick = time.monotonic()
        partition = typed_scope(scope, "contextual", self.contextual.recipe)
        if not self.service.state._rows(
            "SELECT 1 FROM scopes WHERE scope=?", (partition.key,)
        ):
            audit["contextual"] = {"status": "ABSENT_RAW_FALLBACK"}
            self.trace.set(audit)
            return ids, canonical, audit
        policy = QueryPolicy(
            self.service.profile.hash,
            self.service.profile.methods,
            tuple((m, 100) for m in self.service.profile.methods),
            fused_k=100,
            graph_hops=0,
            semantic_neighbors=0,
        )
        retriever = CompetitionRetriever(
            self.service.state,
            self.service.projection,
            self.service.profile,
            query_policy=policy,
        )
        try:
            with self.service.coordinator.search_view(partition) as typed_view:
                ranked, tower_audit = retriever.retrieve(
                    typed_view, query, dense, sparse, top_k=100
                )
                records = self.service.state.get_memories(
                    partition, [m.uid for m, _ in ranked]
                )
                valid = self.contextual.eligible(scope, records)
                views = {}
                added = []
                for hit, _ in ranked:
                    if hit.uid not in valid:
                        continue
                    m = valid[hit.uid]
                    item = m.metadata["hlm_item"]
                    uid = item["source_uid"]
                    raw = self.service.state.get_memories(scope, [uid]).get(uid)
                    if (
                        raw is None
                        or not self.packer._raw(scope, raw, view.published_seq)
                        or m.commit_seq > typed_view.published_seq
                    ):
                        continue
                    canonical[uid] = raw
                    views[uid] = item
                    if uid not in ids and len(added) < 40:
                        ids.append(uid)
                        added.append(uid)
                self.current_views.set(views)
                audit["contextual"] = {
                    "status": "COMPLETE",
                    "eligible_views": len(views),
                    "additional_raw": added,
                    "ranked_view_ids": [m.uid for m, _ in ranked],
                    "tower_audit": {
                        k: v for k, v in tower_audit.items() if k != "canonical"
                    },
                    "prefix_chars": sum(len(v["prefix"]) for v in views.values()),
                    "seconds": time.monotonic() - tick,
                }
        except StateUnavailable as exc:
            audit["contextual"] = {
                "status": "EXPLICIT_RAW_FALLBACK",
                "error_type": type(exc).__name__,
            }
        self.trace.set(audit)
        return ids, canonical, audit

    def _family(self, scope, view, kind, memory, canonical):
        # Existing public audits do not qualify Relation semantic evidence. It
        # still provides bounded navigation. Entity is navigation-only as before.
        if kind not in ("episode", "profile"):
            return None
        family = super()._family(scope, view, kind, memory, canonical)
        if family is None:
            return None
        ratio = family.raw_chars / len(family.content) if family.content else 0
        additional = set(family.supporting_raw) - set(self.original_raw.get())
        synthesis = (
            self.architecture_policy.synthesis_utility_enabled
            and len(set(family.supporting_raw)) >= 2
        )
        reason = (
            "SOURCE_COMPRESSION_CANDIDATE"
            if ratio > 1
            else "ADDITIONAL_SOURCE_BRIDGE"
            if additional
            else "CROSS_SOURCE_SYNTHESIS_CANDIDATE"
            if synthesis
            else "NO_ADDED_SEMANTIC_UTILITY"
        )
        self.utility_audit.set(
            {
                **(self.utility_audit.get() or {}),
                family.family_id: {
                    "reason": reason,
                    "compression_ratio": ratio,
                    "contained_raw_chars": family.raw_chars,
                    "derived_semantic_chars": len(family.semantic_text),
                    "family_chars": len(family.content),
                    "information_preservation": "REQUIRES_SEMANTIC_QA_AUDIT",
                    "additional_sources": sorted(additional),
                    "eligible": ratio > 1 or bool(additional) or synthesis,
                    "cross_source_synthesis_candidate": synthesis,
                },
            }
        )
        if ratio <= 1 and not additional and not synthesis:
            return None
        return family

    def finalize(self, scope, view, query, raw_ids, canonical, requested_top_k):
        hits = super().finalize(scope, view, query, raw_ids, canonical, requested_top_k)
        audit = self.trace.get() or {}
        audit["semantic_utility"] = self.utility_audit.get()
        audit["hierarchical_semantic_route"] = False
        self.trace.set(audit)
        if not self.render_mode.get() or not self.current_views.get():
            return hits
        if hits is None:
            scores = self.service.reranker.rerank(
                query, [canonical[u].text_content for u in raw_ids]
            )
            order = sorted(
                zip(raw_ids, scores, strict=True), key=lambda x: (-x[1], x[0])
            )
            # Existing frozen context and return policies still own raw rendering.
            hits = self.service.context_policy.render(
                scope,
                [
                    {"id": u, "content": canonical[u].text_content, "score": s}
                    for u, s in order[: min(10, requested_top_k)]
                ],
                {u: canonical[u] for u, _ in order[: min(10, requested_top_k)]},
            )
            hits, _ = self.service.return_policy.materialize(hits, requested_top_k)
        result = []
        used = 0
        for hit in hits:
            item = self.current_views.get().get(hit["id"])
            prefix = (
                f"[Retrieval context interpretation]\n{item['prefix']}\n"
                if item and item["prefix"]
                else ""
            )
            # Do not clip raw to buy prefix room. A prefix is optional; skip it
            # when the unchanged full response would otherwise exceed24k.
            remaining_raw = sum(
                len(h["content"]) for h in hits[len(result) :]
            ) + 2 * max(0, len(hits) - len(result) - 1)
            if (
                prefix
                and used + remaining_raw + len(prefix) + (2 if result else 0) <= 24000
            ):
                hit = {**hit, "content": prefix + hit["content"]}
            result.append(hit)
            used += len(hit["content"]) + (2 if len(result) > 1 else 0)
        if (
            len(result) > min(10, requested_top_k)
            or len("\n\n".join(h["content"] for h in result)) > 24000
        ):
            raise ValueError("V3 exceeds frozen context bound")
        audit["prefix_rendering"] = True
        audit["context_chars"] = len("\n\n".join(h["content"] for h in result))
        self.trace.set(audit)
        return result
