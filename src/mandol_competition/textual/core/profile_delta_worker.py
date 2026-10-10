"""Durable experimental profile deltas using the existing publication jobs.

One runtime per Competition service; all profile writers share its scope leases.
Raw publication remains a separate, already-durable step. The experimental API
must hold its outer Add/Search lease until this complete profile frontier exists.
"""
from __future__ import annotations

from dataclasses import replace
from threading import BoundedSemaphore

from mandol_substrate.state import (
    BuildJob,
    CanonicalMemory,
    Edge,
    Mutation,
    StateUnavailable,
)
from mandol_substrate.state.models import digest

from .derivation import DerivationWorker
from .gates import ScopeGates
from .profile_delta import DeltaRecipe, messages, validate
from .user_profile import ProfileWorker, checked_response


def current_delta_ids(state, scope, *, limit=100):
    if not 0 < limit <= 100:
        raise ValueError("Bounded profile query required")
    frontier = state.scope_state(scope)
    if not frontier["canonical_seq"] == frontier["lexical_seq"] == frontier["published_seq"]:
        raise StateUnavailable("Profile frontier incomplete")
    return [r["logical_uid"] for r in state._rows("""SELECT m.logical_uid FROM memories m
        WHERE m.scope=? AND m.memory_type='user_profile_delta' AND m.active=1
        AND m.lifecycle!='deleted' AND m.commit_seq<=?
        AND json_extract(m.metadata,'$.profile_delta_item.status')='current'
        AND NOT EXISTS(SELECT 1 FROM graph_edges e WHERE e.scope=m.scope
          AND e.target_uid=m.logical_uid AND e.active=1
          AND e.relation_type IN ('PROFILE_REPLACES','PROFILE_SUPERSEDES','PROFILE_RETRACTS'))
        ORDER BY m.commit_seq DESC,m.logical_uid LIMIT ?""", (scope.key, frontier["published_seq"], limit))]


class _PreparedWorker(DerivationWorker):
    # Reuse the existing bounded durable provider-receipt algorithm verbatim.
    _receipt = ProfileWorker._receipt

    def __init__(self, coordinator, recipe, provider, fault):
        super().__init__(coordinator, recipe, provider, fault=fault)
        self.provider = provider
        self._scope = self._sources = None

    def run_delta(self, scope, new_uids, context_uids):
        raw = self.state.get_memories(scope, new_uids)
        if set(raw) != set(new_uids):
            raise StateUnavailable("Missing raw profile input")
        root = BuildJob.planned(scope, {u: m.version for u, m in raw.items()}, self.recipe.fingerprint)
        job = self._job(scope, root.job_id)
        frontier = self.state.scope_state(scope)
        if frontier["canonical_seq"] != frontier["published_seq"]:
            pending = self.state._rows("SELECT request_id FROM requests WHERE scope=? AND mutation_seq=?",
                (scope.key, frontier["canonical_seq"]))
            if not job or any(r["request_id"].startswith(root.job_id + ":") for r in pending):
                self._resume_pending(scope, root)
            # A pending receipt child is reconciled by _receipt after its exact
            # input-derived job identity is reconstructed. Unrelated jobs fail.
        if job and job.status == "completed":
            return job.progress["result"]
        if job is None:
            with self.coordinator.search_view(scope):
                ids = current_delta_ids(self.state, scope, limit=100)
                recent = ids[:4]
                # SQLite indexed BM25 is a selector, never a Python postings map.
                query = " ".join(m.text_content for m in raw.values())[:4000]
                found = self.state.lexical.search(scope, query, limit=self.recipe.max_existing,
                    candidate_uids=ids) if ids else []
                ranked = [h.logical_uid for h in found]
                selected = list(dict.fromkeys([*ranked, *recent]))[:self.recipe.max_existing]
                sources = self.state.get_memories(scope, list(dict.fromkeys([*context_uids, *new_uids])))
                existing = self.state.get_memories(scope, selected)
                inp = messages(scope, sources, new_uids, existing, self.recipe)
                job = replace(root, progress={"new_uids": list(new_uids), "context_uids": list(context_uids),
                    "existing_uids": selected, "all_versions": {u: m.version for u, m in {**sources, **existing}.items()},
                    "prompt_hash": digest(inp), "stage": "profile_delta_intent"})
            self._apply(scope, "intent", job)
        sources = self.state.get_memories(scope, list(dict.fromkeys([*job.progress["context_uids"], *new_uids])))
        existing = self.state.get_memories(scope, job.progress["existing_uids"])
        self._scope, self._sources = scope, {**sources, **existing}
        if {u: m.version for u, m in self._sources.items()} != job.progress["all_versions"]:
            raise StateUnavailable("Stale profile proposal input; explicit replan required")
        inp = messages(scope, sources, new_uids, existing, self.recipe)
        if digest(inp) != job.progress["prompt_hash"]:
            raise StateUnavailable("Frozen profile input changed")
        receipt = self._receipt(inp, "delta")
        try:
            proposal = validate(checked_response(receipt), scope, sources, new_uids, existing, self.recipe)
            rejection = None
        except ValueError as exc:
            proposal = None
            rejection = {"status": "REJECTED", "reason": str(exc)[:200]}
        if proposal and proposal["defer_global"]:
            raise StateUnavailable("Broad erasure exceeds bounded profile context; not completed")
        memories, edges = self._outputs(scope, job, proposal, sources, existing, receipt)
        result = {"job_id": job.job_id, "output_uids": [m.logical_uid for m in memories],
            "status": "REJECTED" if rejection else "COMPLETED", "rejection": rejection,
            "operations": [] if proposal is None else [o["op"] for o in proposal["operations"]]}
        complete = replace(job, status="completed", output_uids=tuple(result["output_uids"]),
            progress={**job.progress, "result": result, "receipt_hash": digest(receipt)})
        guard = BuildJob.planned(scope, job.progress["all_versions"], digest([job.job_id, "input-guard"]))
        guard = replace(guard, status="completed", progress={"parent_job": job.job_id})
        mutation = Mutation(memories=tuple(memories), edges=tuple(edges), jobs=(guard, complete),
            spaces=("hlm:profile",) if memories else (), kind="quality_profile_delta")
        self.fault("before_profile_publication")
        self.coordinator.apply(scope, job.job_id + ":completed", digest(mutation_to_dict(mutation)), mutation, result)
        self.fault("after_profile_publication")
        return result

    def _outputs(self, scope, job, proposal, sources, existing, receipt):
        memories, edges = [], []
        for op in proposal["operations"] if proposal else []:
            target = existing.get(op["target_uid"])
            evidence = op["evidence"]
            new_evidence = [e for e in evidence if e["source_uid"] in job.progress["new_uids"] and sources[e["source_uid"]].role == "user"]
            source = sources[new_evidence[0]["source_uid"]]
            if op["op"] == "CLOSE":
                edges.append(Edge("edge_" + digest([scope.key, job.job_id, "close", target.logical_uid]),
                    source.logical_uid, target.logical_uid, "PROFILE_RETRACTS",
                    provenance={"evidence": evidence, "job_id": job.job_id}))
                continue
            value = target.metadata["profile_delta_item"] if op["op"] == "REINFORCE" else op["item"]
            uid = "profile_" + digest([scope.key, job.job_id, op])
            lineage = target.metadata["profile_id"] if target else uid
            source_ids = list(dict.fromkeys(e["source_uid"] for e in evidence))
            metadata = {"user_id": scope.user_id, "hlm_profile": True, "profile_delta_item": value,
                "profile_id": lineage, "profile_version": 1 if target is None else target.metadata["profile_version"] + 1,
                "source_uids": source_ids, "source_spans": evidence,
                "source_versions": {u: sources[u].version for u in source_ids},
                "context_versions": {u: sources[u].version for u in op["context_uids"]},
                "context_uids": op["context_uids"], "recipe_hash": self.recipe.fingerprint,
                "prompt_hash": job.progress["prompt_hash"], "response_hash": digest(receipt),
                "model": self.recipe.model, "provider": self.recipe.provider,
                "response_model": receipt["response_model"], "job_id": job.job_id,
                "operation": op["op"], "previous_profile_uid": target.logical_uid if target else None}
            memories.append(CanonicalMemory(uid, value["statement"], raw_data={"profile_item": value},
                metadata=metadata, role="user", session_id=source.session_id, request_id=job.job_id + ":completed",
                memory_level="L1_Summary", memory_type="user_profile_delta", spaces=("hlm:profile",),
                business_timestamp_ms=source.business_timestamp_ms))
            for e in evidence:
                edges.append(Edge("edge_" + digest([scope.key, uid, "DERIVED_FROM", e["source_uid"]]),
                    uid, e["source_uid"], "DERIVED_FROM", provenance={"quote": e["quote"], "job_id": job.job_id}))
            if target:
                relation = "PROFILE_SUPERSEDES" if op["op"] == "SUPERSEDE" else "PROFILE_REPLACES"
                edges.append(Edge("edge_" + digest([scope.key, uid, relation, target.logical_uid]),
                    uid, target.logical_uid, relation, provenance={"op": op["op"], "job_id": job.job_id}))
        return memories, edges


def mutation_to_dict(mutation):
    from dataclasses import asdict
    return asdict(mutation)


class ProfileDeltaRuntime:
    """Shared same-user serialization with independent users running concurrently."""
    def __init__(self, coordinator, provider, *, fault=None, max_concurrent_gpt=4):
        if type(max_concurrent_gpt) is not int or not 1 <= max_concurrent_gpt <= 8:
            raise ValueError("Profile GPT concurrency must be bounded to 1..8")
        self.coordinator, self.provider, self.fault = coordinator, provider, fault
        self.recipe = DeltaRecipe(coordinator.state.profile_hash)
        self.gates = ScopeGates(timeout=300, max_inflight_scopes=16)
        self._model_slots = BoundedSemaphore(max_concurrent_gpt)

    def _generate(self, inp):
        if not self._model_slots.acquire(timeout=300):
            raise StateUnavailable("Profile GPT preparation capacity unavailable")
        try:
            return self.provider(inp)
        finally:
            self._model_slots.release()

    def run(self, scope, new_uids, *, context_uids=()):
        with self.gates.lease(scope, write=True):
            worker = _PreparedWorker(self.coordinator, self.recipe, self._generate, self.fault)
            return worker.run_delta(scope, new_uids, context_uids)
