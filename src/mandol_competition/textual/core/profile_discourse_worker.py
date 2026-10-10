"""Discourse overlay using existing BuildJob receipts and publication coordinator.

One owning runtime per run (same ownership contract as the Competition service).
Profile/frame writers share this runtime's scope gate; no separate background writer.
Frames live in completed job progress, not in retrieval-eligible MemoryUnits.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import BoundedSemaphore, RLock

from mandol_substrate.state import BuildJob, Mutation, StateUnavailable
from mandol_substrate.state.models import digest

from .gates import ScopeGates
from .profile_delta_worker import _PreparedWorker, mutation_to_dict
from .profile_discourse import (
    DiscourseRecipe,
    chunks,
    compatible_existing,
    empty_frame,
    merge_assertions,
    messages_a,
    messages_c,
    structural_key,
    validate_a,
    validate_c,
)
from .user_profile import checked_response


def current_frame(state, scope, recipe, session_id=None):
    frontier = state.scope_state(scope)
    if (
        not frontier["canonical_seq"]
        == frontier["lexical_seq"]
        == frontier["published_seq"]
    ):
        raise StateUnavailable("Incomplete discourse publication frontier")
    session_filter = (
        " AND json_extract(progress,'$.discourse_session_id') IS ?"
        if recipe.session_scoped_frame
        else ""
    )
    rows = state._rows(
        """SELECT job_id,progress,commit_seq FROM build_jobs
        WHERE scope=? AND recipe_fingerprint=? AND status='completed'
        AND json_extract(progress,'$.discourse_root')=1 AND commit_seq<=?
        """
        + session_filter
        + " ORDER BY commit_seq DESC,job_id LIMIT 1",
        (scope.key, recipe.fingerprint, frontier["published_seq"])
        + ((session_id,) if recipe.session_scoped_frame else ()),
    )
    if not rows:
        return {"job_id": None, "version": 0, "frame": empty_frame()}
    progress = json.loads(rows[0]["progress"])
    return {
        "job_id": rows[0]["job_id"],
        "version": progress["frame_version"],
        "frame": progress["frame"],
    }


def structured_lookup(state, scope, candidate, recipe):
    subject, attribute, scope_value, condition = structural_key(candidate["item"])
    rows = state._rows(
        """SELECT m.logical_uid FROM memories m WHERE m.scope=? AND m.active=1
        AND m.lifecycle!='deleted' AND m.memory_type='user_profile_delta'
        AND json_extract(m.metadata,'$.recipe_hash')=?
        AND json_extract(m.metadata,'$.profile_delta_item.subject')=?
        AND json_extract(m.metadata,'$.profile_delta_item.attribute')=?
        AND json_extract(m.metadata,'$.profile_delta_item.scope') IS ?
        AND json_extract(m.metadata,'$.profile_delta_item.condition') IS ?
        AND json_extract(m.metadata,'$.profile_delta_item.status')='current'
        AND NOT EXISTS(SELECT 1 FROM graph_edges e WHERE e.scope=m.scope AND e.target_uid=m.logical_uid
            AND e.active=1 AND e.relation_type IN ('PROFILE_REPLACES','PROFILE_SUPERSEDES','PROFILE_RETRACTS'))
        ORDER BY m.commit_seq DESC,m.logical_uid LIMIT ?""",
        (
            scope.key,
            recipe.fingerprint,
            subject,
            attribute,
            scope_value,
            condition,
            recipe.max_existing + 1,
        ),
    )
    if len(rows) > recipe.max_existing:
        raise StateUnavailable("Profile target overflow; no arbitrary subset mutation")
    existing = state.get_memories(scope, [r["logical_uid"] for r in rows])
    return compatible_existing(scope, candidate, existing, limit=recipe.max_existing)


class _DiscourseReceiptWorker(_PreparedWorker):
    """Serialize only receipt publication/read-frontier checks within one Add.

    Without this, another chunk's in-flight T1 can look like an unrelated failed
    publication. GPT calls remain outside this short lock and run concurrently.
    """

    def __init__(self, *args, receipt_lock, **kwargs):
        super().__init__(*args, **kwargs)
        self.receipt_lock = receipt_lock

    def _resume_pending(self, scope, job):
        with self.receipt_lock:
            return super()._resume_pending(scope, job)

    def _apply(self, scope, stage, job, memories=(), edges=()):
        with self.receipt_lock:
            return super()._apply(scope, stage, job, memories, edges)


class DiscourseWorker(_PreparedWorker):
    def run_discourse(self, scope, new_uids):
        raw = self.state.get_memories(scope, new_uids)
        if not raw or set(raw) != set(new_uids) or len(set(new_uids)) != len(new_uids):
            raise StateUnavailable("Missing/duplicate raw input")
        sessions = {m.session_id for m in raw.values()}
        if self.recipe.session_scoped_frame and len(sessions) != 1:
            raise StateUnavailable("One discourse Add must belong to one exact session")
        session_id = next(iter(sessions)) if self.recipe.session_scoped_frame else None
        root = BuildJob.planned(
            scope, {u: m.version for u, m in raw.items()}, self.recipe.fingerprint
        )
        job = self._job(scope, root.job_id)
        if job and job.status == "completed":
            self._resume_pending(scope, root)
            return job.progress["result"]
        if job is None:
            with self.coordinator.search_view(scope):
                prior = current_frame(self.state, scope, self.recipe, session_id)
                ids = list(dict.fromkeys([*prior["frame"]["source_uids"], *new_uids]))
                sources = self.state.get_memories(scope, ids)
                if set(sources) != set(ids):
                    raise StateUnavailable("Missing persistent discourse source")
                positions = {u: i for i, u in enumerate(ids)}
                ids.sort(
                    key=lambda u: (
                        sources[u].commit_seq,
                        sources[u].turn_index or 0,
                        positions[u],
                    )
                )
                sources = {u: sources[u] for u in ids}
                job = replace(
                    root,
                    progress={
                        "discourse_root": True,
                        "new_uids": list(new_uids),
                        "raw_request_ids": sorted({m.request_id for m in raw.values()}),
                        "source_uids": ids,
                        "all_versions": {u: m.version for u, m in sources.items()},
                        "base_frame": prior,
                        **(
                            {"discourse_session_id": session_id}
                            if self.recipe.session_scoped_frame
                            else {}
                        ),
                        "stage": "discourse_intent",
                    },
                )
            self._apply(scope, "intent", job)
        else:
            # Root publication may be pending; receipt children recover their own outbox.
            pending = self.state._rows(
                "SELECT request_id FROM requests WHERE scope=? AND mutation_seq=?",
                (scope.key, self.state.scope_state(scope)["canonical_seq"]),
            )
            if pending and pending[0]["request_id"].startswith(root.job_id + ":"):
                self._resume_pending(scope, root)
        loaded = self.state.get_memories(scope, job.progress["source_uids"])
        # SQLite IN hydration has no ordering guarantee. Reapply frozen Add order.
        sources = {u: loaded[u] for u in job.progress["source_uids"] if u in loaded}
        if {u: m.version for u, m in sources.items()} != job.progress["all_versions"]:
            raise StateUnavailable("Stale discourse source versions")
        frame = job.progress["base_frame"]["frame"]
        groups = chunks(sources, new_uids, self.recipe, frame)
        receipt_lock = RLock()

        def extract(index_chunk):
            index, (src, core) = index_chunk
            child = _DiscourseReceiptWorker(
                self.coordinator,
                self.recipe,
                self.provider,
                self.fault,
                receipt_lock=receipt_lock,
            )
            child._scope, child._sources = scope, src
            inp = messages_a(scope, src, core, self.recipe, frame)
            receipt = child._receipt(inp, f"A{index}")
            return validate_a(
                checked_response(receipt), scope, src, core, self.recipe, frame
            ), [receipt]

        rejection = None
        a_results = []
        receipts = []
        ops = []
        existing = {}
        try:
            # GPT work overlaps, SQLite receipt transactions remain short.
            with ThreadPoolExecutor(max_workers=min(4, len(groups))) as pool:
                for result, receipt in pool.map(extract, enumerate(groups)):
                    a_results.append(result)
                    receipts.extend(receipt)
            merged = merge_assertions(a_results)
            if merged["defer_global"]:
                raise StateUnavailable(
                    "Global Profile erasure requires explicit complete operation"
                )
            for index, candidate in enumerate(merged["candidates"]):
                with self.coordinator.search_view(scope):
                    selected = structured_lookup(
                        self.state, scope, candidate, self.recipe
                    )
                self._scope, self._sources = scope, {**sources, **selected}
                inp = messages_c(candidate, selected)
                receipt = self._receipt(inp, f"C{index}")
                receipts.append(receipt)
                existing.update(selected)
                op = validate_c(checked_response(receipt), scope, candidate, selected)
                if op:
                    ops.append(op)
            targets = [o["target_uid"] for o in ops if o["target_uid"]]
            if len(targets) != len(set(targets)):
                raise ValueError("Multiple mutations of one target in an Add")
        except ValueError as exc:
            rejection = str(exc)[:200]
            ops = []
            # An invalid frame must not make the next bare first-person Add self-owned.
            # Preserve prior frame; surface explicit REJECTED, never fake success.
            merged = {"frame": frame, "disagreement_uids": [], "candidates": []}
        if (
            current_frame(self.state, scope, self.recipe, session_id)
            != job.progress["base_frame"]
        ):
            raise StateUnavailable("Discourse frame changed during preparation")
        all_versions = {u: m.version for u, m in {**sources, **existing}.items()}
        job_for_output = replace(
            job, progress={**job.progress, "prompt_hash": digest(receipts)}
        )
        receipt_summary = {
            "response_model": sorted({r["response_model"] for r in receipts}),
            "stage_receipt_hashes": [digest(r) for r in receipts],
        }
        memories, edges = self._outputs(
            scope,
            job_for_output,
            {"operations": ops},
            sources,
            existing,
            receipt_summary,
        )
        result = {
            "job_id": job.job_id,
            "output_uids": [m.logical_uid for m in memories],
            "status": "REJECTED" if rejection else "COMPLETED",
            "rejection": rejection,
            "operations": [o["op"] for o in ops],
            "frame": merged["frame"],
            "disagreement_uids": merged["disagreement_uids"],
        }
        complete = replace(
            job,
            status="completed",
            output_uids=tuple(result["output_uids"]),
            progress={
                **job.progress,
                "frame": merged["frame"],
                "frame_version": job.progress["base_frame"]["version"] + 1,
                "result": result,
                "receipt_hashes": [digest(r) for r in receipts],
            },
        )
        guard = replace(
            BuildJob.planned(scope, all_versions, digest([job.job_id, "input-guard"])),
            status="completed",
            progress={"parent_job": job.job_id},
        )
        mutation = Mutation(
            memories=tuple(memories),
            edges=tuple(edges),
            jobs=(guard, complete),
            spaces=("hlm:profile",) if memories else (),
            kind="quality_profile_discourse",
        )
        self.fault("after_mutation_receipt")
        self.fault("before_profile_publication")
        self.coordinator.apply(
            scope,
            job.job_id + ":completed",
            digest(mutation_to_dict(mutation)),
            mutation,
            result,
        )
        self.fault("after_profile_publication")
        return result


class ProfileDiscourseRuntime:
    def __init__(
        self, coordinator, provider, *, fault=None, max_concurrent_gpt=4, recipe=None
    ):
        if max_concurrent_gpt not in (4, 8):
            raise ValueError("Only qualified bounded GPT concurrency candidates")
        self.coordinator, self.provider, self.fault = coordinator, provider, fault
        self.recipe = recipe or DiscourseRecipe(coordinator.state.profile_hash)
        self.gates = coordinator.__dict__.setdefault(
            "_profile_discourse_gates", ScopeGates(timeout=600, max_inflight_scopes=16)
        )
        self._slots = BoundedSemaphore(max_concurrent_gpt)

    def _generate(self, inp):
        if not self._slots.acquire(timeout=300):
            raise StateUnavailable("Profile GPT capacity unavailable")
        try:
            return self.provider(inp)
        finally:
            self._slots.release()

    def run(self, scope, new_uids):
        with self.gates.lease(scope, write=True):
            frontier = self.coordinator.state.scope_state(scope)
            if frontier["canonical_seq"] != frontier["published_seq"]:
                rows = self.coordinator.state._rows(
                    "SELECT request_id,payload_digest FROM requests WHERE scope=? AND mutation_seq=?",
                    (scope.key, frontier["canonical_seq"]),
                )
                if len(rows) != 1:
                    raise StateUnavailable("Pending publication lacks unique request")
                # Exact durable outbox recovery; never regenerate/replay lexical deltas.
                self.coordinator.apply(
                    scope,
                    rows[0]["request_id"],
                    rows[0]["payload_digest"],
                    Mutation(),
                    {},
                )
            # Resume earlier incomplete root before a later Add can advance its frame.
            pending = self.coordinator.state._rows(
                """SELECT progress FROM build_jobs WHERE scope=?
                AND recipe_fingerprint=? AND status!='completed'
                AND json_extract(progress,'$.discourse_root')=1 ORDER BY commit_seq""",
                (scope.key, self.recipe.fingerprint),
            )
            for row in pending:
                ids = json.loads(row["progress"])["new_uids"]
                if ids != list(new_uids):
                    DiscourseWorker(
                        self.coordinator, self.recipe, self._generate, self.fault
                    ).run_discourse(scope, ids)
            return DiscourseWorker(
                self.coordinator, self.recipe, self._generate, self.fault
            ).run_discourse(scope, new_uids)
