"""Durable conservative consolidation plans over existing canonical derived items.

No originals are deleted. Group/member indexes live in the existing BuildJob
receipt; Search may consume a checked plan, but never writes a plan itself.
"""

import json
from dataclasses import asdict, dataclass, replace
from typing import Literal

from pydantic import Field

from mandol_substrate.state import BuildJob, Mutation, StateUnavailable
from mandol_substrate.state.models import digest
from .conservative_dedup import groups
from .heterogeneous_memory import HeterogeneousRecipe, Strict, Support
from .heterogeneous_worker import typed_scope


class PairDecision(Strict):
    decision: Literal[
        "KEEP_SEPARATE", "SAME_ASSERTION", "SAME_OCCURRENCE_MERGE", "ENTITY_RESOLUTION"
    ]
    reason: str = Field(min_length=1, max_length=1000)
    evidence: list[Support] = Field(max_length=8)


@dataclass(frozen=True)
class ConsolidationRecipe(HeterogeneousRecipe):
    version: str = "conservative-consolidation-v3"

    @property
    def fingerprint(self):
        return digest([asdict(self), "immutable-group-receipt-v1"])


class ConsolidationPlanner:
    def __init__(self, runtime):
        self.runtime = runtime
        self.state = runtime.state

    def plan(self, scope, kind, ids, *, mode="structural", pair_limit=4096):
        if kind not in ("profile", "episode", "entity", "relation"):
            raise ValueError("No V3 consolidation authority for this type")
        if type(pair_limit) is not int or not 1 <= pair_limit <= 131072:
            raise ValueError("Explicit bounded administrative pair budget required")
        partition = typed_scope(scope, kind, self.runtime.recipe)
        with self.runtime.gates.lease(scope, write=True):
            self.runtime._recover(partition)
            memories = self.runtime.eligible(
                scope, self.state.get_memories(partition, ids)
            )
            if set(memories) != set(ids) or len(ids) != len(set(ids)):
                raise StateUnavailable("Stale/foreign/unpublished consolidation member")
            planned = groups(kind, memories, mode=mode, pair_limit=pair_limit)
            frozen = [asdict(g) for g in planned]
            progress = {
                "groups": frozen,
                "backlinks": [link for g in planned for link in g.backlinks],
                "source_hashes": {u: digest(m.metadata) for u, m in memories.items()},
                "mode": mode,
                "kind": kind,
                "source_records_deleted": 0,
            }
            progress["plan_hash"] = digest(progress)
            job = BuildJob.planned(
                partition,
                {u: m.version for u, m in memories.items()},
                digest(["consolidation-v3", self.runtime.recipe.fingerprint, progress]),
            )
            existing = self.state._rows(
                "SELECT status,progress FROM build_jobs WHERE scope=? AND job_id=?",
                (partition.key, job.job_id),
            )
            if existing:
                saved = json.loads(existing[0]["progress"])
                if existing[0]["status"] != "completed" or saved != progress:
                    raise StateUnavailable("Consolidation receipt mismatch")
                return saved
            completed = replace(job, status="completed", progress=progress)
            mutation = Mutation(jobs=(completed,), kind="derived_consolidation_plan")
            self.runtime.publish_mutation(
                partition,
                job.job_id + ":consolidation",
                digest(asdict(mutation)),
                mutation,
                {"plan_hash": progress["plan_hash"]},
            )
            return progress
