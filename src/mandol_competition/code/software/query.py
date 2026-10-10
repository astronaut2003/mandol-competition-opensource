"""Optional, query-only intent and evidence-need features."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodingQueryPlan:
    intent: str
    asks_for_failure: bool
    evidence_needs: tuple[str, ...] = ()
    producer: str = "heuristic-v1"


class CodingQueryRouter:
    def route(self, query: str) -> CodingQueryPlan:
        debug = re.search(r"\b(?:bug|debug|error|exception|traceback|crash|regression|fails?|failed|failure)\b|报错|异常|失败|修复", query, re.I)
        failure = re.search(r"what (?:failed|did not work)|failed (?:attempt|approach)|avoid repeating|失败方案|无效方案", query, re.I)
        return CodingQueryPlan("DEBUG" if debug else "DEVELOPMENT", bool(failure))


class EvidenceNeedDecomposer:
    """Bounded extractive clauses; never synthesize a target solution."""

    def decompose(self, query: str, max_needs: int = 5) -> tuple[str, ...]:
        if not 2 <= max_needs <= 5:
            raise ValueError("Evidence needs must be bounded to 2–5")
        clauses = [part.strip(" .?\n") for part in re.split(r"[;\n?]+|\bwhile\b|\bwithout\b|\bpreserving\b", query, flags=re.I)]
        clauses = list(dict.fromkeys(part for part in clauses if len(part.split()) >= 3))
        return tuple(clauses[:max_needs]) if len(clauses) >= 2 else ()
