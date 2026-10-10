"""Frozen HTTP output budget; canonical and reranker input text stay unchanged."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from mandol_substrate.state.models import ProfileMismatch, digest


@dataclass(frozen=True)
class ReturnPolicy:
    storage_profile_hash: str
    final_k: int = 10
    max_context_chars: int = 24000
    version: str = "explicit-api-content-budget-v1"

    def __post_init__(self):
        if (
            self.version != "explicit-api-content-budget-v1"
            or type(self.final_k) is not int
            or not 1 <= self.final_k <= 100
            or type(self.max_context_chars) is not int
            or not 128 <= self.max_context_chars <= 100000
            or not isinstance(self.storage_profile_hash, str)
            or len(self.storage_profile_hash) != 64
            or any(c not in "0123456789abcdef" for c in self.storage_profile_hash)
        ):
            raise ProfileMismatch("Invalid explicit API output budget")

    @property
    def hash(self):
        return digest(asdict(self))

    def validate(self, profile):
        if profile.hash != self.storage_profile_hash:
            raise ProfileMismatch("Output budget belongs to another physical profile")

    @classmethod
    def load(cls, path):
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result:
                    raise ProfileMismatch("Duplicate output-budget field")
                result[k] = v
            return result

        return cls(**json.loads(Path(path).read_text(), object_pairs_hook=unique))

    def materialize(self, hits, requested_top_k):
        if (
            type(requested_top_k) is not int
            or not 1 <= requested_top_k <= 100
            or len(hits) > requested_top_k
        ):
            raise ValueError("Actual response must honor the incoming top_k")
        ids = []
        for hit in hits:
            if (
                any(
                    not isinstance(hit.get(k), str) or not hit[k].strip()
                    for k in ("id", "content")
                )
                or hit["id"] in ids
            ):
                raise ValueError("Malformed/duplicate source response")
            if hit.get("score") is not None and (
                type(hit["score"]) not in (int, float)
                or not math.isfinite(hit["score"])
            ):
                raise ValueError(
                    "Missing score is allowed; fake nonfinite score is not"
                )
            ids.append(hit["id"])
        result, pieces, used = [], [], 0
        for hit in hits[: min(self.final_k, requested_top_k)]:
            available = self.max_context_chars - used - bool(result)
            if available <= 0:
                break
            content = hit["content"][:available]
            if not content.strip():
                # Do not emit an invalid empty-content success record or add
                # invented text merely to fill a nearly exhausted budget.
                raise ValueError("Budget clips a record to whitespace only")
            result.append({**hit, "content": content})
            used += len(content) + (len(result) > 1)
            pieces.append(
                {
                    "id": hit["id"],
                    "original_chars": len(hit["content"]),
                    "returned_chars": len(content),
                    "complete": content == hit["content"],
                    "source_text_sha256": digest(hit["content"]),
                    "returned_text_sha256": digest(content),
                }
            )
        return result, {
            "policy_hash": self.hash,
            "incoming_top_k": requested_top_k,
            "input_ids": ids,
            "returned_ids": [r["id"] for r in result],
            "joined_context_chars": used,
            "pieces": pieces,
            "canonical_text_modified": False,
            "budget_semantics": "Explicit local quality profile; not organizer-prescribed limit",
        }
