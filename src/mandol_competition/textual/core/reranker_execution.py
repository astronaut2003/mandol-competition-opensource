"""Frozen Querit pair construction, admission checks and score policy."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mandol_competition.textual.core.native_reranker_http import validate_profile
from mandol_competition.textual.core.querit_reference import expected_profile
from mandol_substrate.state.models import ProfileMismatch, digest


@dataclass(frozen=True)
class RerankerExecutionPolicy:
    storage_profile_hash: str
    reference_profile_hash: str
    batch_size: int = 1
    version: str = "querit-original-order-microbatch-v1"

    def __post_init__(self):
        if (
            self.version != "querit-original-order-microbatch-v1"
            or type(self.batch_size) is not int
            or self.batch_size not in (1, 8)
        ):
            raise ProfileMismatch("Unsupported explicit reranker execution policy")
        for value in (self.storage_profile_hash, self.reference_profile_hash):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ProfileMismatch("Complete physical/reference hashes required")

    @property
    def hash(self):
        return digest(asdict(self))

    @classmethod
    def load(cls, path):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ProfileMismatch("Duplicate execution-policy field")
                result[key] = value
            return result

        return cls(**json.loads(Path(path).read_text(), object_pairs_hook=unique))

    def effective_options(self, profile):
        if profile.hash != self.storage_profile_hash:
            raise ProfileMismatch(
                "Execution overlay belongs to another physical profile"
            )
        options = profile.options
        original = options.get("reranker_serving")
        if (
            options.get("reranker_backend") != "native_reference_http"
            or original != expected_profile()
            or digest(original) != self.reference_profile_hash
        ):
            raise ProfileMismatch("Original pinned Querit B8/SDPA reference required")
        effective = copy.deepcopy(options)
        effective["reranker_serving"]["batch_size"] = self.batch_size
        validate_profile(effective["reranker_serving"])
        return effective

    def attestation(self, profile):
        effective = self.effective_options(profile)
        return {
            "execution_policy_hash": self.hash,
            "physical_profile_hash": self.storage_profile_hash,
            "reference_profile_hash": self.reference_profile_hash,
            "actual_native_profile_hash": digest(effective["reranker_serving"]),
            "actual_native_profile": effective["reranker_serving"],
            "changed_fields": [] if self.batch_size == 8 else ["batch_size"],
            "qualification": "CANDIDATE_REQUIRES_REAL_HTTP_PAIRED_QA_AND_RESOURCE_GATES",
        }
