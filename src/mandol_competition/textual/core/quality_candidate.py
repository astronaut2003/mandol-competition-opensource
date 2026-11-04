"""Explicit Textual query/output candidates over the unchanged frozen substrate.

Startup retains the byte-identical storage/quality capsule. The v4 default
selects the separately versioned, user-approved observation-order output view.
Explicit research candidates remain independent; model recipes, physical
profile, SQLite owner and projection identity stay bound to the base capsule.
"""
import json
import os
import re
from pathlib import Path

from mandol_substrate.state.models import ProfileMismatch, digest
from .context_policy import ContextPolicy
from .query_policy import QueryPolicy
from .return_policy import ReturnPolicy
from .run_quality import materialize_policies


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ProfileMismatch("Duplicate quality-candidate field")
        out[key] = value
    return out


def bind_candidate(root, path, *, capsule, profile, baseline_sha256):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 65536:
        raise ProfileMismatch("Bounded quality-candidate file required")
    value = json.loads(path.read_text(), object_pairs_hook=_unique)
    schema = value.get("schema")
    fields = {"schema", "profile_id", "baseline_quality_sha256", "storage_profile_hash", "overrides"}
    if schema == "mandol-textual-quality-candidate-v2":
        fields.add("query_input_policy")
    if schema == "mandol-textual-quality-candidate-v3":
        fields.add("evidence_packaging_policy")
    if schema == "mandol-textual-quality-candidate-v4":
        fields.add("answer_utility_policy")
    if schema == "mandol-textual-quality-candidate-v5":
        fields.add("observation_order_policy")
    if (set(value) != fields
        or schema not in {"mandol-textual-quality-candidate-v1", "mandol-textual-quality-candidate-v2", "mandol-textual-quality-candidate-v3", "mandol-textual-quality-candidate-v4", "mandol-textual-quality-candidate-v5"}
        or not isinstance(value["profile_id"], str)
        or re.fullmatch(r"[a-z][a-z0-9_-]{3,95}", value["profile_id"]) is None
        or value["baseline_quality_sha256"] != baseline_sha256
        or value["storage_profile_hash"] != profile.hash
        or not isinstance(value["overrides"], dict)
        or not value["overrides"]
        or set(value["overrides"]) - {"query", "context", "returns"}):
        raise ProfileMismatch("Candidate identity or frozen baseline binding differs")
    if schema == "mandol-textual-quality-candidate-v2":
        from .query_inputs import QueryInputPolicy
        if not isinstance(value["query_input_policy"], dict):
            raise ProfileMismatch("Explicit query-input policy required")
        QueryInputPolicy(**value["query_input_policy"])
    if schema == "mandol-textual-quality-candidate-v3":
        from .evidence_packaging import EvidencePackagingPolicy
        if not isinstance(value["evidence_packaging_policy"], dict):
            raise ProfileMismatch("Explicit evidence-packaging policy required")
        EvidencePackagingPolicy(**value["evidence_packaging_policy"])
    if schema == "mandol-textual-quality-candidate-v4":
        from .answer_utility import AnswerUtilityPolicy
        if not isinstance(value["answer_utility_policy"],dict):
            raise ProfileMismatch("Explicit existing-evidence selection policy required")
        AnswerUtilityPolicy(**value["answer_utility_policy"])
    if schema == "mandol-textual-quality-candidate-v5":
        from .observation_order import ObservationOrderPolicy
        if not isinstance(value["observation_order_policy"], dict):
            raise ProfileMismatch("Explicit observation-order policy required")
        ObservationOrderPolicy(**value["observation_order_policy"])
    policies = dict(capsule["policies"])
    for name, override in value["overrides"].items():
        if not isinstance(override, dict) or "storage_profile_hash" in override:
            raise ProfileMismatch("Candidate cannot replace its storage binding")
        data = {**json.loads(policies[name]), **override}
        if name == "query":
            for field in ("leg_depths", "weights"):
                if field in data:
                    data[field] = tuple(tuple(row) for row in data[field])
            data["methods"] = tuple(data["methods"])
            QueryPolicy(**data).effective_profile(profile)
        elif name == "context":
            ContextPolicy(**data).validate(profile)
        else:
            ReturnPolicy(**data).validate(profile)
        policies[name] = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if schema == "mandol-textual-quality-candidate-v5" and any(
            json.loads(policies[name]) != json.loads(capsule["policies"][name]) for name in policies):
        raise ProfileMismatch("Observation-order candidate must retain every baseline retrieval/output policy")
    if schema == "mandol-textual-quality-candidate-v5":
        # Keep even the original policy serialization; a metadata-only
        # candidate has no reason to manufacture new frozen-policy bytes.
        policies = dict(capsule["policies"])
    # Never overwrite quality-profile.json or quality-policies.json. A changed
    # experiment ID/config gets an independent, checksum-checked policy cache.
    candidate_capsule = {"format": "explicit-candidate-v1", "baseline_capsule": digest(capsule),
                         "candidate": value, "policies": policies}
    identity_dir = Path(root).resolve() / "artifacts" / "quality-candidate-identities"
    if any(p.is_symlink() for p in (identity_dir, *identity_dir.parents)):
        raise ProfileMismatch("Candidate identity cannot traverse a symlink")
    identity_dir.mkdir(parents=True, exist_ok=True)
    identity = identity_dir / (value["profile_id"] + ".json")
    if identity.is_symlink():
        raise ProfileMismatch("Candidate identity symlink")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    try:
        with identity.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    if identity.read_bytes() != encoded:
        raise ProfileMismatch("Immutable candidate ID reused with another policy")
    paths = materialize_policies(root, candidate_capsule)
    metadata = {"id": value["profile_id"], "sha256": digest(value),
                   "baseline_quality_sha256": baseline_sha256,
                   "storage_profile_hash": profile.hash,
                   "status": "EXPERIMENTAL_NOT_PROMOTED"}
    if schema == "mandol-textual-quality-candidate-v2":
        metadata["query_input_policy"] = value["query_input_policy"]
        metadata["query_input_policy_hash"] = QueryInputPolicy(**value["query_input_policy"]).hash
    if schema == "mandol-textual-quality-candidate-v3":
        metadata["evidence_packaging_policy"] = value["evidence_packaging_policy"]
        metadata["evidence_packaging_policy_hash"] = EvidencePackagingPolicy(**value["evidence_packaging_policy"]).hash
    if schema == "mandol-textual-quality-candidate-v4":
        metadata["answer_utility_policy"]=value["answer_utility_policy"]
        metadata["answer_utility_policy_hash"]=AnswerUtilityPolicy(**value["answer_utility_policy"]).hash
    if schema == "mandol-textual-quality-candidate-v5":
        metadata["observation_order_policy"] = value["observation_order_policy"]
        metadata["observation_order_policy_hash"] = ObservationOrderPolicy(**value["observation_order_policy"]).hash
    return paths, metadata
