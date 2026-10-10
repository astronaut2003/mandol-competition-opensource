"""Portable copies of the five frozen quality policies, never semantic records.

Original absolute bindings remain provenance. Runtime resolution uses only this
small checksum-verified capsule, so moving a run needs no original workspace.
"""

import hashlib
import json
from pathlib import Path

from mandol_substrate.state.models import RetrievalProfile, StateUnavailable, digest

POLICY_NAMES = ("profile", "query", "context", "execution", "returns")
CAPSULE_NAME = "quality-policies.json"


def validate_capsule(capsule, quality, profile):
    if (set(capsule) != {"format", "quality_hash", "policies"}
            or capsule["format"] != 1 or capsule["quality_hash"] != digest(quality)
            or set(capsule["policies"]) != set(POLICY_NAMES)
            or set(quality.get("components", {})) != set(POLICY_NAMES)):
        raise StateUnavailable("portable quality policy manifest mismatch")
    for name in POLICY_NAMES:
        text = capsule["policies"][name]
        if (not isinstance(text, str) or len(text.encode("utf-8")) > 1024 * 1024
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != quality["components"][name]["sha256"]):
            raise StateUnavailable("portable quality policy checksum mismatch")
        json.loads(text)
    if RetrievalProfile(**json.loads(capsule["policies"]["profile"])).hash != profile.hash:
        raise StateUnavailable("portable quality policy changes semantic profile")
    return capsule


def capture_capsule(quality, profile):
    """Explicit creation reads bound configuration files, not datasets/models."""
    if set(quality.get("components", {})) != set(POLICY_NAMES):
        raise StateUnavailable("five frozen quality component bindings required")
    policies = {}
    for name in POLICY_NAMES:
        path = Path(quality["components"][name]["file"])
        if path.stat().st_size > 1024 * 1024:
            raise StateUnavailable("quality policy exceeds bounded configuration size")
        policies[name] = path.read_bytes().decode("utf-8")
    return validate_capsule({"format": 1, "quality_hash": digest(quality), "policies": policies}, quality, profile)


def load_capsule(root, quality, profile):
    path = Path(root) / CAPSULE_NAME
    if path.is_symlink() or path.stat().st_size > 6 * 1024 * 1024:
        raise StateUnavailable("invalid portable quality capsule")
    return validate_capsule(json.loads(path.read_text()), quality, profile)


def materialize_policies(root, capsule):
    """Immutable, run-relative runtime cache; a mismatch fails rather than overwrites."""
    root = Path(root).resolve()
    directory = root / "artifacts" / "quality-policies" / digest(capsule)
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise StateUnavailable("quality policy cache must not traverse symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in POLICY_NAMES:
        path = directory / (name + ".json")
        if path.is_symlink():
            raise StateUnavailable("quality policy cache symlink")
        data = capsule["policies"][name].encode("utf-8")
        try:
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            pass
        if path.read_bytes() != data:
            raise StateUnavailable("quality policy cache changed/incomplete")
        result[name] = path
    return result
