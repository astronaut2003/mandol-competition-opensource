"""Versioned Textual output selection over the unchanged storage capsule."""

from pathlib import Path

from mandol_substrate.state.models import StateUnavailable


ORIGINAL_T0 = "t0-original-v1"
OBSERVATION_T0 = "textual-t0-observation-history-v1"
EVIDENCE_PROFILES = frozenset({"default", ORIGINAL_T0, OBSERVATION_T0})


def validate_evidence_profile(selection):
    if not isinstance(selection, str) or selection not in EVIDENCE_PROFILES:
        raise ValueError("Unknown versioned Textual evidence profile")
    return selection


def resolve_evidence_profile(selection, *, official, experimental=False):
    validate_evidence_profile(selection)
    if selection == OBSERVATION_T0 and (not official or experimental):
        raise StateUnavailable("Promoted Textual evidence requires the v4 raw-only baseline")
    # Explicit research candidates retain their original baseline, rather than
    # silently composing a new default with a previously frozen experiment.
    if selection == "default":
        return OBSERVATION_T0 if official and not experimental else ORIGINAL_T0
    return selection


def observation_profile_path():
    # Installed package resource: works without the repository's configs/ tree.
    return Path(__file__).parents[1] / "profiles/aml_official_v4_v1/observation-history-default-v1.json"
