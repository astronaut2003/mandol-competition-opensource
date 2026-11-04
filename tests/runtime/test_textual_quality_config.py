from pathlib import Path

import pytest

from mandol_competition.config import load_config
from mandol_competition.runtime.__main__ import child_environment, describe, profile_path


def test_local_quality_opt_in_ignores_ambient_legacy_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("MANDOL_TEXTUAL_QUALITY_CANDIDATE", "/production/must-not-be-read.json")
    baseline = load_config(profile_path("local"))
    assert baseline.track("textual").quality_candidate is None
    candidate = tmp_path / "candidate.json"
    monkeypatch.setenv("MANDOL_LOCAL_TEXTUAL_QUALITY_CANDIDATE", str(candidate))
    changed = load_config(profile_path("local"))
    assert changed.track("textual").quality_candidate == str(candidate)
    assert changed.track("code") == baseline.track("code")
    assert changed.track("multimodal") == baseline.track("multimodal")
    assert describe(changed)["tracks"]["textual"]["quality_candidate"] == str(candidate)
    env = child_environment(changed, profile_path("local"))
    assert env["MANDOL_LOCAL_TEXTUAL_QUALITY_CANDIDATE"] == str(candidate)
    assert "MANDOL_TEXTUAL_QUALITY_CANDIDATE" not in env


def test_toml_candidate_path_is_relative_to_configuration_file(tmp_path):
    text = profile_path("local").read_text()
    if "[textual]\n" in text:
        text = text.replace("[textual]\n", '[textual]\nquality_candidate = "candidate.json"\n')
    else:
        text += '\n[textual]\nquality_candidate = "candidate.json"\n'
    config = tmp_path / "deployment.toml"
    config.write_text(text)
    assert load_config(config).track("textual").quality_candidate == str(tmp_path / "candidate.json")


@pytest.mark.parametrize("track", ["code", "multimodal"])
def test_other_tracks_cannot_enable_the_textual_candidate(monkeypatch, track):
    monkeypatch.setenv("MANDOL_LOCAL_" + track.upper() + "_QUALITY_CANDIDATE", "/not-a-profile.json")
    with pytest.raises(ValueError, match="only for Textual"):
        load_config(profile_path("local"))


def test_promoted_default_and_mode_scoped_original_rollback(monkeypatch):
    from mandol_competition.textual.core.evidence_profiles import OBSERVATION_T0, ORIGINAL_T0
    monkeypatch.setenv("MANDOL_TEXTUAL_EVIDENCE_PROFILE", "invalid-ambient-production-setting")
    baseline = load_config(profile_path("local"))
    assert describe(baseline)["tracks"]["textual"]["evidence_profile"] == OBSERVATION_T0
    monkeypatch.setenv("MANDOL_LOCAL_TEXTUAL_EVIDENCE_PROFILE", ORIGINAL_T0)
    rollback = load_config(profile_path("local"))
    assert describe(rollback)["tracks"]["textual"]["evidence_profile"] == ORIGINAL_T0
    assert rollback.track("code") == baseline.track("code")
    assert rollback.track("multimodal") == baseline.track("multimodal")
    env = child_environment(rollback, profile_path("local"))
    assert env["MANDOL_LOCAL_TEXTUAL_EVIDENCE_PROFILE"] == ORIGINAL_T0
    assert "MANDOL_TEXTUAL_EVIDENCE_PROFILE" not in env


def test_toml_evidence_selection_and_invalid_version(tmp_path, monkeypatch):
    from mandol_competition.textual.core.evidence_profiles import ORIGINAL_T0
    text = profile_path("local").read_text().replace("[textual]\n", '[textual]\nevidence_profile = "t0-original-v1"\n')
    config = tmp_path / "local.toml"
    config.write_text(text)
    assert load_config(config).track("textual").evidence_profile == ORIGINAL_T0
    monkeypatch.setenv("MANDOL_LOCAL_TEXTUAL_EVIDENCE_PROFILE", "unknown-version")
    with pytest.raises(ValueError, match="evidence profile"):
        load_config(config)


@pytest.mark.parametrize("track", ["code", "multimodal"])
def test_evidence_selection_is_textual_only(monkeypatch, track):
    monkeypatch.setenv("MANDOL_LOCAL_" + track.upper() + "_EVIDENCE_PROFILE", "t0-original-v1")
    with pytest.raises(ValueError, match="only for Textual"):
        load_config(profile_path("local"))
