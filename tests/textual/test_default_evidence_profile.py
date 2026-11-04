import json

import pytest

from mandol_competition.textual.api.config import AML_V4_QUALITY_SHA256, Settings
from mandol_competition.textual.core.evidence_profiles import (
    OBSERVATION_T0, ORIGINAL_T0, observation_profile_path,
)
from mandol_competition.textual.core.observation_order import renderer_hash
from mandol_substrate.state.models import StateUnavailable


def settings(root, **kwargs):
    return Settings(competition_run_root=str(root), evaluation_namespace="textual",
        memory_system_key="fixture-only", inference_contract="aml-official-eligible-v1",
        qdrant_collection="textual_aml_v4_v1", embedding_url="http://127.0.0.1:18120",
        splade_url="http://127.0.0.1:18120", **kwargs)


def test_packaged_default_preserves_raw_storage_and_candidate_selection():
    from mandol_substrate.state.models import digest

    path = observation_profile_path()
    promoted = json.loads(path.read_text(encoding="utf-8"))
    storage = json.loads((path.parent / "profile.json").read_text(encoding="utf-8"))
    # Validate the published recipe directly, without unpublished experiment
    # descriptors or private Git history. Output annotations cannot change hits.
    assert promoted == {
        "profile_id": OBSERVATION_T0,
        "schema": "mandol-textual-quality-candidate-v5",
        "baseline_quality_sha256": AML_V4_QUALITY_SHA256,
        "storage_profile_hash": digest(storage),
        "overrides": {"returns": {}},
        "observation_order_policy": {
            "version": "textual-observation-history-order-v1",
            "renderer_sha256": renderer_hash(),
            "max_added_chars": 2000,
        },
    }


def test_default_is_packaged_versioned_and_restart_idempotent(tmp_path):
    first = settings(tmp_path).bind_run()
    assert first.experimental_quality_profile is None  # No repository opt-in path.
    assert first.selected_evidence_profile == OBSERVATION_T0
    assert first.quality_candidate["status"] == "DEFAULT_USER_APPROVED_PUBLIC_PROXY"
    assert first.quality_candidate["observation_order_policy"]["renderer_sha256"] == renderer_hash()
    assert observation_profile_path().is_file()
    capsule = {n: (tmp_path / n).read_bytes() for n in
               ("profile.json", "quality-profile.json", "quality-policies.json")}
    identities = list((tmp_path / "artifacts/quality-candidate-identities").glob("*.json"))
    repeated = settings(tmp_path).bind_run()
    assert repeated.quality_candidate == first.quality_candidate
    assert list((tmp_path / "artifacts/quality-candidate-identities").glob("*.json")) == identities
    assert all((tmp_path / name).read_bytes() == data for name, data in capsule.items())


def test_explicit_original_t0_rollback_needs_no_storage_change(tmp_path):
    upgraded = settings(tmp_path).bind_run()
    original = settings(tmp_path, evidence_profile=ORIGINAL_T0).bind_run()
    assert original.selected_evidence_profile == ORIGINAL_T0
    assert original.quality_candidate is None
    for field in ("competition_profile", "competition_query_policy", "competition_context_policy",
                  "competition_return_policy", "competition_reranker_execution_policy"):
        assert open(getattr(upgraded, field), "rb").read() == open(getattr(original, field), "rb").read()
    assert settings(tmp_path).bind_run().quality_candidate == upgraded.quality_candidate


def test_explicit_research_candidate_does_not_inherit_promoted_output(tmp_path):
    baseline = settings(tmp_path / "state", evidence_profile=ORIGINAL_T0).bind_run()
    profile = json.loads(open(baseline.competition_profile).read())
    from mandol_substrate.state.models import digest
    candidate = dict(schema="mandol-textual-quality-candidate-v1", profile_id="research-raw-control",
        baseline_quality_sha256=AML_V4_QUALITY_SHA256, storage_profile_hash=digest(profile),
        overrides={"returns": {"final_k": 8}})
    path = tmp_path / "research.json"
    path.write_text(json.dumps(candidate))
    research = settings(tmp_path / "state", experimental_quality_profile=str(path)).bind_run()
    assert research.selected_evidence_profile == "research-raw-control"
    assert "observation_order_policy" not in research.quality_candidate


@pytest.mark.parametrize("experimental", ["experimental_quality_profile", "experimental_hlm_policy"])
def test_promoted_profile_cannot_be_silently_combined_with_experiments(tmp_path, experimental):
    with pytest.raises(StateUnavailable, match="raw-only"):
        settings(tmp_path / "state", evidence_profile=OBSERVATION_T0, **{experimental: "not-read.json"}).bind_run()
    assert not (tmp_path / "state").exists()


def test_legacy_research_family_retains_original_default(tmp_path):
    cfg = Settings(competition_run_root=str(tmp_path), evaluation_namespace="textual", memory_system_key="fixture-only")
    cfg.bind_run()
    assert cfg.selected_evidence_profile == ORIGINAL_T0 and cfg.quality_candidate is None
    cfg.evidence_profile = OBSERVATION_T0
    with pytest.raises(StateUnavailable, match="v4"):
        cfg.bind_run()


@pytest.mark.parametrize("selection", ["unknown-version", "", True])
def test_invalid_evidence_profile_fails_before_creating_storage(tmp_path, selection):
    with pytest.raises(ValueError, match="evidence profile"):
        settings(tmp_path / "state", evidence_profile=selection).bind_run()
    assert not (tmp_path / "state").exists()
