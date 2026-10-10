import json

import pytest

from mandol_competition.textual.api.config import Settings, QUALITY_SHA256
from mandol_competition.textual.core.context_policy import ContextPolicy
from mandol_competition.textual.core.return_policy import ReturnPolicy
from mandol_competition.textual.runtime import seed_frozen_capsule
from mandol_substrate.state.models import ProfileMismatch, StateUnavailable


def _config(tmp_path):
    root = tmp_path / "state"
    seed_frozen_capsule(root)
    from mandol_competition.textual.core.profile import load_profile
    profile = load_profile(root / "profile.json")
    value = {"schema": "mandol-textual-quality-candidate-v1", "profile_id": "textual-output-100-candidate-v1",
             "baseline_quality_sha256": QUALITY_SHA256, "storage_profile_hash": profile.hash,
             "overrides": {"context": {"selected_k": 100}, "returns": {"final_k": 100, "max_context_chars": 100000}}}
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    return root, path, value


def _settings(root, path=None):
    return Settings(competition_run_root=str(root), evaluation_namespace="textual", memory_system_key="test-key",
                    experimental_quality_profile=str(path) if path else None)


def test_opt_in_changes_only_query_output_and_baseline_remains_byte_identical(tmp_path):
    root, path, value = _config(tmp_path)
    original = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    baseline = _settings(root).bind_run()
    candidate = _settings(root, path).bind_run()
    assert ReturnPolicy.load(baseline.competition_return_policy).final_k == 10
    assert ContextPolicy.load(candidate.competition_context_policy).selected_k == 100
    assert ReturnPolicy.load(candidate.competition_return_policy).final_k == 100
    assert candidate.competition_profile == baseline.competition_profile or (
        open(candidate.competition_profile, 'rb').read() == open(baseline.competition_profile, 'rb').read())
    assert candidate.quality_candidate["id"] == value["profile_id"]
    assert candidate.bind_run() is candidate
    assert all((root / name).read_bytes() == content for name, content in original.items())
    # Turning the experiment off rebinds a new service to the actual baseline.
    assert ReturnPolicy.load(_settings(root).bind_run().competition_return_policy).final_k == 10


def test_wrong_physical_recipe_unauthorized_override_and_reused_candidate_id_fail_closed(tmp_path):
    root, path, value = _config(tmp_path)
    _settings(root, path).bind_run()
    changed = {**value, "overrides": {"returns": {"final_k": 50}}}
    path.write_text(json.dumps(changed))
    with pytest.raises(ProfileMismatch, match="Immutable"):
        _settings(root, path).bind_run()
    for update in ({"storage_profile_hash": "0" * 64}, {"baseline_quality_sha256": "0" * 64},
                   {"overrides": {"profile": {"embedding_model": "another"}}},
                   {"overrides": {"context": {"renderer_sha256": "0" * 64}}},
                   {"overrides": {"returns": {"final_k": 101}}}):
        path.write_text(json.dumps({**value, **update}))
        with pytest.raises((ProfileMismatch, ValueError, TypeError)):
            _settings(root, path).bind_run()


def test_default_still_rejects_arbitrary_runtime_override(tmp_path):
    root, path, _ = _config(tmp_path)
    settings = _settings(root)
    settings.competition_return_policy = str(path)
    with pytest.raises(StateUnavailable, match="frozen capsule"):
        settings.bind_run()


def test_query_input_v2_is_explicit_and_does_not_change_physical_or_model_recipes(tmp_path):
    root, path, value = _config(tmp_path)
    value.update(schema="mandol-textual-quality-candidate-v2",
        query_input_policy={"version": "textual-all-choice-hints-v1", "max_rendered_chars": 16000})
    path.write_text(json.dumps(value))
    baseline = _settings(root).bind_run()
    candidate = _settings(root, path).bind_run()
    assert candidate.quality_candidate["query_input_policy"] == value["query_input_policy"]
    assert open(candidate.competition_profile, 'rb').read() == open(baseline.competition_profile, 'rb').read()
    assert len(candidate.quality_candidate["query_input_policy_hash"]) == 64
    value["query_input_policy"]["version"] = "undeclared"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="query-input"):
        _settings(root, path).bind_run()


def test_evidence_packaging_v3_does_not_modify_add_recipe(tmp_path):
    root, path, value = _config(tmp_path)
    value.update(schema="mandol-textual-quality-candidate-v3",
        evidence_packaging_policy={"version": "textual-provenance-families-v1", "selected_k": 100,
                                   "max_context_chars": 60000})
    path.write_text(json.dumps(value))
    candidate = _settings(root, path).bind_run()
    assert candidate.quality_candidate["evidence_packaging_policy"] == value["evidence_packaging_policy"]
    assert len(candidate.quality_candidate["evidence_packaging_policy_hash"]) == 64
    assert candidate.experimental_hlm_policy is None
    value["evidence_packaging_policy"]["selected_k"] = 101
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="evidence-packaging"):
        _settings(root, path).bind_run()
