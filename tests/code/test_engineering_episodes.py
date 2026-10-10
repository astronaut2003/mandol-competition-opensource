import json

import pytest

from mandol_substrate.representation import MemoryUnit
from mandol_competition.code.software import (
    EngineeringAttempt, EngineeringEpisodeBuilder, EngineeringEpisodeMemory,
    EvidenceClaim, ExperienceOutcome, RepairMemorySchema, ValidationEvidence, VerificationLevel,
)


def source(uid="m1", text="  TypeError: exact diagnostic\n", user="u", role="user"):
    return MemoryUnit(uid=uid, raw_data={"text_content": text},
                      metadata={"user_id": user, "session_id": "s", "role": role})


def test_episode_preserves_raw_and_stable_identity_across_add_batches():
    a = source()
    builder = EngineeringEpisodeBuilder()
    first = builder.build([a], user_id="u", session_id="s")
    second = builder.build([a, source("m2", "Tests passed", role="assistant")], user_id="u", session_id="s")
    assert first.episode_id == second.episode_id
    assert second.problem == a.raw_data["text_content"] == "  TypeError: exact diagnostic\n"
    assert second.outcome == ExperienceOutcome.UNKNOWN
    assert second.verification_level == VerificationLevel.OBSERVED
    assert second.source_uids == ["m1", "m2"]


def test_cross_user_and_changed_source_are_rejected():
    builder = EngineeringEpisodeBuilder()
    with pytest.raises(ValueError, match="cross-user"):
        builder.build([source(user="other")], user_id="u", session_id="s")
    a = source()
    episode = builder.build([a], user_id="u", session_id="s")
    a.raw_data["text_content"] = "overwritten"
    with pytest.raises(ValueError, match="changed"):
        episode.validate_provenance({a.uid: a})


def test_error_name_mentions_are_not_certified_observed_symptoms():
    text = "The documentation explains how to catch AttributeError. No error occurred."
    for role in ("user", "assistant", "tool"):
        raw = source(text=text, role=role)
        episode = EngineeringEpisodeBuilder().build([raw], user_id="u", session_id="s")
        assert episode.symptoms
        assert all(claim.verification == VerificationLevel.INFERRED for claim in episode.symptoms)
        assert all(claim.text in text for claim in episode.symptoms)
        assert episode.outcome == ExperienceOutcome.UNKNOWN
        assert raw.raw_data["text_content"] == text
        episode.validate_provenance({raw.uid: raw})


def test_verified_requires_final_matching_execution_evidence():
    a = source(text="All tests passed, verified!")
    episode = EngineeringEpisodeBuilder().build([a], user_id="u", session_id="s")
    episode.verification_level = VerificationLevel.VERIFIED
    with pytest.raises(ValueError, match="execution"):
        episode.validate_provenance({a.uid: a})
    episode.validation = [ValidationEvidence("pytest", a.uid, 0, "passed", True)]
    with pytest.raises(ValueError, match="execution"):
        episode.validate_provenance({a.uid: a})
    a.raw_data["execution_result"] = {"command": "pytest", "exit_code": 0, "result": "passed", "final": True}
    episode = EngineeringEpisodeBuilder().build([a], user_id="u", session_id="s")
    episode.validation = [ValidationEvidence("pytest", a.uid, 0, "passed", True)]
    episode.verification_level = VerificationLevel.VERIFIED
    episode.validate_provenance({a.uid: a})


def test_failed_attempt_roundtrips_with_provenance():
    a = source(text="Increasing timeout failed; cancellation contract still broken.")
    episode = EngineeringEpisodeBuilder().build([a], user_id="u", session_id="s")
    episode.attempts = [EngineeringAttempt(EvidenceClaim("Increase timeout", (a.uid,)),
                                           outcome=ExperienceOutcome.FAILURE,
                                           negative_evidence=(a.uid,),
                                           failure_reason=EvidenceClaim(a.raw_data["text_content"], (a.uid,), VerificationLevel.OBSERVED))]
    episode.validate_provenance({a.uid: a})
    serialized = json.loads(json.dumps(episode.to_memory_unit().raw_data))
    restored = EngineeringEpisodeMemory.from_memory_unit(MemoryUnit(uid=episode.episode_id, raw_data=serialized))
    assert restored.attempts[0].outcome == ExperienceOutcome.FAILURE
    assert restored.attempts[0].negative_evidence == (a.uid,)
    restored.validate_provenance({a.uid: a})
    assert "cancellation contract" in restored.retrieval_text()


def test_repair_remains_backward_compatible_and_keeps_negative_evidence():
    old = RepairMemorySchema(symptom="failure").to_memory_unit()
    assert old.raw_data["outcome"] == "unknown"
    repair = RepairMemorySchema(symptom="failure", negative_evidence=["raw:1"],
                                failure_reason="timeout increases do not fix cancellation").to_memory_unit()
    assert repair.raw_data["negative_evidence"] == ["raw:1"]
    assert "cancellation" in repair.raw_data["text_content"]


def test_later_failed_validation_supersedes_earlier_success_and_anchors_are_checked():
    a, b = source("a", "first validation"), source("b", "TypeError: regression")
    a.raw_data["execution_result"] = {"command": "pytest", "exit_code": 0, "result": "passed", "final": True}
    b.raw_data["execution_result"] = {"command": "pytest", "exit_code": 1, "result": "failed", "final": True}
    episode = EngineeringEpisodeBuilder().build([a, b], user_id="u", session_id="s")
    assert episode.outcome == ExperienceOutcome.FAILURE
    assert episode.verification_level == VerificationLevel.OBSERVED
    episode.verification_level = VerificationLevel.VERIFIED
    with pytest.raises(ValueError, match="final execution"):
        episode.validate_provenance({"a": a, "b": b})
    episode.verification_level = VerificationLevel.OBSERVED
    episode.code_anchors[0]["raw"] = "fabricated"
    with pytest.raises(ValueError, match="exact source token"):
        episode.validate_provenance({"a": a, "b": b})


def test_reported_outcome_remains_observed_and_supersedes_old_validation():
    executed, reported = source("execution", "pytest passed"), source("report", "External evaluator reports failure")
    executed.raw_data["execution_result"] = {"command": "pytest", "exit_code": 0, "result": "passed", "final": True}
    reported.raw_data["outcome_observation"] = {"outcome": "failure", "producer": "public-dataset"}
    episode = EngineeringEpisodeBuilder().build([executed, reported], user_id="u", session_id="s")
    assert episode.outcome == ExperienceOutcome.FAILURE
    assert episode.outcome_evidence == [reported.uid]
    assert episode.verification_level == VerificationLevel.OBSERVED
    episode.verification_level = VerificationLevel.VERIFIED
    with pytest.raises(ValueError, match="superseding"):
        episode.validate_provenance({unit.uid: unit for unit in (executed, reported)})


def test_v1_verified_episode_migrates_without_changing_raw_or_identity():
    raw = source(text="pytest passed")
    raw.raw_data["execution_result"] = {"command": "pytest", "exit_code": 0, "result": "passed", "final": True}
    original = EngineeringEpisodeBuilder().build([raw], user_id="u", session_id="s")
    persisted = json.loads(json.dumps(original.to_memory_unit().raw_data))
    persisted["schema_version"] = 1
    del persisted["outcome_evidence"]
    restored = EngineeringEpisodeMemory.from_memory_unit(MemoryUnit(uid=original.episode_id, raw_data=persisted))
    restored.validate_provenance({raw.uid: raw})
    assert restored.schema_version == 2
    assert restored.episode_id == original.episode_id
    assert restored.source_hashes == original.source_hashes
    assert restored.outcome_evidence == [raw.uid]


def test_stale_persisted_success_cannot_hide_later_raw_failure():
    old, later = source("old", "pytest passed"), source("later", "Final assessment failed")
    old.raw_data["execution_result"] = {"command": "pytest", "exit_code": 0, "result": "passed", "final": True}
    later.raw_data["outcome_observation"] = {"outcome": "failure", "producer": "public-assessment"}
    episode = EngineeringEpisodeBuilder().build([old, later], user_id="u", session_id="s")
    episode.outcome, episode.outcome_evidence = ExperienceOutcome.SUCCESS, [old.uid]
    episode.verification_level = VerificationLevel.VERIFIED
    restored = EngineeringEpisodeMemory.from_memory_unit(episode.to_memory_unit())
    with pytest.raises(ValueError, match="superseding raw"):
        restored.validate_provenance({old.uid: old, later.uid: later})
    # Omitting a later failed validation from a derived list cannot erase it.
    later.raw_data.pop("outcome_observation")
    later.raw_data["execution_result"] = {"command": "pytest", "exit_code": 1, "result": "failed", "final": True}
    episode = EngineeringEpisodeBuilder().build([old, later], user_id="u", session_id="s")
    episode.validation = episode.validation[:1]
    episode.outcome, episode.outcome_evidence = ExperienceOutcome.SUCCESS, [old.uid]
    episode.verification_level = VerificationLevel.VERIFIED
    with pytest.raises(ValueError, match="superseding raw"):
        episode.validate_provenance({old.uid: old, later.uid: later})


def test_boolean_exit_status_cannot_be_forged_into_verified_execution():
    raw = source(text="passed")
    raw.raw_data["execution_result"] = {"command": "pytest", "exit_code": False, "result": "passed", "final": True}
    episode = EngineeringEpisodeBuilder().build([raw], user_id="u", session_id="s")
    assert episode.outcome == ExperienceOutcome.UNKNOWN and not episode.validation
    episode.validation = [ValidationEvidence("pytest", raw.uid, 0, "passed", True)]
    episode.verification_level = VerificationLevel.VERIFIED
    with pytest.raises(ValueError, match="raw execution"):
        episode.validate_provenance({raw.uid: raw})
