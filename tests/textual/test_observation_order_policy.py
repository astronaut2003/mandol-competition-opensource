from dataclasses import asdict, replace
import json

import pytest

from mandol_competition.textual.core.observation_order import ObservationOrderPolicy, renderer_hash
from mandol_competition.textual.core.public_api_source import SourceTurn, prepare_session
from mandol_competition.textual.api.config import Settings, QUALITY_SHA256
from mandol_competition.textual.runtime import seed_frozen_capsule
from mandol_substrate.state.models import CanonicalMemory, ProfileMismatch, Scope, StateUnavailable


def fixture():
    scope = Scope("textual", "same-user")
    source = prepare_session(scope, "session", [SourceTurn("a", "user", "I started the course."),
                                                SourceTurn("b", "assistant", "I stopped the course.")])
    req = source["requests"][0]
    records = {}
    for i, msg in enumerate(req["messages"]):
        uid = scope.memory_uid(req["session_id"], req["request_id"], i)
        records[uid] = CanonicalMemory(uid, msg["content"], metadata={"aml": True, "user_id": scope.user_id},
                                      session_id=req["session_id"], request_id=req["request_id"],
                                      role=msg["role"], turn_index=i, commit_seq=1)
    hits = [dict(id=uid, content=m.text_content, score=.5) for uid, m in records.items()]
    return scope, source, hits, records


def test_canonical_observation_view_preserves_source_and_hit_order():
    scope, source, hits, records = fixture()
    policy = ObservationOrderPolicy(renderer_hash())
    result, audit = policy.render(scope, list(reversed(hits)), records, query="How did the course develop?",
                                  choices=None, published_seq=1)
    assert len(result) == len(hits)
    assert all(original['content'] in rendered['content']
               for original, rendered in zip(reversed(hits), result, strict=True))
    assert [r["id"] for r in result] == [r["id"] for r in reversed(hits)]
    assert audit["source_text_preserved"] and not audit["temporal_validity_inferred"]
    assert policy.render(scope, hits, records, query="When was the course?", choices=None, published_seq=1)[0] == hits


@pytest.mark.parametrize("change", [
    {"metadata": {"aml": True, "user_id": "other-user"}},
    {"commit_seq": 2}, {"active": False}, {"memory_level": "L1_Abstract"}, {"request_id": "different"},
])
def test_cross_scope_unpublished_or_invalid_source_fails_closed(change):
    scope, _, hits, records = fixture()
    records[hits[0]["id"]] = replace(records[hits[0]["id"]], **change)
    with pytest.raises(StateUnavailable):
        ObservationOrderPolicy(renderer_hash()).render(scope, hits, records, query="Show my course history",
                                                       choices=None, published_seq=1)


def test_source_revision_and_clipped_value_do_not_receive_invented_order():
    scope, _, hits, records = fixture()
    records[hits[0]["id"]] = replace(records[hits[0]["id"]], version=2)
    hits[1]["content"] = hits[1]["content"][:6]
    result, audit = ObservationOrderPolicy(renderer_hash()).render(scope, hits, records,
        query="What happened previously?", choices=None, published_seq=1)
    assert result == hits and not audit["activated"]


def test_candidate_binding_retains_t0_and_rejects_combined_quality_changes(tmp_path):
    root = tmp_path / "state"
    seed_frozen_capsule(root)
    from mandol_competition.textual.core.profile import load_profile
    profile = load_profile(root / "profile.json")
    value = dict(schema="mandol-textual-quality-candidate-v5", profile_id="textual-source-order-v1",
                 baseline_quality_sha256=QUALITY_SHA256, storage_profile_hash=profile.hash,
                 overrides={"returns": {}}, observation_order_policy=asdict(ObservationOrderPolicy(renderer_hash())))
    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text(json.dumps(value))
    def settings(path=None):
        return Settings(competition_run_root=str(root), evaluation_namespace="textual", memory_system_key="test-key",
                        experimental_quality_profile=str(path) if path else None)
    baseline = settings().bind_run()
    candidate = settings(candidate_file).bind_run()
    for field in ["competition_profile", "competition_query_policy", "competition_context_policy",
                  "competition_reranker_execution_policy", "competition_return_policy"]:
        assert open(getattr(candidate, field), "rb").read() == open(getattr(baseline, field), "rb").read()
    assert candidate.quality_candidate["observation_order_policy_hash"] == ObservationOrderPolicy(renderer_hash()).hash
    bad = dict(value, profile_id="textual-illegal-combination", overrides={"returns": {"final_k": 20}})
    candidate_file.write_text(json.dumps(bad))
    with pytest.raises(ProfileMismatch, match="retain every baseline"):
        settings(candidate_file).bind_run()
    candidate_file.write_text(json.dumps(value | {"observation_order_policy": dict(value["observation_order_policy"], renderer_sha256="0" * 64)}))
    with pytest.raises(ProfileMismatch, match="fingerprint"):
        settings(candidate_file).bind_run()
