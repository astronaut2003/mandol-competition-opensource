import json
from dataclasses import replace

from mandol_competition.textual.core.event_state_memory import EventStateRecipe
from mandol_competition.textual.core.heterogeneous_worker import typed_scope
from mandol_competition.textual.core.hlm_builder import HighLevelMemoryBuilder
from mandol_competition.textual.core.hlm_production import HighLevelMemoryPolicy, ProductionHLMRecipe
from mandol_substrate.state import Mutation
from tests.common.test_publication import coordinator


def test_plans_and_cancellations_are_distinct_states_with_raw_provenance(store, profile, scope, memory, tmp_path):
    coordinator_, _ = coordinator(store, profile, tmp_path)
    source = replace(memory, text_content="I plan to visit Hangzhou next month. I cancelled the museum visit.",
        role="user", metadata={"aml": True, "user_id": scope.user_id}, turn_index=0)
    coordinator_.apply(scope, "raw", "body", Mutation(memories=(source,)), {})
    calls = []

    def provider(messages):
        calls.append(messages)
        base = {"assertion_reason": "Explicit self-report of an intention or cancellation, not a completed trip.",
            "event_type": "ACTIVITY", "actor": "CURRENT_USER", "object": None, "participants": [],
            "location": None, "state_before": None, "state_after": None, "context_source_uids": [], "confidence": 1.0}
        plan = {**base, "assertion_kind": "EVENT_PLAN", "status": "PLAN", "action": "visit Hangzhou",
            "text": "The user plans to visit Hangzhou next month.", "time_precision": "RELATIVE", "event_time": "next month",
            "evidence": [{"source_uid": source.logical_uid, "quote": "I plan to visit Hangzhou next month."}]}
        cancel = {**base, "assertion_kind": "EVENT_CANCELLATION", "status": "CANCELLED", "action": "cancel museum visit",
            "text": "The user cancelled the museum visit.", "time_precision": "UNKNOWN", "event_time": None,
            "evidence": [{"source_uid": source.logical_uid, "quote": "I cancelled the museum visit."}]}
        return {"status": "PASS", "response_model": "gpt-4o-mini-2024-07-18", "finish_reason": "stop",
                "content": json.dumps({"episodes": [plan, cancel]}), "usage": {}, "latency_ms": 0}

    policy = HighLevelMemoryPolicy(episodic_enabled=True)
    baseline = HighLevelMemoryBuilder.from_competition(coordinator_, provider, policy=policy)
    try:
        assert baseline.build_session(scope, [source.logical_uid])[0]["output_uids"]["episode"] == []
    finally:
        baseline.close()
    candidate = HighLevelMemoryBuilder.from_competition(coordinator_, provider,
        recipe=EventStateRecipe(profile.hash, policy=policy))
    try:
        result = candidate.build_session(scope, [source.logical_uid])
        uids = result[0]["output_uids"]["episode"]
        assert len(uids) == 2
        partition = typed_scope(scope, "episode", candidate.canonical.recipe)
        derived = store.get_memories(partition, uids)
        assert {m.metadata["hlm_item"]["status"] for m in derived.values()} == {"PLAN", "CANCELLED"}
        assert all("Source assertion:" in m.text_content for m in derived.values())
        assert set(candidate.canonical.eligible(scope, derived)) == set(uids)
        assert candidate.build_session(scope, [source.logical_uid]) == result
        assert len(calls) == 2  # One per distinct recipe; replay is durable.
        assert store.get_memories(scope, [source.logical_uid])[source.logical_uid].text_content == source.text_content
        coordinator_.apply(scope, "correction", "corrected-body",
            Mutation(memories=(replace(source, version=2, text_content="The trip is no longer planned."),)), {})
        assert candidate.canonical.eligible(scope, derived) == {}
    finally:
        candidate.close()


def test_default_fingerprint_and_occurrence_semantics_remain_frozen():
    from pathlib import Path
    value = json.loads((Path(__file__).resolve().parents[2] /
        "configs/profiles/textual/profile-episode-binding-v2.json").read_text())
    policy = HighLevelMemoryPolicy(**value["policy"])
    baseline = ProductionHLMRecipe(value["base_storage_profile_hash"], policy=policy)
    candidate = EventStateRecipe(value["base_storage_profile_hash"], policy=policy)
    assert baseline.fingerprint == value["recipe_fingerprint"]
    assert ("EVENT_PLAN", "PLAN") not in baseline.admissible_episode_states
    assert ("EVENT_PLAN", "PLAN") in candidate.admissible_episode_states
    assert ("EVENT_PLAN", "COMPLETED") not in candidate.admissible_episode_states
    assert candidate.fingerprint != baseline.fingerprint
