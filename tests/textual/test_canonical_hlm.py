import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mandol_competition.textual.core.hlm_builder import HighLevelMemoryBuilder, CanonicalPublicationWriter
from mandol_competition.textual.core.hlm_production import HighLevelMemoryPolicy
from mandol_competition.textual.core.heterogeneous_worker import typed_scope
from mandol_substrate.state import Mutation, StateUnavailable
from tests.common.test_publication import coordinator


def _source(memory, scope):
    return replace(memory, text_content="For this project, review changes before releasing.", role="user", turn_index=0,
                   metadata={"aml": True, "user_id": scope.user_id})


def _provider(calls):
    def complete(messages):
        payload = json.loads(messages[-1]["content"])
        calls.append(payload["stage"])
        assert "untrusted" in messages[0]["content"]
        source = payload["sources"][0]
        return {"status": "PASS", "response_model": "gpt-4o-mini-2024-07-18", "finish_reason": "stop",
                "content": json.dumps({"summaries": [{"text": source["text"], "evidence": [{
                    "source_uid": source["source_uid"], "quote": source["text"]}],
                    "context_source_uids": [], "confidence": 1.0}]}), "usage": {}, "latency_ms": 0}
    return complete


def test_canonical_only_facade_replay_and_stale_source_guard(store, profile, scope, memory, tmp_path):
    c, _ = coordinator(store, profile, tmp_path)
    raw = _source(memory, scope)
    c.apply(scope, "raw", "raw", Mutation(memories=(raw,)), {})
    calls = []
    facade = HighLevelMemoryBuilder.from_competition(c, _provider(calls),
                policy=HighLevelMemoryPolicy(hierarchical_enabled=True))
    try:
        first = facade.build_session(scope, [raw.logical_uid])
        assert first == facade.build_session(scope, [raw.logical_uid])
        assert calls == ["hierarchical"]
        runtime = facade.canonical
        uids = first[0]["output_uids"]["hierarchical"]
        assert len(uids) == 1
        derived = store.get_memories(typed_scope(scope, "hierarchical", runtime.recipe), uids)
        assert set(runtime.eligible(scope, derived)) == set(uids)
        assert store.lexical.inspect_stats(scope)["n_docs"] == 1  # Typed index is independent.
        assert runtime.check_consistency(scope)["status"] == "PASS"
        c.apply(scope, "correct", "correct", Mutation(memories=(replace(raw, version=2, text_content="Corrected rule"),)), {})
        assert runtime.eligible(scope, derived) == {}
    finally:
        facade.close()


def test_durable_preparation_survives_publication_fault_without_second_llm(store, profile, scope, memory, tmp_path):
    c, projection = coordinator(store, profile, tmp_path)
    raw = _source(memory, scope)
    c.apply(scope, "raw", "raw", Mutation(memories=(raw,)), {})
    calls, fired = [], []

    def fault(stage):
        if stage == "after_durable_hierarchical" and not fired:
            fired.append(stage)
            raise RuntimeError("process interruption")

    policy = HighLevelMemoryPolicy(hierarchical_enabled=True)
    f = HighLevelMemoryBuilder.from_competition(c, _provider(calls), policy=policy, fault=fault)
    try:
        with pytest.raises(RuntimeError):
            f.build_session(scope, [raw.logical_uid])
    finally:
        f.close()
    resumed = HighLevelMemoryBuilder.from_competition(c, _provider(calls), policy=policy)
    try:
        assert resumed.build_session(scope, [raw.logical_uid])[0]["output_uids"]["hierarchical"]
        assert calls == ["hierarchical"]
    finally:
        resumed.close()


def test_disabled_hlm_does_not_call_provider_or_import_historical_models(store, profile, scope, memory, tmp_path):
    c, _ = coordinator(store, profile, tmp_path)
    calls = []
    facade = HighLevelMemoryBuilder.from_competition(c, _provider(calls))
    try:
        assert facade.build_session(scope, [memory.logical_uid]) == []
        assert not calls
    finally:
        facade.close()


def test_publication_failure_reaches_caller_and_close_drains():
    def apply(*args):
        if args == ("failure",):
            raise StateUnavailable("T1/T2 failure")
        return args
    writer = CanonicalPublicationWriter(SimpleNamespace(apply=apply))
    success = writer.submit_publication("success")
    failure = writer.submit_publication("failure")
    writer.close()
    assert success.result() == ("success",)
    with pytest.raises(StateUnavailable):
        failure.result()
    with pytest.raises(RuntimeError):
        writer.submit_publication("after close")
