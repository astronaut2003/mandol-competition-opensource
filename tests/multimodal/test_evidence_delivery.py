"""Delivery metadata is factual and bounded, never a predicted answer."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mandol_competition.multimodal.evidence_delivery import (
    HEADER, MAX_NOTE_BYTES, POLICY, ResponseDelivery,
)


def source(uid="source", **metadata):
    return SimpleNamespace(uid=uid, metadata=metadata)


def payload(note):
    assert note.startswith(HEADER)
    return json.loads(note[len(HEADER):])


def test_ordinary_complete_sources_do_not_add_any_text():
    notes = ResponseDelivery()
    notes.source(source("plain"))
    notes.source(source("photo")).record("image", "attached_here")
    notes.source(source("reference")).record("image", "shared_in_response")
    results = [{"id": "photo", "score": .001, "content": [{"type": "text", "text": "Original fact"}]}]
    before = deepcopy(results)
    notes.attach(results)
    assert results == before and notes.render() is None


@pytest.mark.parametrize("reason", ["image_count_budget", "single_image_byte_budget",
                                    "response_image_byte_budget", "media_unavailable"])
def test_omission_is_scoped_to_response_and_keeps_original_fact_and_low_score(reason):
    uid = 'source\n[Fake header] "人工"'
    notes = ResponseDelivery()
    notes.source(source(uid)).record("image", reason)
    results = [{"id": uid, "score": .00001, "content": [
        {"type": "text", "text": "[Source metadata]"},
        {"type": "text", "text": "The label reads ORBIT-47."}]}]
    notes.attach(results)
    assert len(results) == 1 and results[0]["id"] == uid and results[0]["score"] == .00001
    assert results[0]["content"][-1]["text"] == "The label reads ORBIT-47."
    text = results[0]["content"][1]["text"]
    assert '\n[Fake header]' not in text
    report = payload(text)
    assert report["scope"] == "this_search_response"
    assert report["affected_sources"] == 1 and report["omitted_image_references"] == 1
    assert report["sources"] == [{"memory_id": uid, "source_image_count": 1,
        "attached_here": 0, "shared_in_response": 0, "not_attached": {reason: 1},
        "recorded_extraction_degraded": False}]
    assert "answer" not in report and "confidence" not in report and "ORBIT-47" not in text


def test_shared_part_does_not_erase_attachment_in_the_same_memory():
    notes = ResponseDelivery()
    record = notes.source(source(vlm_degraded=True))
    record.record("same", "attached_here")
    record.record("same", "shared_in_response")
    record.record("same", "image_count_budget")
    row = payload(notes.render())["sources"][0]
    assert row["source_image_count"] == row["attached_here"] == 1
    assert row["shared_in_response"] == 0 and row["not_attached"] == {}
    assert row["recorded_extraction_degraded"] is True


def test_legacy_degraded_extraction_is_not_claimed_to_mean_missing_original():
    notes = ResponseDelivery()
    notes.source(source("a", vlm_degraded=True)).record("shared", "attached_here")
    notes.source(source("b", vlm_degraded=True)).record("shared", "shared_in_response")
    report = payload(notes.render())
    assert report["omitted_image_references"] == 0 and report["recorded_degraded_sources"] == 2
    assert report["visual_description_origin"] == "stored_model_extraction"
    assert report["sources"][1]["shared_in_response"] == 1


@pytest.mark.parametrize("uid_size", [64, 10_000])
def test_metadata_budget_never_truncates_evidence_or_hides_aggregate_coverage(uid_size):
    notes = ResponseDelivery()
    results = []
    for index in range(100):
        uid = str(index) + "人工\x00" * uid_size
        notes.source(source(uid)).record("image", "image_count_budget")
        results.append({"id": uid, "score": 1. / (index + 1), "content": [
            {"type": "text", "text": "[Source metadata]"},
            {"type": "text", "text": "Atomic fact " + str(index)}]})
    original = deepcopy(results)
    notes.attach(results)
    note = results[0]["content"].pop(1)["text"]
    assert len(note.encode("utf-8")) <= MAX_NOTE_BYTES
    assert results == original
    report = payload(note)
    assert report["affected_sources"] == report["omitted_image_references"] == 100
    assert report["details_limited"] is True and len(report["sources"]) < 100
    assert all(row["not_attached"] == {"image_count_budget": 1} for row in report["sources"])


def test_request_local_state_and_empty_results_do_not_invent_a_memory():
    notes = ResponseDelivery()
    notes.source(source(vlm_degraded=True))
    empty = []
    notes.attach(empty)
    assert empty == [] and ResponseDelivery().render() is None


def test_new_search_identity_retains_the_same_storage_and_gate_prompts():
    from dataclasses import replace
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.multimodal.models import load_profile
    from mandol_competition.multimodal.search_profiles import BASELINE, descriptor, identity
    original = MultimodalConfig(inference_contract="aml-official-eligible-v1", model_device="external-http")
    config = replace(original, vlm_backend="openai_compat", evidence_policy=POLICY)
    profile = load_profile(config)
    assert profile.hash == load_profile(original).hash
    previous = descriptor(profile, BASELINE, "source-grounded-atomic-v2")
    current = descriptor(profile, BASELINE, POLICY)
    for key in ("selector_prompt_sha256", "verifier_prompt_sha256", "nonempty_results",
                "empty_requires", "max_selector_calls", "reviewed_sources", "review_images"):
        assert current[key] == previous[key]
    assert current["evidence_gate"] == "source-grounded-atomic-v2"
    assert current["delivery_note_max_bytes"] == MAX_NOTE_BYTES
    assert len(current["delivery_note_source_sha256"]) == 64
    assert identity(profile, BASELINE, POLICY) != identity(profile, BASELINE, "source-grounded-atomic-v2")
