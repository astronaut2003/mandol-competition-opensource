"""Artificial evidence cases; all mini decisions are explicit offline stubs."""
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mandol_substrate.representation import MemoryUnit
from mandol_competition.inference.evidence_selector import EvidenceSelector, Selection
from mandol_competition.inference.errors import InferenceRateLimited, InferenceSchemaError, InferenceUnavailable
from mandol_competition.multimodal.config import MultimodalConfig, ConfigurationError
from mandol_competition.multimodal.evidence import select_sources, source_text
from mandol_competition.multimodal.protocol import SearchRequest


class Mini:
    def __init__(self, indices=(), error=None, content=None):
        self.indices, self.error, self.content = indices, error, content
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error:
            raise self.error
        return {"model": "gpt-4o-mini", "choices": [{"finish_reason": "stop", "message": {
            "content": self.content if self.content is not None else json.dumps({"indices": self.indices})}}]}

    def close(self):
        pass


def memory(uid, text, *, session="s1", role="user", images=()):
    return MemoryUnit(uid, {"text_content": text, "original_content": text,
        "content_parts": [{"type": "image", "media_id": image} for image in images]},
        {"user_id": "alice", "session_id": session, "role": role, "timestamp": "2026-01-02", "message_index": 0})


def request(top_k=100):
    return SearchRequest(query="Which journal did I buy?", user_id="alice", top_k=top_k,
                         options=["A. field notes", "B. travel notes"])


def test_empty_selection_does_not_pass_topic_distractors_to_reader():
    client = Mini(indices=[])
    selector = EvidenceSelector(None, client=client, strict=True)
    sources = [(memory("a", "A desk and a closed notebook are visible."), .8)]
    assert select_sources(selector, request(), sources, SimpleNamespace(get=lambda _: None)) == []
    payload = client.calls[0][0][1]["content"][0]["text"]
    assert json.loads(payload)["options"] == request().options
    assert sources[0][0].raw_data["text_content"] == "A desk and a closed notebook are visible."


def test_linked_sources_remain_distinct_and_only_existing_sources_return():
    client = Mini(indices=[1, 0])
    selector = EvidenceSelector(None, client=client, strict=True)
    a = memory("a", "I chose a green journal.", session="choice")
    b = memory("b", "I bought the one I selected yesterday.", session="purchase", role="assistant")
    hits = select_sources(selector, request(), [(a, .9), (b, .8)], SimpleNamespace(get=lambda _: None))
    assert [u.uid for u, _ in hits] == ["b", "a"]
    assert [score for _, score in hits] == [1., .5]
    blocks = client.calls[0][0][1]["content"]
    assert "choice" in blocks[1]["text"] and "purchase" in blocks[2]["text"]
    assert '"role": "assistant"' in source_text(b)


@pytest.mark.parametrize("content", ['{"indices":[99]}', '{"indices":[0,0]}', '{"indices":[true]}',
                                     '{"indices":[],"answer":"guessed"}', '{}', 'not json'])
def test_invalid_selector_output_is_visible_failure(content):
    selector = EvidenceSelector(None, client=Mini(content=content), strict=True)
    with pytest.raises(InferenceSchemaError):
        select_sources(selector, request(), [(memory("a", "a"), .8)], SimpleNamespace(get=lambda _: None))


def test_quota_error_remains_429_and_legacy_selector_fallback_is_preserved():
    error = InferenceRateLimited(retry_after=37)
    for strict in (False, True):
        selector = EvidenceSelector(None, client=Mini(error=error), strict=strict)
        if strict:
            with pytest.raises(InferenceRateLimited) as caught:
                selector.select(query="q", options=None, candidates=[{"text": "x"}], limit=1)
            assert caught.value.retry_after == 37
        else:
            assert selector.select(query="q", options=None, candidates=[{"text": "x"}], limit=1).status == "fallback"


def test_shared_image_is_sent_once_to_selector_and_retains_source_binding():
    client = Mini(indices=[0, 1])
    media = SimpleNamespace(get=lambda _: (b"unchanged original image", "image/png"))
    a, b = memory("a", "first", images=["photo"]), memory("b", "second", images=["photo"])
    hits = select_sources(EvidenceSelector(None, client=client, strict=True), request(), [(a, 1.), (b, .9)], media)
    blocks = client.calls[0][0][1]["content"]
    image_blocks = [b for b in blocks if b["type"] == "image_url"]
    assert len(image_blocks) == 1
    assert base64.b64decode(image_blocks[0]["image_url"]["url"].split(",")[1]) == b"unchanged original image"
    assert "image_in_candidate" in blocks[-1]["text"]
    assert [u.uid for u, _ in hits] == ["a", "b"]


def test_effective_search_identity_changes_without_reindexing():
    from mandol_competition.multimodal.models import load_profile
    from mandol_competition.multimodal.search_profiles import identity, BASELINE
    original = MultimodalConfig(inference_contract="aml-official-eligible-v1", model_device="external-http")
    candidate = replace(original, vlm_backend="openai_compat", evidence_policy="source-grounded-v1",
                        vision_failure_policy="retry-required-v1")
    assert load_profile(original).hash == load_profile(candidate).hash
    assert identity(load_profile(original), BASELINE) != identity(load_profile(candidate), BASELINE, candidate.evidence_policy)


def test_invalid_or_fake_evidence_policy_is_rejected():
    with pytest.raises(ConfigurationError):
        MultimodalConfig(evidence_policy="source-grounded-v1")
    with pytest.raises(ConfigurationError):
        MultimodalConfig(vision_failure_policy="silently-ignore")


def test_selector_failure_never_becomes_empty_success():
    selector = SimpleNamespace(select=lambda **_: Selection((0,), "fallback", 0))
    with pytest.raises(InferenceUnavailable):
        select_sources(selector, request(), [(memory("a", "a"), .8)], SimpleNamespace(get=lambda _: None))


@pytest.mark.parametrize("policy", ["source-grounded-v1", "source-grounded-atomic-v2", "source-grounded-delivery-v3"])
def test_modern_factory_reads_only_mm_activation_flags(tmp_path, monkeypatch, policy):
    from mandol_competition.runtime import apps
    settings = SimpleNamespace(runtime=SimpleNamespace(mode="modern", limits={}, vlm_credential_env="FIXTURE_KEY"),
        root=tmp_path, namespace="multimodal", memory_system_key="fixture", allow_unauthenticated_dev=False,
        inference={"embedding": "http://unused", "sparse": "http://unused", "reranker": "http://unused",
                   "vlm": "http://mini.invalid/v1"}, qdrant_endpoint="http://unused", collection="mm",
        qdrant_api_key=None, inference_contract="aml-official-eligible-v1", retrieval_policy="baseline",
        reranker_pool={}, reranker_credential_env="FIXTURE_KEY")
    monkeypatch.setattr(apps, "_settings", lambda _: settings)
    monkeypatch.setattr("mandol_competition.multimodal.api.create_app", lambda config: config)
    monkeypatch.setenv("MANDOL_MM_EVIDENCE_POLICY", policy)
    monkeypatch.setenv("MANDOL_MM_VISION_FAILURE_POLICY", "retry-required-v1")
    config = apps.multimodal()
    assert config.evidence_policy == policy and config.vision_failure_policy == "retry-required-v1"
    assert config.vlm_request_model == "gpt-4o-mini" and config.vlm_api_key_env == "FIXTURE_KEY"


def test_long_unicode_evidence_is_bounded_only_in_selector_view():
    from mandol_competition.multimodal.evidence import SELECTOR_TEXT_BYTES
    text = "人工证据\x00" * 1000
    sources = [(memory(str(i), text), 1.) for i in range(64)]
    mini = Mini(indices=[0])
    chosen = select_sources(EvidenceSelector(None, client=mini, strict=True), request(), sources,
                            SimpleNamespace(get=lambda _: None))
    blocks = mini.calls[0][0][1]["content"][1:]
    previews = [json.loads(block["text"])["text"] for block in blocks]
    assert sum(len(json.dumps(t, ensure_ascii=False).encode('utf-8')) for t in previews) <= SELECTOR_TEXT_BYTES
    assert all("Selector preview truncated" in t for t in previews)
    assert chosen[0][0].raw_data["text_content"] == text
