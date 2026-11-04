"""Recall-preservation fault cases; no real models or official samples."""
from types import SimpleNamespace

import pytest

from mandol_competition.inference.evidence_selector import Selection
from mandol_competition.inference.errors import InferenceRateLimited, InferenceSchemaError
from mandol_competition.multimodal.atomic_guard import guard_sources, prepare_review
from mandol_competition.multimodal.protocol import SearchRequest
from .test_grounded_evidence import memory


class Guard:
    def __init__(self, primary=(), verification=(), error=None):
        self.primary, self.verification, self.error = primary, verification, error
        self.calls = []

    def select(self, **inputs):
        self.calls.append(("primary", inputs))
        return Selection(self.primary, "selected", 0.)

    def confirm_absence(self, **inputs):
        self.calls.append(("verification", inputs))
        if self.error:
            raise self.error
        return Selection(self.verification, "selected", 0.)


def query(top_k=100, text="What label is on the folder?", images=()):
    return SearchRequest(user_id="alice", top_k=top_k, query=[{"type": "text", "text": text},
        *({"type": "image_url", "image_url": {"url": uri}} for uri in images)],
        options=["A. ORBIT-47", "B. ATLAS-92"])


NO_MEDIA = SimpleNamespace(get=lambda _: None)


def test_evidence_at_rank_100_keeps_original_100_ids_order_scores_and_objects():
    pool = [(memory(str(i), "Label ORBIT-47." if i == 99 else "A small unrelated note."), 1. - i / 200.)
            for i in range(100)]
    guard = Guard(primary=(99,))
    result = guard_sources(guard, query(), pool, NO_MEDIA)
    assert result == pool and all(before is after for before, after in zip(pool, result, strict=True))
    assert len(guard.calls) == 1 and len(guard.calls[0][1]["candidates"]) == 100
    assert "ORBIT-47" in guard.calls[0][1]["candidates"][99]["text"]


@pytest.mark.parametrize("fact", ["The folder is teal.", "The printed label reads ORBIT-47.",
    "There are three folders.", "I did not buy that folder.",
    "Correction: I bought the blue folder, not the green one.",
    "Yesterday's selected folder was the one I bought today."])
def test_first_false_negative_is_vetoed_by_direct_or_partial_evidence(fact):
    pool = [(memory("noise", "A closed notebook."), .9), (memory("fact", fact, session="later"), .4)]
    guard = Guard(primary=(), verification=(1,))
    assert guard_sources(guard, query(), pool, NO_MEDIA) == pool
    assert [kind for kind, _ in guard.calls] == ["primary", "verification"]
    assert fact in guard.calls[1][1]["candidates"][1]["text"]


def test_complete_twice_rejected_pool_can_still_abstain_without_blind_padding():
    pool = [(memory("unrelated", "A lake and trees."), .93)]
    guard = Guard()
    assert prepare_review(query(), pool, NO_MEDIA).complete
    assert guard_sources(guard, query(), pool, NO_MEDIA) == []
    assert len(guard.calls) == 2
    assert guard.calls[0][1]["options"] == query().options
    assert "ORBIT-47" not in guard.calls[0][1]["candidates"][0]["text"]


@pytest.mark.parametrize("text", ["prefix " * 800 + "The label is ORBIT-47.",
                                  "人工说明\x00" * 700 + "尾部标注 ORBIT-47。"])
def test_unseen_tail_cannot_be_cleared_as_absent(text):
    pool = [(memory("long", text), .7)]
    guard = Guard()
    review = prepare_review(query(), pool, NO_MEDIA)
    assert not review.complete and "source_text_not_fully_reviewed" in review.incomplete_reasons
    assert "ORBIT-47" not in review.candidates[0]["text"]
    assert guard_sources(guard, query(), pool, NO_MEDIA) == pool and len(guard.calls) == 1
    assert pool[0][0].raw_data["text_content"] == text


def test_missing_truncation_marker_does_not_mean_complete_review():
    # The first 2400 characters fit the byte budget, but the full source is longer.
    text = "x" * 3000 + "important atomic fact"
    review = prepare_review(query(), [(memory("long", text), .8)], NO_MEDIA)
    assert "Selector preview truncated" not in review.candidates[0]["text"]
    assert not review.complete


def test_seventeenth_unseen_image_preserves_all_original_sources():
    pool = [(memory(str(i), "Photo.", images=[f"photo-{i}"]), .9 - i / 100.) for i in range(17)]
    media = SimpleNamespace(get=lambda _: (b"original artificial image", "image/png"))
    guard = Guard()
    review = prepare_review(query(), pool, media)
    assert sum(len(c["images"]) for c in review.candidates) == 16
    assert not review.complete and "source_image_not_reviewed" in review.incomplete_reasons
    assert guard_sources(guard, query(), pool, media) == pool and len(guard.calls) == 1


def test_byte_budget_omission_preserves_source_and_does_not_generate_empty(monkeypatch):
    monkeypatch.setattr("mandol_competition.multimodal.atomic_guard.REVIEW_URI_BYTES", 32)
    pool = [(memory("photo", "Photo.", images=["photo"]), .7)]
    media = SimpleNamespace(get=lambda _: (b"original artificial image", "image/png"))
    assert guard_sources(Guard(), query(), pool, media) == pool


def test_missing_source_media_is_not_proof_of_irrelevance():
    pool = [(memory("photo", "Photo.", images=["missing"]), .7)]
    review = prepare_review(query(), pool, NO_MEDIA)
    assert not review.complete and "source_image_missing" in review.incomplete_reasons
    assert guard_sources(Guard(), query(), pool, NO_MEDIA) == pool


def test_shared_image_reference_is_complete_when_original_is_actually_supplied():
    pool = [(memory("a", "First photo.", images=["shared"]), .9),
            (memory("b", "Second photo.", images=["shared"]), .8)]
    media = SimpleNamespace(get=lambda _: (b"original artificial image", "image/png"))
    review = prepare_review(query(), pool, media)
    assert review.complete and sum(len(c["images"]) for c in review.candidates) == 1
    assert "image_in_candidate" in review.candidates[1]["text"]
    guard = Guard(verification=(1,))
    assert guard_sources(guard, query(), pool, media) == pool


def test_unseen_query_image_also_blocks_an_absence_verdict():
    request = query(images=["data:image/png;base64,QUJD"] * 17)
    pool = [(memory("a", "A photo description."), .8)]
    review = prepare_review(request, pool, NO_MEDIA)
    assert not review.complete and "query_image_not_reviewed" in review.incomplete_reasons
    guard = Guard()
    assert guard_sources(guard, request, pool, NO_MEDIA) == pool and len(guard.calls) == 1


@pytest.mark.parametrize("top_k", [1, 7, 17, 100])
def test_existing_requested_limit_and_scores_are_preserved(top_k):
    pool = [(memory(str(i), "An explicit atomic fact."), .98 - i / 1000.) for i in range(100)]
    assert guard_sources(Guard(primary=(0,)), query(top_k), pool, NO_MEDIA) == pool[:top_k]


def test_empty_corpus_needs_no_selector_or_media_calls():
    assert guard_sources(None, query(), [], None) == []


def test_verifier_failure_preserves_rate_limit_instead_of_false_abstention_or_silent_success():
    guard = Guard(error=InferenceRateLimited(retry_after=83))
    with pytest.raises(InferenceRateLimited) as caught:
        guard_sources(guard, query(), [(memory("a", "A lake."), .8)], NO_MEDIA)
    assert caught.value.retry_after == 83 and len(guard.calls) == 2


@pytest.mark.parametrize("indices", [(100,), (True,), (0, 1), (-1,)])
def test_bad_verifier_output_is_typed_failure(indices):
    with pytest.raises(InferenceSchemaError):
        guard_sources(Guard(verification=indices), query(), [(memory("a", "A lake."), .8)], NO_MEDIA)


@pytest.mark.parametrize("response", [None, Selection((0,), "fallback", 0.), Selection((0,), "empty", 0.)])
def test_inconsistent_or_missing_guard_response_never_becomes_false_empty(response):
    guard = Guard()
    guard.select = lambda **_: response
    with pytest.raises(InferenceSchemaError):
        guard_sources(guard, query(), [(memory("a", "A lake."), .8)], NO_MEDIA)


def test_v2_changes_method_identity_but_not_canonical_profile():
    from dataclasses import replace
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.multimodal.models import load_profile
    from mandol_competition.multimodal.search_profiles import BASELINE, identity, descriptor
    original = MultimodalConfig(inference_contract="aml-official-eligible-v1", model_device="external-http")
    config = replace(original, vlm_backend="openai_compat", evidence_policy="source-grounded-atomic-v2")
    before, after = load_profile(original), load_profile(config)
    assert before.hash == after.hash
    assert identity(after, BASELINE, config.evidence_policy) != identity(after, BASELINE, "source-grounded-v1")
    assert descriptor(after, BASELINE, config.evidence_policy)["nonempty_results"] == "baseline-ids-order-scores"
