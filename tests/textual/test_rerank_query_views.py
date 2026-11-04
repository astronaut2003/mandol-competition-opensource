import math
from types import SimpleNamespace

import pytest

from mandol_competition.inference.errors import InferenceInputRejected, InferenceDeadlineExceeded, InferenceRateLimited, InferenceSchemaError
from mandol_substrate.deadlines import deadline_budget
from mandol_competition.textual.core.rerank_query_views import (
    MAX_QUERY_VIEWS, MAX_VIEW_UTF8_BYTES, MAX_QUERY_UTF8_BYTES, RRF_K, RerankQueryLimitError,
    descriptor, plan_query, rerank_query_views,
)


class Scorer:
    def __init__(self, outcomes):
        self.outcomes, self.calls = iter(outcomes), []

    def rerank(self, query, documents):
        self.calls.append((query, documents))
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.parametrize("text", [" x\n original query ", "x" * MAX_VIEW_UTF8_BYTES,
                                  "你" * (MAX_VIEW_UTF8_BYTES // 3), "🙂" * 7000])
def test_short_query_and_documents_are_unchanged_and_provider_scores_preserved(text):
    documents = [" raw\n document ", "Second document"]
    client = Scorer([[.17, .93]])
    assert rerank_query_views(client, text, documents) == [.17, .93]
    assert client.calls == [(text, documents)] and client.calls[0][1] is documents
    assert plan_query(text).parts == (text,)


@pytest.mark.parametrize("query", ["x" * 50_001, "word " * 10_100,
    "开头原文\n" + "中文🙂混合文本。" * 8000 + "\n尾部限定词",
    "FIRST\n" + "a" * 31_000 + "\nMIDDLE\n" + "b" * 31_000 + "\nLAST"])
def test_lossless_split_keeps_all_characters_and_each_view_within_byte_limit(query):
    plan = plan_query(query)
    assert "".join(plan.parts) == query
    assert all(0 < len(part) <= MAX_VIEW_UTF8_BYTES for part in plan.parts)
    assert plan.original_chars == len(query) and plan.original_utf8_bytes == len(query.encode("utf-8"))
    assert len(plan.parts) <= MAX_QUERY_VIEWS
    if query == "x" * 50_001:
        assert len(plan.views) == 2


def test_two_views_share_all_original_documents_and_fuse_ranks_not_raw_scores():
    query = "x" * 50_001
    documents = ["source zero", "source one", "source two"]
    client = Scorer([[.99, .98, .01], [.02, .03, .01]])
    scores = rerank_query_views(client, query, documents)
    assert len(client.calls) == 2 and all(d is documents for _, d in client.calls)
    assert "".join(q for q, _ in client.calls) == query
    expected = [(1 / 61 + 1 / 62) * 61 / 2,
                (1 / 62 + 1 / 61) * 61 / 2, (1 / 63 + 1 / 63) * 61 / 2]
    assert scores == pytest.approx(expected)
    assert scores[0] == scores[1] > scores[2]
    assert all(math.isfinite(s) and 0 <= s <= 1 for s in scores)


def test_equal_provider_scores_share_rank_and_do_not_invent_index_preference():
    assert rerank_query_views(Scorer([[.2, .2], [.9, .9]]), "x" * 50_001, ["a", "b"]) == [1., 1.]


def test_whitespace_only_piece_is_retained_in_plan_but_not_sent_as_a_blank_query():
    query = "Important query\n" + " " * 50_000
    plan = plan_query(query)
    assert "".join(plan.parts) == query and len(plan.views) == 1
    client = Scorer([[.8]])
    assert rerank_query_views(client, query, ["a"]) == [.8] and len(client.calls) == 1


@pytest.mark.parametrize("second", [InferenceRateLimited(retry_after=91), InferenceDeadlineExceeded()])
def test_later_failure_propagates_without_a_partial_success(second):
    client = Scorer([[.9, .1], second])
    with pytest.raises(type(second)):
        rerank_query_views(client, "x" * 50_001, ["a", "b"])
    assert len(client.calls) == 2


@pytest.mark.parametrize("scores", [[.8], [float("nan"), .2], [True, .2], [1.1, .1]])
def test_bad_later_scores_do_not_leak_partial_first_view_results(scores):
    client = Scorer([[.9, .1], scores])
    with pytest.raises(InferenceSchemaError):
        rerank_query_views(client, "x" * 50_001, ["a", "b"])


def test_no_provider_dispatch_for_empty_documents_or_excessive_lossless_plan():
    client = Scorer([])
    assert rerank_query_views(client, "query", []) == []
    with pytest.raises(InferenceInputRejected):
        rerank_query_views(client, "x" * (MAX_QUERY_UTF8_BYTES + 1), ["a"])
    assert not client.calls


def test_expired_shared_deadline_dispatches_nothing():
    client = Scorer([])
    with deadline_budget(0):
        with pytest.raises(InferenceDeadlineExceeded):
            rerank_query_views(client, "x" * 50_001, ["a"])
    assert not client.calls


def test_deadline_between_views_stops_before_next_dispatch_and_returns_no_partial_scores(monkeypatch):
    from mandol_substrate import deadlines
    clock=[1000.]
    monkeypatch.setattr(deadlines,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    class SlowFirst(Scorer):
        def rerank(self, query, documents):
            result = super().rerank(query, documents)
            clock[0]+=.15
            return result
    client = SlowFirst([[.9, .1]])
    with deadline_budget(.1):
        with pytest.raises(InferenceDeadlineExceeded):
            rerank_query_views(client, "x" * 50_001, ["a", "b"])
    assert len(client.calls) == 1


def test_planning_that_consumes_the_shared_deadline_dispatches_nothing(monkeypatch):
    from mandol_substrate import deadlines
    from mandol_competition.inference import rerank_query_views as shared
    clock=[1000.]
    monkeypatch.setattr(deadlines,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    original=shared.encode_text
    def slow_encode(text):
        encoded=original(text)
        clock[0]+=.11
        return encoded
    monkeypatch.setattr(shared,'encode_text',slow_encode)
    client=Scorer([])
    with deadline_budget(.1):
        with pytest.raises(InferenceDeadlineExceeded):
            rerank_query_views(client,'x'*50_001,['a','b'])
    assert not client.calls


def test_effective_identity_names_normalized_rank_fusion_and_bounds():
    policy = descriptor()
    assert policy["same_document_pool"] and policy["rrf_k"] == RRF_K
    assert "not-provider-probability" in policy["score_semantics"] and len(policy["sha256"]) == 64
