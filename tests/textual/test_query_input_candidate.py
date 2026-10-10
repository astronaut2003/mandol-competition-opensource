from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from mandol_competition.textual.api.competition import CompetitionMemoryService
from mandol_competition.textual.core.query_inputs import QueryInputPolicy, candidate_query
from mandol_substrate.state import CanonicalMemory, StateUnavailable
from mandol_competition.textual.core.remote_inference import RemoteSpladeClient


def test_choices_are_unverified_all_or_original_and_default_ignores_them():
    choices = ["First hypothesis", "Second hypothesis"]
    query = "Which information applies?"
    assert candidate_query(None, query, choices) == query
    policy = QueryInputPolicy(max_rendered_chars=512)
    rendered, reason = policy.render(query, choices)
    assert query in rendered and all(choice in rendered for choice in choices)
    assert "Unverified" in rendered and reason == "ALL_CHOICES_UNVERIFIED_HINTS"
    assert choices == ["First hypothesis", "Second hypothesis"]
    assert policy.render(query, choices + ["x" * 600])[0] == query  # Never favor first choices.
    assert policy.render(query, None)[0] == query
    with pytest.raises(ValueError):
        policy.render(query, {"gold": "never accepted"})


@pytest.mark.parametrize("enabled", [False, True])
def test_real_service_call_path_enriches_only_candidate_generation(enabled):
    # The real service orchestrator with fake dependencies: not real-model qualification.
    service = CompetitionMemoryService.__new__(CompetitionMemoryService)
    source = CanonicalMemory("raw", "Actual stored evidence", metadata={"user_id": "user"})
    candidate = {"query_input_policy": {"version": "textual-all-choice-hints-v1", "max_rendered_chars": 16000}} if enabled else None
    service.settings = SimpleNamespace(evaluation_namespace="textual", quality_candidate=candidate)
    service.admission = lambda: nullcontext()
    service._hlm_lease = lambda *args, **kwargs: nullcontext()
    service.run = SimpleNamespace(ensure_scope=lambda scope: None)
    view = SimpleNamespace(graph=SimpleNamespace(traverse=lambda *args, **kwargs: []))
    service.coordinator = SimpleNamespace(search_view=lambda scope: nullcontext(view))
    service.state = SimpleNamespace(get_memories=lambda scope, uids: {uid: source for uid in uids})
    service.profile = SimpleNamespace(options={"rerank_k": 100, "graph_hops": 0,
        "graph_neighbors": 2, "graph_candidate_limit": 10, "semantic_neighbors": 0, "semantic_threshold": .5})
    service.query_policy = service.context_policy = service.return_policy = None
    calls = {}

    def encode(query):
        calls["encode"] = query
        return [1.], {}

    def retrieve(view, query, dense, sparse, **kwargs):
        calls["retrieve"] = query
        return [(SimpleNamespace(uid="raw"), .1)], {}

    def rerank(query, documents):
        calls["rerank"] = query
        assert documents == [source.text_content]
        return [.8]

    service.encoders = SimpleNamespace(encode_query=encode,
        splade=SimpleNamespace(count_tokens=lambda text: 40, max_tokens=512))
    service.retriever = SimpleNamespace(retrieve=retrieve)
    service.reranker = SimpleNamespace(rerank=rerank)
    assert service.search("user", "Original question", 100, choices=["Choice A", "Choice B"])[0]["content"] == source.text_content
    assert calls["rerank"] == "Original question"
    assert calls["encode"] == calls["retrieve"]
    assert ("Choice A" in calls["encode"]) == enabled


def test_sparse_budget_never_truncates_to_a_subset_of_choices():
    candidate = {"query_input_policy": {"version": "textual-all-choice-hints-v1"}}
    query, choices = "Original question", ["Short first choice", "Long last choice"]
    assert candidate_query(candidate, query, choices,
        sparse_client=SimpleNamespace(count_tokens=lambda text: 513, max_tokens=512)) == query
    with pytest.raises(ValueError, match="admission"):
        candidate_query(candidate, query, choices)


@pytest.mark.parametrize("response", [
    {"tokens": [101, 102], "count": True}, {"tokens": [101], "count": 2},
    {"tokens": [True], "count": 1}, {"tokens": [30522], "count": 1},
    {"tokens": [], "count": 0},
])
def test_remote_sparse_token_budget_rejects_malformed_admission(response):
    client = RemoteSpladeClient.__new__(RemoteSpladeClient)
    client._request = lambda *args: response
    with pytest.raises(StateUnavailable):
        client.count_tokens("Nonempty query")
