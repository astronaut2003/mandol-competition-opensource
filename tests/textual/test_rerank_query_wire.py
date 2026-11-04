"""Real Textual search orchestration/SQLite/HTTP, simulated native provider."""
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import httpx
import pytest

from mandol_competition.inference.dashscope_rerank import BACKEND, DashScopeRerankerClient
from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.textual.api.app import create_app
from mandol_competition.textual.api.competition import CompetitionMemoryService
from mandol_competition.textual.api.config import Settings
from mandol_competition.textual.core.rerank_query_views import MAX_QUERY_VIEWS, MAX_VIEW_UTF8_BYTES, MAX_QUERY_UTF8_BYTES
from mandol_substrate.state import Mutation, Scope
from tests.common.test_publication import coordinator


def provider(handler):
    transport = HTTPTransport("https://rerank.invalid", max_attempts=2,
        transport=httpx.MockTransport(handler), sleep=lambda _: None)
    return DashScopeRerankerClient(transport=transport)


def service_for(store, profile, scope, memory, tmp_path, scorer):
    docs = [replace(memory, logical_uid="source-a", text_content="Full first source.\n" * 500),
            replace(memory, logical_uid="source-b", text_content="Full second source.")]
    publication, _ = coordinator(store, profile, tmp_path)
    publication.apply(scope, "artificial", "artificial", Mutation(memories=tuple(docs)), {})
    service = CompetitionMemoryService.__new__(CompetitionMemoryService)
    service._ready = False
    service.start = lambda: setattr(service, "_ready", True)
    service.shutdown = lambda: setattr(service, "_ready", False)
    service.health = lambda: True
    service.admission = nullcontext
    service._hlm_lease = lambda *_, **__: nullcontext()
    service.scope = lambda user: Scope(scope.namespace, user)
    service.run = SimpleNamespace(ensure_scope=lambda _: None)
    observed = {"embedding": [], "retrieval": []}
    def encode(query):
        observed["embedding"].append(query)
        return [1., 0., 0.], {1: 1.}
    service.encoders = SimpleNamespace(encode_query=encode)
    @contextmanager
    def search_view(selected_scope):
        with publication.search_view(selected_scope) as view:
            yield SimpleNamespace(published_seq=view.published_seq,
                graph=SimpleNamespace(traverse=lambda *_, **__: []))
    service.coordinator = SimpleNamespace(search_view=search_view)
    def retrieve(view, query, dense, sparse, **_):
        observed["retrieval"].append(query)
        return [(SimpleNamespace(uid=doc.logical_uid), .9) for doc in docs], {}
    service.retriever = SimpleNamespace(retrieve=retrieve)
    service.profile = SimpleNamespace(options={"rerank_k": 100, "graph_hops": 0,
        "graph_neighbors": 0, "graph_candidate_limit": 100, "semantic_neighbors": 0})
    service.settings = SimpleNamespace(quality_candidate=None, reranker_backend=BACKEND)
    service.state, service.reranker = store, scorer
    service.query_policy = service.context_policy = service.return_policy = None
    service.navigation = service.answer_utility_policy = service.observation_order_policy = None
    return service, observed, docs


@contextmanager
def api_for(service, tmp_path):
    settings = Settings(competition_run_root=str(tmp_path/"api"), evaluation_namespace="synthetic",
                        memory_system_key="artificial-key")
    with TestClient(create_app(settings, memory_service=service), raise_server_exceptions=False) as client:
        client.headers["X-Api-Key"] = "artificial-key"
        until = time.monotonic() + 5
        while client.get("/health").status_code != 200:
            assert time.monotonic() < until
            time.sleep(.01)
        yield client


def test_reported_400_is_reproduced_then_native_search_uses_two_full_pool_requests(store, profile, scope, memory, tmp_path):
    payloads = []
    def handler(request):
        value = json.loads(request.content)
        payloads.append(value)
        if len(value["input"]["query"]) > 50_000:
            return httpx.Response(400, json={"code": "InvalidParameter", "message": "query should not be larger than 50000"})
        tail = "TAIL-KEY" in value["input"]["query"]
        return httpx.Response(200, json={"output": {"results": [
            {"index": 1, "relevance_score": .9 if tail else .1},
            {"index": 0, "relevance_score": .1 if tail else .9}]}})
    scorer = provider(handler)
    query = "HEAD-KEY\n" + "x" * 50_000 + "\nTAIL-KEY"
    try:
        with pytest.raises(InferenceUnavailable):
            scorer._score_native(query, ["original a", "original b"])
        assert len(payloads) == 1  # The invalid original is never retried.
        payloads.clear()
        service, observed, documents = service_for(store, profile, scope, memory, tmp_path, scorer)
        with api_for(service, tmp_path) as client:
            response = client.post("/search", json={"user_id": scope.user_id, "query": query, "top_k": 100})
            assert response.status_code == 200
            results = response.json()["data"]
            assert [r["id"] for r in results] == ["source-a", "source-b"]
            assert [r["content"] for r in results] == [d.text_content for d in documents]
            json.dumps(results, allow_nan=False)
        assert observed["embedding"] == observed["retrieval"] == [query]
        assert len(payloads) == 2 and "".join(p["input"]["query"] for p in payloads) == query
        assert all(len(p["input"]["query"]) <= MAX_VIEW_UTF8_BYTES for p in payloads)
        assert all(p["input"]["documents"] == [d.text_content for d in documents] for p in payloads)
        assert store._rows("SELECT count(*) AS n FROM memories")[0]["n"] == 2
    finally:
        scorer.close()


def test_failure_on_second_piece_returns_429_without_partial_success(store, profile, scope, memory, tmp_path):
    class Scorer:
        calls = 0
        def rerank(self, query, documents):
            self.calls += 1
            if self.calls == 2:
                raise InferenceRateLimited(retry_after=91)
            return [.9, .1]
    scorer = Scorer()
    service, _, _ = service_for(store, profile, scope, memory, tmp_path, scorer)
    with api_for(service, tmp_path) as client:
        response = client.post("/search", json={"user_id": scope.user_id, "query": "x" * 50_001, "top_k": 100})
        assert response.status_code == 429 and response.headers["Retry-After"] == "91"
        assert "data" not in response.json() and scorer.calls == 2


def test_capacity_limit_returns_explicit_413_before_any_reranker_dispatch(store, profile, scope, memory, tmp_path):
    class Forbidden:
        def rerank(self, *_):
            pytest.fail("Oversized plan must not dispatch")
    service, _, _ = service_for(store, profile, scope, memory, tmp_path, Forbidden())
    with api_for(service, tmp_path) as client:
        response = client.post("/search", json={"user_id": scope.user_id,
            "query": "x" * (MAX_QUERY_UTF8_BYTES + 1), "top_k": 100})
        assert response.status_code == 413 and "data" not in response.json()


def test_16_parallel_searches_split_independently_with_full_original_sources(store, profile, scope, memory, tmp_path):
    class Scorer:
        def __init__(self):
            self.calls = []
        def rerank(self, query, documents):
            self.calls.append((query, tuple(documents)))
            return [.9, .1]
    scorer = Scorer()
    service, observed, docs = service_for(store, profile, scope, memory, tmp_path, scorer)
    query = "HEAD\n" + "word " * 10_100 + "\nTAIL"
    with api_for(service, tmp_path) as client:
        with ThreadPoolExecutor(max_workers=16) as workers:
            replies = list(workers.map(lambda _: client.post("/search", json={
                "user_id": scope.user_id, "query": query, "top_k": 100}), range(16)))
        assert len(replies) == 16 and all(r.status_code == 200 for r in replies)
        assert all([row["id"] for row in r.json()["data"]] == ["source-a", "source-b"] for r in replies)
        assert all([row["content"] for row in r.json()["data"]] == [d.text_content for d in docs] for r in replies)
    assert len(scorer.calls) == 32
    assert all(len(q) <= MAX_VIEW_UTF8_BYTES for q, _ in scorer.calls)
    assert observed["embedding"] == observed["retrieval"] == [query] * 16


def test_non_qwen_backend_keeps_original_unsplit_query_and_original_score_semantics(store, profile, scope, memory, tmp_path):
    class Original:
        def __init__(self):
            self.calls = []
        def rerank(self, query, documents):
            self.calls.append((query, documents))
            return [.91, .12]
    scorer = Original()
    service, _, _ = service_for(store, profile, scope, memory, tmp_path, scorer)
    service.settings.reranker_backend = "profile"
    query = "x" * 50_001
    with api_for(service, tmp_path) as client:
        result = client.post("/search", json={"user_id": scope.user_id, "query": query, "top_k": 100})
        assert result.status_code == 200 and [row["score"] for row in result.json()["data"]] == [.91, .12]
    assert len(scorer.calls) == 1 and scorer.calls[0][0] == query
