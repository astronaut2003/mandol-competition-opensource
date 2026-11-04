"""Reproduce the provider input limit using synthetic data and real Search HTTP."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPJointEncoders
from mandol_competition.inference.dashscope import DashScopeEmbeddingClient, MODEL, official_embedding_recipes
from mandol_competition.inference.errors import InferenceInputRejected, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.embedding_query_views import MAX_VIEW_TOKENS, MAX_QUERY_UTF8_BYTES, plan_query
from tests.textual.test_rerank_query_wire import api_for, service_for
from mandol_competition.inference.embedding_tokenizer import count_tokens


class Scorer:
    def __init__(self):
        self.calls = []
    def rerank(self, query, documents):
        self.calls.append((query, tuple(documents)))
        return [.9, .1]


def native_encoder(payloads, *, failure_at=None):
    def handler(request):
        body = json.loads(request.content)
        payloads.append(body)
        texts = body['input']['texts']
        if any(count_tokens(text) + 1 > 8192 for text in texts):
            return httpx.Response(400, json={'code': 'InvalidParameter',
                'message': 'Artificial token limit rejection'})
        if failure_at is not None and len(payloads) == failure_at:
            return httpx.Response(429, headers={'Retry-After': '93'}, json={'code': 'Throttling'})
        rows = []
        for i, text in enumerate(texts):
            # Orthogonal head/tail representations exercise meaningful merging.
            tail = 'TAIL-KEY' in text
            dense = [0., 1., 0.] if tail else [1., 0., 0.]
            rows.append({'text_index': i, 'embedding': dense + [0.] * 1021,
                'sparse_embedding': [{'index': 7 if tail else 3, 'value': 1.}]})
        return httpx.Response(200, json={'output': {'embeddings': rows}})
    native = DashScopeEmbeddingClient(transport=HTTPTransport('https://embedding.invalid',
        transport=httpx.MockTransport(handler), max_attempts=1, sleep=lambda _: None))
    dense, sparse = official_embedding_recipes()
    class Proxy:
        def request(self, method, path, payload=None, **_):
            if method == 'GET' and path == '/v1/recipe':
                from dataclasses import asdict
                return {'recipes': [{'hash': r.hash, 'recipe': asdict(r)} for r in (dense, sparse)]}
            assert method == 'POST' and path == '/v1/encode'
            pairs = native.encode(payload['input'], query=payload['input_type'] == 'query')
            return {'model': MODEL, 'data': [{'index': i, 'pair': {'embedding': p.dense,
                'sparse': {str(k): v for k, v in p.sparse.items()}}} for i, p in enumerate(pairs)]}
        def close(self):
            native.close()
    return native, HTTPJointEncoders(HTTPEmbeddingClient('unused', dense, transport=Proxy()),
                                    HTTPSparseEncoderClient('unused', sparse, transport=Proxy()))


def prepare(store, profile, scope, memory, tmp_path, encoder, scorer):
    service, observed, docs = service_for(store, profile, scope, memory, tmp_path, scorer)
    service.settings.inference_contract = 'aml-official-eligible-v1'
    service.encoders = encoder
    original = service.retriever.retrieve
    observed['pairs'] = []
    def retrieve(view, query, dense, sparse, **kwargs):
        observed['pairs'].append((dense, sparse))
        return original(view, query, dense, sparse, **kwargs)
    service.retriever = SimpleNamespace(retrieve=retrieve)
    return service, observed, docs


def test_provider_token_limit_then_whole_search_succeeds_with_all_query_parts(
        store, profile, scope, memory, tmp_path):
    payloads, scorer = [], Scorer()
    native, encoder = native_encoder(payloads)
    query = 'HEAD-KEY\n' + '你' * 33_001 + '\nTAIL-KEY'
    try:
        with pytest.raises(InferenceUnavailable) as raised:
            native.encode([query], query=True)
        assert raised.value.status_code == 400 and len(payloads) == 1
        payloads.clear()
        service, observed, docs = prepare(store, profile, scope, memory, tmp_path, encoder, scorer)
        before = store._rows('SELECT count(*) n FROM memories')[0]['n']
        with api_for(service, tmp_path) as client:
            response = client.post('/search', json={'user_id': scope.user_id, 'query': query, 'top_k': 100})
        assert response.status_code == 200
        hits = response.json()['data']
        assert [h['id'] for h in hits] == ['source-a', 'source-b']
        assert [h['content'] for h in hits] == [d.text_content for d in docs]
        assert observed['retrieval'] == [query]
        sent = [body['input']['texts'][0] for body in payloads]
        assert ''.join(sent) == query
        assert all(count_tokens(text) <= MAX_VIEW_TOKENS for text in sent)
        assert all(body['parameters']['text_type'] == 'query' for body in payloads)
        dense, sparse = observed['pairs'][0]
        assert math.hypot(*dense) == pytest.approx(1.) and set(sparse) == {3, 7}
        assert ''.join(q for q, _ in scorer.calls) == query  # Existing rerank split also preserves it.
        assert store._rows('SELECT count(*) n FROM memories')[0]['n'] == before
    finally:
        native.close()


def test_later_embedding_quota_returns_429_without_retrieval_or_partial_data(
        store, profile, scope, memory, tmp_path):
    payloads, scorer = [], Scorer()
    native, encoder = native_encoder(payloads, failure_at=2)
    try:
        service, observed, _ = prepare(store, profile, scope, memory, tmp_path, encoder, scorer)
        with api_for(service, tmp_path) as client:
            response = client.post('/search', json={'user_id': scope.user_id, 'query': '你' * 33_001, 'top_k': 100})
        assert response.status_code == 429 and response.headers['Retry-After'] == '93'
        assert 'data' not in response.json() and not observed['retrieval'] and not scorer.calls
        assert len(payloads) == 2
    finally:
        native.close()


def test_capacity_413_before_any_model_dispatch(store, profile, scope, memory, tmp_path):
    class Forbidden:
        def encode_query(self, *_):
            pytest.fail('Over-capacity query dispatched')
    service, _, _ = prepare(store, profile, scope, memory, tmp_path, Forbidden(), Scorer())
    with api_for(service, tmp_path) as client:
        response = client.post('/search', json={'user_id': scope.user_id,
            'query': 'x' * (MAX_QUERY_UTF8_BYTES + 1), 'top_k': 100})
    assert response.status_code == 413 and 'data' not in response.json()


def test_16_parallel_searches_keep_query_evidence_and_independent_merges(
        store, profile, scope, memory, tmp_path):
    payloads, scorer = [], Scorer()
    native, encoder = native_encoder(payloads)
    query = 'HEAD-KEY\n' + '字🙂' * 5500 + '\nTAIL-KEY'
    try:
        service, observed, docs = prepare(store, profile, scope, memory, tmp_path, encoder, scorer)
        with api_for(service, tmp_path) as client:
            with ThreadPoolExecutor(max_workers=16) as workers:
                replies = list(workers.map(lambda _: client.post('/search', json={
                    'user_id': scope.user_id, 'query': query, 'top_k': 100}), range(16)))
        assert len(replies) == 16 and all(r.status_code == 200 for r in replies)
        assert all([h['content'] for h in r.json()['data']] == [d.text_content for d in docs] for r in replies)
        assert observed['retrieval'] == [query] * 16
        assert all(count_tokens(p['input']['texts'][0]) <= MAX_VIEW_TOKENS for p in payloads)
        assert len(payloads) == 16 * len(plan_query(query).views)
    finally:
        native.close()


def test_reference_contract_keeps_original_encoder_call(store, profile, scope, memory, tmp_path):
    class Encoder:
        calls = []
        def encode_query(self, query):
            self.calls.append(query)
            return [1., 0., 0.], {3: 1.}
    encoder = Encoder()
    service, _, _ = prepare(store, profile, scope, memory, tmp_path, encoder, Scorer())
    service.settings.inference_contract = 'reference-v1'
    query = '你' * 33_001
    with api_for(service, tmp_path) as client:
        assert client.post('/search', json={'user_id': scope.user_id, 'query': query, 'top_k': 100}).status_code == 200
    assert encoder.calls == [query]
