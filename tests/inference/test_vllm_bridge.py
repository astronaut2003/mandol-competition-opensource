import importlib
import json
from pathlib import Path
import sys

import httpx
from fastapi.testclient import TestClient
import pytest

from mandol_competition.inference.http import HTTPTransport
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes


def setup(monkeypatch, *, wrong_dimension=False, wrong_revision=False):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / 'services/inference/src'))
    module = importlib.import_module('mandol_inference.vllm_bridge')
    profile = load_profile(MultimodalConfig(model_device='external-http',
        retrieval_profile_path=root / 'src/mandol_competition/multimodal/profiles/vllm-bf16-v1.json'))
    calls = []
    def client(recipe):
        def handle(request):
            if request.url.path == '/version':
                return httpx.Response(200, json={'version': '0.25.1'})
            if request.url.path == '/v1/models':
                return httpx.Response(200, json={'data': [{'id': recipe.served_alias,
                    'root': '/cached/' + ('wrong' if wrong_revision else recipe.model_revision)}]})
            if request.url.path == '/health':
                return httpx.Response(200, json={})
            body = json.loads(request.content)
            calls.append((recipe.channel, body))
            if recipe.channel == 'reranker':
                data = [{'index': i, 'score': .5} for i in range(len(body['text_2']))]
            else:
                dim = 2 if wrong_dimension else recipe.dimension
                data = [{'index': i, 'embedding': [1.] + [0.] * (dim - 1)} for i in range(len(body['input']))]
            return httpx.Response(200, json={'model': recipe.served_alias, 'data': data})
        return HTTPTransport('http://example.invalid', transport=httpx.MockTransport(handle))
    rs = recipes(profile)
    app = module.create_app(profile=profile, dense_http=client(rs[0]), sparse_http=client(rs[1]), reranker_http=client(rs[2]))
    return app, rs, calls


def test_bridge_attests_native_identity_and_only_translates_http(monkeypatch):
    app, rs, calls = setup(monkeypatch)
    with TestClient(app) as client:
        assert len(client.get('/v1/recipe').json()['recipes']) == 3
        for recipe, route, fields in zip(rs, ['/v1/embeddings', '/v1/sparse', '/v1/score'],
            [{'input': ['q'], 'input_type': 'query'}, {'input': ['doc']}, {'query': 'q', 'documents': ['doc']}], strict=True):
            body = dict(model=recipe.served_alias, recipe_hash=recipe.hash, **fields)
            response = client.post(route, json=body)
            assert response.status_code == 200
            assert recipe.dtype == 'bfloat16'
        assert calls[0][1]['input'] == [rs[0].query_prompt + 'q']
        assert calls[0][1]['truncation_side'] == 'right'
        assert calls[2][1]['use_activation'] is True
    assert not {'torch', 'transformers', 'sentence_transformers'} & sys.modules.keys()


def test_bridge_rejects_wrong_dimension(monkeypatch):
    app, rs, _ = setup(monkeypatch, wrong_dimension=True)
    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json={'model': rs[0].served_alias,
            'recipe_hash': rs[0].hash, 'input': ['test']})
        assert response.status_code == 503


def test_bridge_rejects_wrong_model_snapshot_before_serving(monkeypatch):
    app, _, _ = setup(monkeypatch, wrong_revision=True)
    with pytest.raises(ValueError, match='snapshot mismatch'):
        with TestClient(app):
            pass


def test_reranker_bridge_has_no_dense_or_sparse_endpoint_dependency(monkeypatch):
    app,rs,_=setup(monkeypatch)
    module=importlib.import_module('mandol_inference.vllm_bridge')
    profile=load_profile(MultimodalConfig(model_device='external-http',
        retrieval_profile_path=Path(__file__).resolve().parents[2]/'src/mandol_competition/multimodal/profiles/vllm-bf16-v1.json'))
    for name in ('MANDOL_VLLM_EMBEDDING_ENDPOINT','MANDOL_VLLM_SPARSE_ENDPOINT'):
        monkeypatch.delenv(name,raising=False)
    seen=[]
    def handle(request):
        seen.append(request.url.path)
        if request.url.path=='/version':return httpx.Response(200,json={'version':'0.25.1'})
        if request.url.path=='/v1/models':return httpx.Response(200,json={'data':[
            {'id':rs[2].served_alias,'root':'/cached/'+rs[2].model_revision}]})
        if request.url.path=='/health':return httpx.Response(200,json={})
        return httpx.Response(200,json={'model':rs[2].served_alias,'data':[{'index':0,'score':.9}]})
    http=HTTPTransport('http://bge.invalid',transport=httpx.MockTransport(handle))
    isolated=module.create_app(profile=profile,reranker_http=http,channels=('reranker',))
    with TestClient(isolated) as client:
        assert client.get('/health').status_code==200
        declared=client.get('/v1/recipe').json()['recipes']
        assert [r['hash'] for r in declared]==[rs[2].hash]
        assert client.post('/v1/score',json={'model':rs[2].served_alias,
            'recipe_hash':rs[2].hash,'query':'q','documents':['evidence']}).json()['data'][0]['score']==.9
        assert client.post('/v1/embeddings',json={'model':rs[0].served_alias,
            'recipe_hash':rs[0].hash,'input':['q']}).status_code==404
    assert '/score' in seen
