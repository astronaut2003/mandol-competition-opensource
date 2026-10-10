import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes


def test_external_fp32_worker_contract_without_loading_models():
    path = Path(__file__).resolve().parents[2] / 'services/inference/src/mandol_inference/multimodal.py'
    spec = importlib.util.spec_from_file_location('worker', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    profile = load_profile(MultimodalConfig())
    class Encoder:
        def encode_dense(self, texts, query=False): return [[1.] + [0.] * 1023 for _ in texts]
        def encode_sparse(self, texts): return [{4: 1e-8, 9: 2.} for _ in texts]
        def encode_query(self, text): return self.encode_dense([text])[0], self.encode_sparse([text])[0]
    class Reranker:
        def score(self, query, documents): return [.5 for _ in documents]
    with TestClient(module.create_app(profile=profile, encoders=Encoder(), reranker=Reranker())) as client:
        assert client.get('/health').status_code == 200
        assert len(client.get('/v1/recipe').json()['recipes']) == 3
        for recipe, route, body in zip(recipes(profile), ['/v1/embeddings', '/v1/sparse', '/v1/score'],
                [{'input': ['text']}, {'input': ['text']}, {'query': 'query', 'documents': ['text']}], strict=True):
            body.update(model=recipe.served_alias, recipe_hash=recipe.hash)
            assert len(client.post(route, json=body).json()['data']) == 1
            body['recipe_hash'] = 'wrong'
            assert client.post(route, json=body).status_code == 409
    import sys
    assert not {'torch', 'transformers', 'sentence_transformers'} & sys.modules.keys()
