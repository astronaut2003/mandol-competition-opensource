"""A provider pair is accepted atomically; no incomplete vectors reach storage."""
from copy import deepcopy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from mandol_competition.inference.dashscope import (
    DashScopeEmbeddingClient, EncodedPair, MODEL, official_embedding_recipes, validate_pairs,
)
from mandol_competition.inference.errors import InferenceSchemaError, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport


def row(index=0):
    return {"text_index": index, "embedding": [1.] + [0.] * 1023,
            "sparse_embedding": [{"index": 51765, "value": .4}]}


@pytest.mark.parametrize("bad", [None, [], {"output": None}, {"output": []},
    {"output": {"embeddings": []}}])
def test_partial_and_invalid_envelopes(bad):
    with pytest.raises(InferenceSchemaError):
        validate_pairs(bad, 1)


@pytest.mark.parametrize("field,value", [
    ("embedding", [1., 0.]), ("embedding", [0.] * 1024),
    ("embedding", [float("nan")] + [0.] * 1023),
    ("embedding", [True] + [0.] * 1023),
    ("sparse_embedding", []),
    ("sparse_embedding", [{"index": -1, "value": 1.}]),
    ("sparse_embedding", [{"index": 2**32, "value": 1.}]),
    ("sparse_embedding", [{"index": 7, "value": -1.}]),
    ("sparse_embedding", [{"index": 7, "value": float("nan")}]),
    ("sparse_embedding", [{"index": 7, "value": 1.}, {"index": 7, "value": 2.}]),
])
def test_vectors_fail_closed(field, value):
    bad = row()
    bad[field] = value
    with pytest.raises(InferenceSchemaError):
        validate_pairs({"output": {"embeddings": [bad]}}, 1)


def test_native_batching_query_mode_and_pair_order():
    calls = []
    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"output": {"embeddings":
            [row(i) for i in reversed(range(len(body["input"]["texts"])))]}})
    wire = HTTPTransport("https://provider.invalid", transport=httpx.MockTransport(handler))
    client = DashScopeEmbeddingClient(transport=wire)
    assert len(client.encode(["q"] * 11, query=True)) == 11
    assert [len(c["input"]["texts"]) for c in calls] == [10, 1]
    assert all(c["model"] == MODEL and c["parameters"]["text_type"] == "query"
               and c["parameters"]["output_type"] == "dense&sparse" for c in calls)
    client.close()


def test_native_unknown_outcome_does_not_bill_again():
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private", request=request)
    client = DashScopeEmbeddingClient(transport=HTTPTransport("https://provider.invalid",
        transport=httpx.MockTransport(handler), max_attempts=3))
    with pytest.raises(InferenceUnavailable):
        client.encode(["data"])
    assert len(calls) == 1
    client.close()


def test_bridge_recipe_gate_and_native_sparse_ids(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / 'services/inference/src'))
    create_app = importlib.import_module('mandol_inference.dashscope_bridge').create_app
    class Fake:
        def __init__(self):
            self.calls = []
        def encode(self, texts, query=False):
            self.calls.append((texts, query))
            return [EncodedPair([1.] + [0.] * 1023, {51765: .4}) for _ in texts]
        def close(self):
            pass
    provider = Fake()
    dense, sparse = official_embedding_recipes()
    with TestClient(create_app(lambda: provider)) as client:
        payload = {"model": MODEL, "input": ["question"], "input_type": "query",
                   "dense_recipe_hash": dense.hash, "sparse_recipe_hash": sparse.hash}
        response = client.post("/v1/encode", json=payload)
        assert response.status_code == 200
        assert response.json()["data"][0]["pair"]["sparse"] == {"51765": .4}
        assert provider.calls[-1] == (["question"], True)
        count = len(provider.calls)
        wrong = deepcopy(payload)
        wrong["dense_recipe_hash"] = "wrong"
        assert client.post("/v1/encode", json=wrong).status_code == 409
        assert len(provider.calls) == count
        assert client.get("/health").status_code == 200


def test_mm_official_profile_keeps_bge_recipe_and_baseline_unmodified():
    from pathlib import Path
    from mandol_competition.multimodal.models import load_profile
    from mandol_competition.multimodal.inference import recipes
    from mandol_competition.multimodal.config import MultimodalConfig
    baseline = load_profile(MultimodalConfig(retrieval_profile_path=Path("src/mandol_competition/multimodal/profiles/vllm-bf16-v1.json"), model_device="external-http"))
    candidate = load_profile(MultimodalConfig(retrieval_profile_path=Path("configs/profiles/aml-official-eligible/multimodal-v1.json"), model_device="external-http"))
    before, after = recipes(baseline), recipes(candidate)
    assert before[2].hash == after[2].hash
    assert candidate.embedding_model == candidate.splade_model == MODEL
    assert before[0].hash != after[0].hash
    assert candidate.options["representation_policy"] == baseline.options["representation_policy"]
