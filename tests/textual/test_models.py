import asyncio
import os
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from mandol_competition.textual.core.models import ServingProfile, VllmModelClient
from mandol_substrate.state import ProfileMismatch, StateUnavailable

EMBED = ServingProfile(
    "Qwen/Qwen3-Embedding-0.6B",
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    1024,
)
RERANK = ServingProfile(
    "Qwen/Qwen3-Reranker-0.6B",
    "6e9e69830b95c52b5fd889b7690dda3329508de3",
    "6e9e69830b95c52b5fd889b7690dda3329508de3",
    1,
    task="reranker",
    normalize=False,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "bfloat16"),
        ("runner", "generate"),
        ("max_tokens", 512),
        ("score_template_sha256", "wrong-template"),
        ("hf_overrides", {}),
    ],
)
def test_runtime_profile_mismatch_fails_before_model_use(field, value):
    from mandol_competition.textual.core.serving import RERANK_OVERRIDES, SCORE_TEMPLATE_HASH

    runtime = {
        "contract": "vllm-runtime-v1",
        "dtype": "float32",
        "runner": "pooling",
        "max_tokens": 4096,
        "score_template_sha256": SCORE_TEMPLATE_HASH,
        "hf_overrides": RERANK_OVERRIDES,
    }
    runtime[field] = value
    client = VllmModelClient(
        "http://test",
        RERANK,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=runtime)
        ),
    )
    try:
        with pytest.raises(ProfileMismatch):
            client.validate_profile()
    finally:
        client.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"data": [{"index": 0, "embedding": [0] * 3}]},
        {"data": [{"index": 0, "embedding": [float("nan")] * 3}]},
        {"data": [{"index": 2, "embedding": [1, 0, 0]}]},
        {"data": [{"index": 0, "embedding": [1, 0]}]},
    ],
)
def test_bad_embedding_no_fallback(payload):
    import json

    client = VllmModelClient(
        "http://test",
        replace(EMBED, dimension=3),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(payload))
        ),
    )
    try:
        with pytest.raises(StateUnavailable):
            client.embed(["x"])
    finally:
        client.close()


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure", ["timeout", "reset", "oom", "unavailable", "malformed", "partial"]
)
def test_rerank_failure_sync_async(failure, monkeypatch):
    monkeypatch.setattr("mandol_competition.textual.core.models.time.sleep", lambda seconds: None)

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("injected", request=request)
        if failure == "reset":
            raise httpx.ReadError("reset", request=request)
        if failure in {"oom", "unavailable"}:
            return httpx.Response(503, json={"error": failure})
        if failure == "malformed":
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(
            200,
            json={"model": RERANK.served_name, "data": [{"index": 0, "score": 0.5}]},
        )

    client = VllmModelClient(
        "http://test", RERANK, transport=httpx.MockTransport(handler)
    )
    client._preparation = SimpleNamespace(
        prepare=lambda query, document, instruction: SimpleNamespace(document=document)
    )
    try:
        with pytest.raises(StateUnavailable):
            client.rerank("q", ["a", "b"])
        with pytest.raises(StateUnavailable):
            asyncio.run(client.rerank_async("q", ["a", "b"]))
    finally:
        client.close()


@pytest.mark.integration_vllm
def test_real_embedding_batch_profile_tokenizer():
    url = os.environ.get("MANDOL_VLLM_EMBED_URL")
    if not url:
        pytest.skip("explicit real vLLM embedding URL required")
    from transformers import AutoTokenizer
    client = VllmModelClient(url, EMBED, batch_size=2)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            EMBED.model,
            revision=EMBED.revision,
            cache_dir=os.environ.get("MANDOL_MODEL_CACHE"),
            local_files_only=True,
        )
        assert client.validate_profile(tokenizer)
        batch = client.embed(["Alice works at a hospital.", "北京的旅行", "memory"])
        assert len(batch) == 3 and all(len(v) == 1024 for v in batch)
        assert np.linalg.norm(batch[0]) == pytest.approx(1, abs=1e-6)
        assert client.embed(["Alice works at a hospital."])[0] == pytest.approx(
            batch[0], rel=1e-5, abs=1e-6
        )
        assert client.health()
        wrong = VllmModelClient(
            url, replace(EMBED, revision="a" * 40, tokenizer_revision="a" * 40)
        )
        try:
            with pytest.raises(ProfileMismatch):
                wrong.validate_profile()
        finally:
            wrong.close()
    finally:
        client.close()


@pytest.mark.integration_vllm
def test_real_reranker_sync_async_ordering_profile():
    url = os.environ.get("MANDOL_VLLM_RERANK_URL")
    if not url:
        pytest.skip("explicit real vLLM reranker URL required")
    from transformers import AutoTokenizer
    client = VllmModelClient(url, RERANK, batch_size=2)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            RERANK.model,
            revision=RERANK.revision,
            cache_dir=os.environ.get("MANDOL_MODEL_CACHE"),
            local_files_only=True,
        )
        assert client.validate_profile(tokenizer)
        docs = [
            "The capital of China is Beijing.",
            "Cats enjoy sleeping.",
            "Paris is the capital of France.",
        ]
        scores = client.rerank("What is the capital of China?", docs)
        assert len(scores) == 3 and scores[0] > max(scores[1:])
        assert asyncio.run(
            client.rerank_async("What is the capital of China?", docs)
        ) == pytest.approx(scores, rel=1e-5, abs=1e-6)
        assert client.health()
    finally:
        client.close()
