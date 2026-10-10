"""Production transport contracts; synthetic responses unless explicitly opted in."""

import json
import os
from pathlib import Path

import httpx
import pytest

from mandol_competition.textual.core.models import ServingProfile
from mandol_competition.textual.core.querit_reference import PROMPT, expected_profile
from mandol_competition.textual.core.remote_inference import (
    RemoteDenseClient,
    RemoteQueritClient,
    RemoteSpladeClient,
    private_endpoint,
    relevance,
    sparse_vector,
)
from mandol_substrate.state.models import ProfileMismatch, StateUnavailable

ROOT = Path(__file__).resolve().parents[2]
PROFILE = json.loads(
    (ROOT / "src/mandol_competition/textual/profiles/source_b1_public_v1/profile.json").read_text()
)
DENSE = ServingProfile(**PROFILE["options"]["embedding_serving"])


def test_bf16_softmax_rounding_keeps_original_score():
    # Observed native BF16 Querit response. No normalization of either value.
    values = [0.5703125, 0.431640625]
    assert sum(values) == 1.001953125
    assert relevance(values) == values[1] - values[0]
    assert relevance([0.0, 1.0]) == 1.0


@pytest.mark.parametrize('values', [[0.5001, 0.50185], [0.5, 0.5078125],
                                    [-0.001, 1.001], [float('nan'), 1.0]])
def test_bf16_admission_does_not_relax_other_invalid_probabilities(values):
    with pytest.raises(StateUnavailable):
        relevance(values)


@pytest.mark.parametrize(
    "url",
    [
        "https://8.8.8.8",
        "http://8.8.8.8",
        "http://example.com",
        "http://127.0.0.1",
        "http://u:p@10.77.0.6",
        "http://10.77.0.6?key=x",
        "http://10.77.0.6/v1",
        None,
        "http://10.77.0.6#x",
    ],
)
def test_private_admission(url):
    with pytest.raises(ProfileMismatch):
        private_endpoint(url)


def reply(model, vectors, field):
    return {
        "model": model,
        "data": [{"index": i, field: v} for i, v in enumerate(vectors)],
    }


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [0.0] * 1024,
        [1.0] * 1023,
        [float("inf")] * 1024,
        [True] * 1024,
        ["1"] * 1024,
    ],
)
def test_dense_invalid_vectors_fail_closed(bad):
    # httpx JSON itself refuses nonfinite values; use raw response for fault injection.
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, content=json.dumps(reply(RemoteDenseClient.alias, [bad], "embedding"))
        )
    )
    client = RemoteDenseClient("http://10.77.0.6:8001", DENSE, transport=transport)
    try:
        with pytest.raises(StateUnavailable):
            client.embed(["text"])
    finally:
        client.close()


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "embedding": [1] * 1024}] * 2,
        [{"index": True, "embedding": [1] * 1024}],
        [{"index": 1, "embedding": [1] * 1024}],
    ],
)
def test_malformed_dense_batch(rows):
    client = RemoteDenseClient(
        "http://10.77.0.6:8001",
        DENSE,
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"model": RemoteDenseClient.alias, "data": rows}
            )
        ),
    )
    try:
        with pytest.raises(StateUnavailable):
            client.embed(["text"])
    finally:
        client.close()


def test_dense_order_normalization_and_timeout():
    calls = []

    def serve(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": RemoteDenseClient.alias,
                "data": [
                    {"index": 1, "embedding": [0, 2] + [0] * 1022},
                    {"index": 0, "embedding": [3] + [0] * 1023},
                ],
            },
        )

    client = RemoteDenseClient(
        "http://10.77.0.6:8001", DENSE, transport=httpx.MockTransport(serve)
    )
    try:
        vectors = client.embed(["first", "second"])
        assert vectors[0][0] == 1 and vectors[1][1] == 1
        assert calls[0]["truncate_prompt_tokens"] == 4096
    finally:
        client.close()

    def timeout(_):
        raise httpx.ReadTimeout("upstream failed")

    client = RemoteDenseClient(
        "http://10.77.0.6:8001", DENSE, transport=httpx.MockTransport(timeout)
    )
    try:
        with pytest.raises(StateUnavailable, match="ReadTimeout"):
            client.embed(["text"])
    finally:
        client.close()


def test_splade_documents_queries_and_positive_coordinates():
    calls = []

    def serve(request):
        payload = json.loads(request.content)
        calls.append(payload)
        values = [0.0] * 30522
        values[3 if payload["input"] == ["first"] else 30521] = 2.5
        return httpx.Response(200, json=reply("splade-v3", [values], "data"))

    client = RemoteSpladeClient(
        "http://10.77.0.6:8002", transport=httpx.MockTransport(serve)
    )
    try:
        assert client.encode_documents(["first", "second"]) == [{3: 2.5}, {30521: 2.5}]
        assert client.encode_query("first") == {3: 2.5}
        assert all(
            p["task"] == "embed"
            and len(p["input"]) == 1
            and p["truncate_prompt_tokens"] == 512
            for p in calls
        )
    finally:
        client.close()


@pytest.mark.parametrize(
    "bad", [[], [0] * 30522, [-1] + [0] * 30521, [float("nan")] + [0] * 30521]
)
def test_sparse_malformed_no_empty_fallback(bad):
    with pytest.raises(StateUnavailable):
        sparse_vector(bad)


@pytest.mark.parametrize(
    "bad", [[], [1], [True, False], [0, 0], [0.3, 0.3], [-0.1, 1.1], [float("nan"), 1]]
)
def test_querit_probabilities(bad):
    with pytest.raises(StateUnavailable):
        relevance(bad)


def test_official_preparation_b1_order_score_and_identity():
    calls = []

    def serve(request):
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        calls.append((path, body))
        if path == "/health":
            return httpx.Response(200)
        if path == "/version":
            return httpx.Response(200, json={"version": "0.29.0"})
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "querit-4b",
                            "root": RemoteQueritClient.root,
                            "max_model_len": 40960,
                        }
                    ]
                },
            )
        if path == "/openapi.json":
            return httpx.Response(200, json={"paths": {"/classify": {}}})
        if path == "/tokenize":
            ids = {"[CLS]": [151665], "<|endoftext|>": [151643]}.get(
                body["prompt"], [8] if body["prompt"].endswith("second") else [7]
            )
            return httpx.Response(200, json={"tokens": ids, "count": len(ids)})
        if path == "/detokenize":
            return httpx.Response(
                200,
                json={
                    "prompt": "[CLS]" if body["tokens"] == [151665] else "<|endoftext|>"
                },
            )
        assert path == "/classify" and len(body["input"]) == 1
        assert body["input"][0][-1] == 151665
        assert "truncate_prompt_tokens" not in body and not body["add_special_tokens"]
        probs = [0.9, 0.1] if body["input"][0][0] == 8 else [0.2, 0.8]
        return httpx.Response(
            200,
            json={
                "model": "querit-4b",
                "data": [{"index": 0, "num_classes": 2, "probs": probs}],
            },
        )

    client = RemoteQueritClient(
        "http://10.77.0.15:8003",
        expected_profile(batch_size=1),
        transport=httpx.MockTransport(serve),
    )
    try:
        client.validate_profile()
        assert client.rerank("query", ["first", "second"]) == pytest.approx([0.6, -0.8])
        prompts = [body["prompt"] for path, body in calls if path == "/tokenize"]
        assert PROMPT.format(query="query", document="first") in prompts
        assert (
            client.attestation()["limitations"]["numerical_reference_oracle"]
            == "NOT_COMPLETED"
        )
    finally:
        client.close()
    with pytest.raises(ProfileMismatch):
        RemoteQueritClient("http://10.77.0.15:8003", expected_profile(batch_size=8))


@pytest.mark.parametrize(
    "fault", ["alias", "root", "version", "length", "task", "health"]
)
def test_admission_never_accepts_unexpected_server(fault):
    def serve(request):
        path = request.url.path
        if path == "/health":
            return httpx.Response(503 if fault == "health" else 200)
        if path == "/version":
            return httpx.Response(
                200, json={"version": "0.17.1" if fault == "version" else "0.29.0"}
            )
        if path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "bad"
                            if fault == "alias"
                            else RemoteDenseClient.alias,
                            "root": "bad"
                            if fault == "root"
                            else RemoteDenseClient.root,
                            "max_model_len": 32 if fault == "length" else 32768,
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"paths": {}})

    client = RemoteDenseClient(
        "http://10.77.0.6:8001", DENSE, transport=httpx.MockTransport(serve)
    )
    try:
        with pytest.raises(StateUnavailable):
            client.validate_profile()
    finally:
        client.close()


@pytest.mark.integration_vllm
@pytest.mark.skipif(
    os.getenv("MANDOL_VPC_TEST") != "1", reason="explicit VPC integration opt-in"
)
def test_real_production_clients():
    dense = RemoteDenseClient(os.environ["EMBEDDING_ENDPOINT"], DENSE)
    splade = RemoteSpladeClient(os.environ["SPLADE_ENDPOINT"])
    querit = RemoteQueritClient(
        os.environ["RERANK_ENDPOINT"], expected_profile(batch_size=1)
    )
    try:
        for client in (dense, splade, querit):
            client.validate_profile()
        docs = ["My project is Orion.", "I planted tomatoes in the garden."]
        assert all(len(v) == 1024 for v in dense.embed(docs))
        assert all(
            v and all(0 <= i < 30522 and w > 0 for i, w in v.items())
            for v in splade.encode_documents(docs)
        )
        scores = querit.rerank("What is my project?", docs)
        assert len(scores) == 2 and scores[0] > scores[1]
        again = querit.rerank("What is my project?", docs)
        assert again == pytest.approx(scores, abs=0.02)
    finally:
        for client in (dense, splade, querit):
            client.close()
