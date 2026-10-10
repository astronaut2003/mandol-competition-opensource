import hashlib
import json

import httpx
import pytest

from mandol_competition.textual.core import native_reranker_http as native
from mandol_competition.textual.core.querit_reference import expected_profile
from mandol_competition.textual.core.reranker_client import (
    NativeReferenceRerankerClient,
    create_reranker_client,
)
from mandol_substrate.state.models import ProfileMismatch, StateUnavailable


def transport(profile, calls, *, incomplete=False):
    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "schema": native.SCHEMA, "status": "loaded",
                "model": profile["model_name"], "profile": profile,
                "profile_sha256": native.fingerprint(profile), "output_kind": "raw_score",
            })
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={
            "schema": native.SCHEMA, "model": profile["model_name"],
            "profile_sha256": native.fingerprint(profile),
            "score_semantics": profile["score_semantics"], "output_kind": "raw_score",
            "query_sha256": hashlib.sha256(body["query"].encode()).hexdigest(),
            "documents_sha256": native.fingerprint(body["documents"]),
            "data": ([{"index": 1, "score": .6}] if incomplete else [
                {"index": 1, "score": .6}, {"index": 0, "score": -.4},
            ]),
        })
    return httpx.MockTransport(handle)


def test_native_adapter_preserves_exact_inputs_negative_scores_and_order():
    profile, calls = expected_profile(), []
    client = NativeReferenceRerankerClient(
        "http://127.0.0.1:18106", profile, transport=transport(profile, calls),
    )
    try:
        assert client.validate_profile()["profile"] == profile
        assert client.profile.model == profile["model_name"]
        assert client.profile.revision == profile["model_revision"]
        assert client.rerank("query\n", [" doc", "中文"]) == [-.4, .6]
        assert calls[0]["query"] == "query\n"
        assert calls[0]["documents"] == [" doc", "中文"]
        assert client.rerank("empty corpus", []) == []
        assert len(calls) == 1
    finally:
        client.close()


def test_missing_native_score_never_replaced_with_zero_or_original_ranking():
    profile, calls = expected_profile(), []
    client = NativeReferenceRerankerClient(
        "http://127.0.0.1:18106", profile, transport=transport(profile, calls, incomplete=True),
    )
    try:
        with pytest.raises(StateUnavailable):
            client.rerank("q", ["a", "b"])
        assert len(calls) == 1
    finally:
        client.close()


def test_explicit_backend_and_model_contract_never_fall_back():
    with pytest.raises(ProfileMismatch):
        create_reranker_client("http://127.0.0.1", {"reranker_backend": "unknown"})
    changed = {**expected_profile(), "score_semantics": "sigmoid"}
    with pytest.raises(ValueError):
        NativeReferenceRerankerClient("http://127.0.0.1", changed)
    for value in (0, float("nan"), float("inf"), True):
        with pytest.raises(ProfileMismatch):
            NativeReferenceRerankerClient("http://127.0.0.1", expected_profile(), request_timeout=value)
