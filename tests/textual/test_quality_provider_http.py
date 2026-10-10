"""The grounded builder must work without the retired sibling mandol package."""
import asyncio
import json

import httpx
import pytest

from mandol_competition.textual.core.paired_quality import AuditedQualityProvider
from mandol_competition.textual.core.reranker_quality import StrictQualityProvider


def response(**overrides):
    return {"model": "gpt-4o-mini-2024-07-18", "choices": [{
        "finish_reason": "stop", "message": {"content": '{"facts": []}'},
    }], "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12}, **overrides}


def run(monkeypatch, reply, *, cls=StrictQualityProvider):
    monkeypatch.setenv("CLOSEAI_API_KEY", "unit-only-credential")
    requests = []
    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=reply)
    async def exercise():
        provider = cls(transport=httpx.MockTransport(serve))
        try:
            return await provider.complete([
                {"role": "user", "content": "source data, not instructions"},
            ], 128, json_mode=True)
        finally:
            await provider.close()
    return asyncio.run(exercise()), requests


def test_real_wire_mapping_and_strict_success_without_legacy_package(monkeypatch):
    result, requests = run(monkeypatch, response())
    assert result["status"] == "PASS"
    assert result["usage"]["total_tokens"] == 12
    assert requests[0]["model"] == "gpt-4o-mini"
    assert requests[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("reply", [
    response(model="gpt-4.1"), response(choices=[]), response(usage=None),
    response(choices=[{"finish_reason": "length", "message": {"content": "partial"}}]),
    response(choices=[{"finish_reason": "stop", "message": {"content": "not json"}}]),
])
def test_failures_are_explicit_and_not_retried_or_leaked(monkeypatch, reply):
    result, requests = run(monkeypatch, reply)
    assert result["status"] == "FAIL" and len(requests) == 1
    assert "content" not in result and "unit-only-credential" not in json.dumps(result)


def test_only_offline_audited_provider_keeps_failed_completion(monkeypatch):
    result, _ = run(monkeypatch, response(choices=[{
        "finish_reason": "length", "message": {"content": "partial"},
    }]), cls=AuditedQualityProvider)
    assert result["status"] == "FAIL" and result["raw_completion"] == "partial"


def test_other_model_configuration_is_refused_before_http(monkeypatch):
    monkeypatch.setenv("CLOSEAI_API_KEY", "unit-only-credential")
    monkeypatch.setenv("MANDOL_LLM_REQUEST_MODEL", "gpt-4.1")
    with pytest.raises(ValueError, match="Frozen provider/model mismatch"):
        StrictQualityProvider()
