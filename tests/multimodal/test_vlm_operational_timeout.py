"""Transport controls never change model identity or visual evidence policy."""
from dataclasses import replace

import httpx
import pytest

from mandol_competition.multimodal.config import ConfigurationError, MultimodalConfig, get_config
from mandol_competition.observability.diagnostics import _exception_metadata
from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.http import deadline_budget
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient, VisionRequestError


def test_visual_attempt_has_more_headroom_with_a_bounded_environment_override(monkeypatch):
    assert MultimodalConfig().vlm_timeout_seconds == 180
    get_config.cache_clear()
    try:
        monkeypatch.setenv("MANDOL_MM_VLM_TIMEOUT_SECONDS", "240")
        assert get_config().vlm_timeout_seconds == 240
        assert get_config().vlm_model == "gpt-4o-mini"
    finally:
        get_config.cache_clear()


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 601])
def test_visual_timeout_cannot_escape_the_bounded_transport_policy(value):
    with pytest.raises(ConfigurationError):
        replace(MultimodalConfig(), vlm_timeout_seconds=value)


@pytest.mark.parametrize("name", ["ReadTimeout", "ReadError", "RemoteProtocolError", "ConnectTimeout"])
def test_network_failure_type_is_observable_without_message_or_body(name):
    error = InferenceUnavailable(name, unknown_outcome=True)
    safe = _exception_metadata(error)
    assert safe["inference_error_class"] == name and safe["unknown_outcome"] is True
    assert "message" not in safe and "body" not in safe


@pytest.mark.parametrize("unsafe", ["private_prompt_or_secret", ["ReadTimeout"], {"body": "private"}])
def test_diagnostics_ignore_unrecognized_or_non_string_error_metadata(unsafe):
    error = InferenceUnavailable("ReadTimeout")
    error.error_class = unsafe
    assert "inference_error_class" not in _exception_metadata(error)


@pytest.mark.parametrize("deadline", [None, .5])
def test_longer_visual_timeout_keeps_the_outer_deadline_and_never_repeats_unknown_calls(deadline):
    seen = []
    def timeout(request):
        seen.append(request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("private fixture content must not be logged", request=request)
    client = OpenAICompatVisionClient(model="gpt-4o-mini", api_key="owned-fixture-key",
        base_url="http://owned-fixture.invalid/v1", timeout=MultimodalConfig().vlm_timeout_seconds)
    client._http.http.client.close()
    client._http.http.client = httpx.Client(base_url="http://owned-fixture.invalid/v1",
        transport=httpx.MockTransport(timeout), trust_env=False)
    try:
        with deadline_budget(deadline if deadline is not None else 1000):
            with pytest.raises(VisionRequestError) as caught:
                client.complete(prompt="owned synthetic fixture", image_data_uri="artificial")
        assert caught.value.unknown_outcome and not caught.value.retryable
        assert len(seen) == 1 and 0 < seen[0] <= (deadline or 180)
        if deadline is None:
            assert seen[0] == pytest.approx(180, abs=.01)
        safe = _exception_metadata(caught.value.__cause__)
        assert safe["inference_error_class"] == "ReadTimeout"
        assert "private" not in str(safe)
    finally:
        client.close()
