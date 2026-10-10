import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.vlm.base import ExtractionError, VisionExtractor
from mandol_competition.multimodal.vlm.openai_compat import (
    OpenAICompatExtractor,
    OpenAICompatVisionClient,
    VisionRequestError,
    VisionResponse,
    build_vision_client,
    is_retryable_status,
)

IMAGE = "data:image/png;base64,QUJD"
CONFIG = MultimodalConfig(vlm_model="vision-model", vlm_max_attempts=3)


class StubClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def complete(self, *, prompt, image_data_uri):
        self.calls.append((prompt, image_data_uri))
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return VisionResponse(outcome, "vision-model", {"total_tokens": 12}, 2.5)


def extract(extractor, *, context_text="ctx", image_position=0, image_count=1):
    return extractor.extract(
        image_data_uri=IMAGE, context_text=context_text,
        image_position=image_position, image_count=image_count,
    )


def test_concurrent_extractions_keep_usage_with_their_own_image():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    class Client:
        def complete(self, *, prompt, image_data_uri):
            return VisionResponse({'description': 'visible'}, 'vision-model',
                                  {'total_tokens': int(image_data_uri)}, 1.)
    extractor = OpenAICompatExtractor(model_name='vision-model', config=CONFIG, client=Client())
    def request(index):
        extractor.extract(image_data_uri=str(index), context_text='', image_position=0, image_count=1)
        barrier.wait(timeout=5)
        result = extractor.take_observations()
        assert extractor.take_observations() == []
        return result
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(request, (1, 2)))
    assert results == [[('1', {'total_tokens': 1}, 1.)], [('2', {'total_tokens': 2}, 1.)]]


def sdk_client(content, *, usage=None, error=None, captured=None):
    def complete_request(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=usage,
        )

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete_request)))
    return OpenAICompatVisionClient(
        model="vision-model", api_key="unused", base_url="https://example.invalid/v1",
        timeout=4.0, client_factory=lambda: sdk,
    )


@pytest.mark.parametrize(
    ("status", "expected"),
    [(None, True), (408, True), (425, True), (429, True), (500, True),
     (502, True), (503, True), (504, True), (400, False), (401, False),
     (403, False), (404, False), (422, False)],
)
def test_retryable_status(status, expected):
    assert is_retryable_status(status) is expected


def test_client_sends_ordered_vision_blocks_and_parses_usage():
    captured = {}
    usage = SimpleNamespace(prompt_tokens=9, completion_tokens=3, total_tokens=12)
    client = sdk_client('{"description": "cat"}', usage=usage, captured=captured)
    response = client.complete(prompt="describe", image_data_uri=IMAGE)

    assert captured == {
        "model": "vision-model", "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": IMAGE}},
        ]}],
    }
    assert response.payload == {"description": "cat"}
    assert response.model == "vision-model"
    assert response.usage == {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}
    assert response.latency_ms >= 0


def test_actual_sdk_serializes_to_configured_endpoint_without_network():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "mock-response", "object": "chat.completion", "created": 1,
            "model": "vision-model", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"description": "visible"}'},
            }],
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as transport:
        client = OpenAICompatVisionClient(
            model="vision-model", api_key="unused", base_url="https://domestic.example.invalid/v1",
            timeout=4.0,
            client_factory=lambda: OpenAI(
                api_key="unused", base_url="https://domestic.example.invalid/v1",
                http_client=transport, max_retries=0,
            ),
        )
        assert client.complete(prompt="describe", image_data_uri=IMAGE).payload == {
            "description": "visible",
        }

    assert len(requests) == 1
    assert str(requests[0].url) == "https://domestic.example.invalid/v1/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer unused"
    blocks = json.loads(requests[0].content)["messages"][0]["content"]
    assert blocks == [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": IMAGE}},
    ]


@pytest.mark.parametrize("content", [
    '```json\n{"description": "fenced"}\n```',
    'Sure. {"description": "fenced"} Done.',
    'Here: {"description": "nested } brace", "objects": [{"name": "bag"}]} end',
])
def test_client_accepts_json_surrounded_by_markdown_or_prose(content):
    assert sdk_client(content).complete(prompt="p", image_data_uri=IMAGE).payload["description"] in {
        "fenced", "nested } brace",
    }


@pytest.mark.parametrize("content", [None, "no JSON", '{"description": ', '[{"description": "x"}]'])
def test_client_rejects_missing_or_non_object_json(content):
    with pytest.raises(ExtractionError):
        sdk_client(content).complete(prompt="p", image_data_uri=IMAGE)


@pytest.mark.parametrize(("status", "retryable"), [(None, True), (429, True), (503, True), (401, False)])
def test_client_normalizes_transport_errors(status, retryable):
    error = RuntimeError("transport failed")
    if status is not None:
        error.status_code = status
    with pytest.raises(VisionRequestError) as caught:
        sdk_client(None, error=error).complete(prompt="p", image_data_uri=IMAGE)
    assert caught.value.status_code == status
    assert caught.value.retryable is retryable
    assert caught.value.__cause__ is error


def test_sdk_retries_are_disabled(monkeypatch):
    captured = {}

    def factory(endpoint, model, **kwargs):
        captured.update(kwargs, base_url=endpoint, model=model)
        return object()

    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.HTTPVLMClient", factory)
    OpenAICompatVisionClient(
        model="any-model", api_key="any-key", base_url="https://example.invalid/v1", timeout=4.0,
    )
    assert captured == {
        "api_key": "any-key", "base_url": "https://example.invalid/v1",
        "timeout": 4.0, "model": "any-model",
    }


def test_build_client_uses_arbitrary_key_environment_and_endpoint(monkeypatch):
    captured = {}
    monkeypatch.setenv("MY_VISION_SERVICE_TOKEN", "secret")
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.HTTPVLMClient", lambda endpoint, model, **kwargs: captured.update(kwargs, base_url=endpoint, model=model))
    config = dataclasses.replace(
        CONFIG, vlm_base_url="https://domestic.example.invalid/v1",
        vlm_api_key_env="MY_VISION_SERVICE_TOKEN",
    )
    client = build_vision_client(config)
    assert client.model == "vision-model"
    assert captured["api_key"] == "secret"
    assert captured["base_url"] == config.vlm_base_url
    assert captured["model"] == config.vlm_model


def test_build_client_requires_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("MY_VISION_SERVICE_TOKEN", raising=False)
    with pytest.raises(ExtractionError, match="base_url"):
        build_vision_client(CONFIG)
    config = dataclasses.replace(
        CONFIG, vlm_base_url="https://domestic.example.invalid/v1",
        vlm_api_key_env="MY_VISION_SERVICE_TOKEN",
    )
    with pytest.raises(ExtractionError, match="MY_VISION_SERVICE_TOKEN"):
        build_vision_client(config)


def test_extractor_preserves_visual_structure_prompt_and_metadata():
    payload = {
        "description": "  a red bag ", "entities": [{"name": "Grace", "type": "person"}],
        "objects": [{"name": "bag", "count": 1}], "ocr_text": "OPEN",
        "events": [{"subject": "Grace", "action": "holds", "object": "bag"}],
        "relations": [{"source": "Grace", "predicate": "holds", "target": "bag"}],
        "tags": ["red"], "location_hint": "kitchen",
    }
    client = StubClient([payload])
    extractor = OpenAICompatExtractor(model_name="vision-model", config=CONFIG, client=client)
    assert isinstance(extractor, VisionExtractor)
    result = extract(extractor, context_text="Grace holds a bag", image_position=1, image_count=2)
    assert result.description == "a red bag"
    assert result.entities[0].name == "Grace"
    assert result.objects[0].name == "bag"
    assert result.ocr_text == "OPEN"
    assert result.events[0].action == "holds"
    assert result.relations[0].target == "bag"
    assert result.tags == ["red"]
    assert result.location_hint == "kitchen"
    assert (result.model, result.prompt_version, result.schema_version) == (
        "vision-model", CONFIG.prompt_version, CONFIG.schema_version,
    )
    assert "Grace holds a bag" in client.calls[0][0]
    assert "second of 2" in client.calls[0][0]
    assert client.calls[0][1] == IMAGE
    assert extractor.take_observations() == [(IMAGE, {"total_tokens": 12}, 2.5)]
    assert extractor.take_observations() == []


def test_extractor_retries_transient_requests_and_invalid_structured_payload(monkeypatch):
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.time.sleep", lambda _delay: None)
    client = StubClient([
        VisionRequestError("busy", retryable=True, status_code=429),
        {"description": 42}, {"description": "ok"},
    ])
    extractor = OpenAICompatExtractor(model_name="vision-model", config=CONFIG, client=client)
    assert extract(extractor).description == "ok"
    assert len(client.calls) == 3
    assert extractor.take_observations() == [(IMAGE, {"total_tokens": 12}, 2.5)]


def test_extractor_does_not_retry_permanent_error(monkeypatch):
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.time.sleep", lambda _delay: pytest.fail("slept"))
    client = StubClient([VisionRequestError("bad key", retryable=False, status_code=401)])
    with pytest.raises(ExtractionError, match="VisionRequestError"):
        extract(OpenAICompatExtractor(model_name="vision-model", config=CONFIG, client=client))
    assert len(client.calls) == 1


def test_extractor_stops_after_configured_attempt_budget(monkeypatch):
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.time.sleep", lambda _delay: None)
    client = StubClient([VisionRequestError("busy", retryable=True)] * 5)
    config = dataclasses.replace(CONFIG, vlm_max_attempts=2)
    with pytest.raises(ExtractionError, match="after 2 attempt"):
        extract(OpenAICompatExtractor(model_name="vision-model", config=config, client=client))
    assert len(client.calls) == 2


def test_extractor_honors_provider_retry_after_and_refuses_unbounded_wait(monkeypatch):
    waits = []
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep', waits.append)
    client = StubClient([VisionRequestError('busy', retryable=True, status_code=429, retry_after=5.),
                         {'description': 'ok'}])
    assert extract(OpenAICompatExtractor(model_name='vision-model', config=CONFIG, client=client)).description == 'ok'
    assert waits == [5.]
    for delay in (61., float('inf')):
        client = StubClient([VisionRequestError('busy', retryable=True, status_code=429, retry_after=delay)])
        with pytest.raises(ExtractionError):
            extract(OpenAICompatExtractor(model_name='vision-model', config=CONFIG, client=client))
        assert len(client.calls) == 1
    assert waits == [5.]
