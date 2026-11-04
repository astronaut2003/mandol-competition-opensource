from dataclasses import replace

import pytest

from mandol_competition.inference.errors import InferenceDeadlineExceeded, InferenceRateLimited
from mandol_competition.inference.http import deadline_budget
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.vlm.base import ExtractionError
from mandol_competition.multimodal.vlm.openai_compat import (
    OpenAICompatExtractor, VisionExtractionUnavailable, VisionRequestError, VisionResponse,
)

CONFIG = MultimodalConfig(vision_failure_policy="retry-required-v1")


class Client:
    def __init__(self, outcomes):
        self.outcomes, self.calls = iter(outcomes), 0

    def complete(self, **_):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return VisionResponse(outcome, "gpt-4o-mini", None, 1.)


def extract(client, config=CONFIG):
    return OpenAICompatExtractor(model_name="gpt-4o-mini", config=config, client=client).extract(
        image_data_uri="artificial", context_text="caption", image_position=0, image_count=1)


@pytest.mark.parametrize("first", [VisionRequestError("busy", retryable=True, status_code=503),
                                  ExtractionError("invalid JSON"), {}])
def test_known_failure_retries_to_success_without_empty_extraction(monkeypatch, first):
    sleeps = []
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.time.sleep", sleeps.append)
    client = Client([first, {"description": "A teal folder marked PLAN."}])
    assert extract(client).description == "A teal folder marked PLAN."
    assert client.calls == 2 and sleeps == [1]


def test_exhausted_validation_returns_dependency_failure(monkeypatch):
    monkeypatch.setattr("mandol_competition.multimodal.vlm.openai_compat.time.sleep", lambda _: None)
    client = Client([{}, {}, {}])
    with pytest.raises(VisionExtractionUnavailable) as caught:
        extract(client)
    assert caught.value.retry_safe and not caught.value.unknown_outcome
    assert client.calls == 3


def test_unknown_paid_outcome_is_not_automatically_reissued():
    client = Client([VisionRequestError("unknown", retryable=True, unknown_outcome=True)])
    with pytest.raises(VisionExtractionUnavailable) as caught:
        extract(client)
    assert caught.value.unknown_outcome and not caught.value.retry_safe and client.calls == 1


@pytest.mark.parametrize("status", [429, 503])
def test_transport_managed_retries_are_not_multiplied(status):
    client = Client([VisionRequestError("exhausted", retryable=False, status_code=status,
                                       retry_managed=True, retry_after=42)])
    with pytest.raises(InferenceRateLimited if status == 429 else VisionExtractionUnavailable) as caught:
        extract(client)
    assert client.calls == 1
    if status == 429:
        assert caught.value.retry_after == 42


def test_expired_budget_dispatches_nothing_and_short_budget_does_not_oversleep():
    client = Client([ExtractionError("invalid")])
    with deadline_budget(0):
        with pytest.raises(InferenceDeadlineExceeded):
            extract(client)
    assert client.calls == 0
    with deadline_budget(.1):
        with pytest.raises(InferenceDeadlineExceeded):
            extract(client)
    assert client.calls == 1


def test_old_empty_cache_is_not_reused_as_success(tmp_path):
    from mandol_competition.multimodal.vlm.cache import CachingExtractor, ExtractionCache
    from mandol_competition.multimodal.vlm.base import Extraction
    class Source:
        calls = 0
        def extract(self, **_):
            self.calls += 1
            return Extraction(description="A striped envelope.", model="fixture", prompt_version="v1", schema_version="1")
    inner = Source()
    cache = ExtractionCache(tmp_path)
    baseline = CachingExtractor(inner, cache, model="fixture", prompt_version="v1", schema_version="1")
    inputs = dict(image_data_uri="data:image/png;base64,QUJD", context_text="photo", image_position=0, image_count=1)
    from mandol_competition.multimodal.vlm.cache import _media_id
    media_id = _media_id(inputs["image_data_uri"])
    key = baseline._key(media_id, "photo", 0, 1)
    cache.put(media_id, key, Extraction(model="fixture", prompt_version="v1", schema_version="1"))
    strict = CachingExtractor(inner, cache, model="fixture", prompt_version="v1", schema_version="1", require_evidence=True)
    assert strict.extract(**inputs).description == "A striped envelope."
    assert strict.extract(**inputs).description == "A striped envelope." and inner.calls == 1
