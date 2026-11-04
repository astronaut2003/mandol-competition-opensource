"""Synthetic outage recovery must be bounded without permanently poisoning Add."""
from dataclasses import asdict

from fastapi.testclient import TestClient
import pytest

from mandol_competition.inference.errors import InferenceRateLimited
from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.service import PreparationLedger
from mandol_competition.multimodal.vlm.openai_compat import VisionExtractionUnavailable

from .backend_support import request
from .test_sqlite_qdrant import image_part
from .test_vision_checkpoint_recovery import (
    Vision, add_service, extraction, ledger, published, seed, strict_config,
)


@pytest.fixture
def recovery_clock(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(PreparationLedger, "_recovery_now", lambda _: clock[0], raising=False)
    return clock


def test_legacy_exhausted_image_recovers_after_persisted_cooldown(backend, recovery_clock):
    body = request(content=[image_part("red"), image_part("blue")])
    complete = {"status": "complete", "extraction": asdict(extraction("A red card."))}
    seed(backend, body, {"0:0": complete, "0:1": {"status": "unknown", "recovery_attempts": 3}})
    vision = Vision()
    for _ in range(2):
        with pytest.raises(InferenceRateLimited) as caught:
            add_service(backend, vision).handle(body)
        assert caught.value.retry_after == 60
        assert vision.calls == 0 and not published(backend)
        saved = ledger(backend, body).value["vision"]
        assert saved["0:0"] == complete
        assert saved["0:1"]["recovery_window_started_at"] == 1000.
    recovery_clock[0] += 61
    assert add_service(backend, vision).handle(body)["success"]
    assert vision.calls == 1 and len(published(backend)) == 1
    assert add_service(backend, vision).handle(body)["success"] and vision.calls == 1


def test_unknown_calls_are_capped_within_window_and_resume_after_outage(backend, recovery_clock):
    body = request(content=[image_part()])
    seed(backend, body, {"0:0": {"status": "unknown"}})
    vision = Vision([VisionExtractionUnavailable(retry_safe=False, unknown_outcome=True) for _ in range(3)])
    for attempt in range(1, 4):
        with pytest.raises(VisionExtractionUnavailable):
            add_service(backend, vision).handle(body)
        assert ledger(backend, body).value["vision"]["0:0"]["recovery_attempts"] == attempt
    with pytest.raises(InferenceRateLimited):
        add_service(backend, vision).handle(body)
    assert vision.calls == 3 and not published(backend)
    recovery_clock[0] += 60
    assert add_service(backend, vision).handle(body)["success"]
    assert vision.calls == 4 and len(published(backend)) == 1


def test_cooldown_returns_http_retry_after_without_dispatch_or_empty_success(backend, recovery_clock):
    body = request(content=[image_part()])
    seed(backend, body, {"0:0": {"status": "started", "recovery_attempts": 3}})
    vision = Vision()
    with TestClient(create_app(strict_config(backend), extractor=vision, backend=backend),
                    raise_server_exceptions=False) as client:
        first = client.post("/add", json=body.model_dump())
        assert first.status_code == 429 and first.headers["Retry-After"] == "60"
        assert vision.calls == 0 and not published(backend)
        recovery_clock[0] += 61
        success = client.post("/add", json=body.model_dump())
        replay = client.post("/add", json=body.model_dump())
        assert success.status_code == replay.status_code == 200
        assert success.json() == replay.json() and vision.calls == 1


@pytest.mark.parametrize("window", [-1, True, "1000", float("nan"), float("inf")])
def test_invalid_window_cannot_authorize_an_unbounded_recovery(backend, recovery_clock, window):
    from mandol_competition.multimodal.vlm.openai_compat import VisionRecoveryExhausted
    body = request(content=[image_part()])
    # Non-finite values cannot be committed through canonical JSON, so seed
    # the object at the ledger boundary rather than corrupting a real DB.
    preparation = ledger(backend, body)
    preparation.value["vision"] = {"0:0": {
        "status": "unknown", "recovery_attempts": 3, "recovery_window_started_at": window}}
    vision = Vision()
    class Cache:
        def get(self, *_):
            return None
    vision._cache = Cache()
    vision._key = lambda *_: "owned-fixture-key"
    with pytest.raises(VisionRecoveryExhausted):
        preparation.extract(key="0:0", vision=vision,
            image_data_uri=image_part()["image_url"]["url"], context_text="", image_position=0, image_count=1)
    assert vision.calls == 0 and not published(backend)


def test_clock_regression_does_not_reset_the_reserved_allowance(backend, recovery_clock):
    body = request(content=[image_part()])
    seed(backend, body, {"0:0": {"status": "unknown", "recovery_attempts": 3,
                                "recovery_window_started_at": 1000.}})
    recovery_clock[0] = 990.
    vision = Vision()
    with pytest.raises(InferenceRateLimited) as caught:
        add_service(backend, vision).handle(body)
    assert caught.value.retry_after == 60 and vision.calls == 0
    assert ledger(backend, body).value["vision"]["0:0"]["recovery_attempts"] == 3


def test_parallel_retries_after_cooldown_publish_once(backend, recovery_clock):
    from concurrent.futures import ThreadPoolExecutor
    body = request(content=[image_part()])
    seed(backend, body, {"0:0": {"status": "unknown", "recovery_attempts": 3,
                                "recovery_window_started_at": 900.}})
    vision = Vision()
    with ThreadPoolExecutor(max_workers=8) as workers:
        responses = list(workers.map(lambda _: add_service(backend, vision).handle(body), range(8)))
    assert all(response["success"] for response in responses)
    assert vision.calls == 1 and len(published(backend)) == 1
