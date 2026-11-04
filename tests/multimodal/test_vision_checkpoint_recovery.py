"""Artificial checkpoint recovery through the real SQLite/Qdrant Add path."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import threading

import pytest
from fastapi.testclient import TestClient

from mandol_substrate.state import Conflict
from mandol_substrate.state.models import digest
from mandol_competition.inference.errors import InferenceDeadlineExceeded, InferenceRateLimited
from mandol_competition.inference.http import deadline_budget
from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.service import AddService, PreparationLedger, _cached_extractor
from mandol_competition.multimodal.vlm.base import Extraction
from mandol_competition.multimodal.vlm.cache import _media_id
from mandol_competition.multimodal.vlm.openai_compat import VisionExtractionUnavailable

from .backend_support import request
from .test_sqlite_qdrant import image_part


@pytest.fixture(autouse=True)
def fixed_recovery_clock(monkeypatch):
    monkeypatch.setattr(PreparationLedger, '_recovery_now', lambda _: 1000.)


class Vision:
    model_name = 'fixture-vision'

    def __init__(self, outcomes=()):
        self.calls = 0
        self.outcomes = iter(outcomes)

    def extract(self, **_):
        self.calls += 1
        outcome = next(self.outcomes, None)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome or extraction('A blue card marked RECOVERY.')


def extraction(description):
    return Extraction(description=description, model=Vision.model_name,
                      prompt_version='multimodal_sgm_v1', schema_version='1')


def strict_config(backend):
    return replace(backend.config, vision_failure_policy='retry-required-v1')


def add_service(backend, vision):
    return AddService(strict_config(backend), extractor=vision, backend=backend)


def ledger(backend, body):
    return PreparationLedger(backend, body, digest(body.model_dump()), strict=True)


def seed(backend, body, entries):
    prepared = ledger(backend, body)
    prepared.value['vision'] = entries
    prepared.save()


def published(backend, user='alice'):
    return list(backend.state.iter_memories(backend.scope(user)))


@pytest.mark.parametrize('status', ['unknown', 'started', 'failed', 'degraded'])
def test_legacy_checkpoint_recovers_without_repeating_completed_image(backend, status):
    body = request(content=[image_part('red'), image_part('blue')])
    seed(backend, body, {'0:0': {'status': 'complete', 'extraction': asdict(extraction('A red card.'))},
                         '0:1': {'status': status}})
    vision = Vision()
    add = add_service(backend, vision)
    assert add.handle(body)['success']
    assert vision.calls == 1
    units = published(backend)
    assert len(units) == 1 and not units[0][0].metadata.get('vlm_degraded', False)
    assert add.handle(body)['success'] and vision.calls == 1
    assert len(published(backend)) == 1
    with pytest.raises(Conflict):
        add.handle(request(content='A changed payload for the same request ID.'))
    assert vision.calls == 1


@pytest.mark.parametrize('status', ['unknown', 'started', 'failed', 'degraded'])
@pytest.mark.parametrize('attempts', [0, 3])
def test_valid_scoped_cache_wins_even_when_recovery_allowance_is_exhausted(backend, status, attempts):
    part = image_part()
    body = request(content=[part])
    seed(backend, body, {'0:0': {'status': status, 'recovery_attempts': attempts}})
    vision = Vision()
    cached = _cached_extractor(strict_config(backend), body.user_id, vision)
    media_id = _media_id(part['image_url']['url'])
    cache_key = cached._key(media_id, '', 0, 1)
    cached._cache.put(media_id, cache_key, extraction('A cached red card.'))
    assert add_service(backend, vision).handle(body)['success']
    assert vision.calls == 0


def test_empty_cache_is_replaced_with_real_evidence_during_recovery(backend):
    part = image_part()
    body = request(content=[part])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    vision = Vision()
    cached = _cached_extractor(strict_config(backend), body.user_id, vision)
    media_id = _media_id(part['image_url']['url'])
    cache_key = cached._key(media_id, '', 0, 1)
    cached._cache.put(media_id, cache_key, extraction(''))
    assert add_service(backend, vision).handle(body)['success']
    assert vision.calls == 1 and len(published(backend)) == 1


def test_recovery_allowance_is_durable_across_new_handlers_and_unknown_outcomes(backend):
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    vision = Vision([VisionExtractionUnavailable(retry_safe=False, unknown_outcome=True) for _ in range(3)])
    for attempt in range(1, 4):
        with pytest.raises(VisionExtractionUnavailable):
            add_service(backend, vision).handle(body)
        checkpoint = ledger(backend, body).value['vision']['0:0']
        assert checkpoint == {'status': 'unknown', 'recovery_attempts': attempt,
                              'recovery_window_started_at': 1000.}
        assert vision.calls == attempt and not published(backend)
    with pytest.raises(InferenceRateLimited):
        add_service(backend, vision).handle(body)
    assert vision.calls == 3 and not published(backend)


def test_process_loss_consumes_durable_recovery_allowance_before_dispatch(backend):
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    interrupted = Vision([SystemExit('artificial process interruption')])
    with pytest.raises(SystemExit):
        add_service(backend, interrupted).handle(body)
    assert ledger(backend, body).value['vision']['0:0'] == {
        'status': 'started', 'recovery_attempts': 1, 'recovery_window_started_at': 1000.}
    recovered = Vision()
    assert add_service(backend, recovered).handle(body)['success']
    assert recovered.calls == 1 and len(published(backend)) == 1


@pytest.mark.parametrize('failure', [InferenceRateLimited(retry_after=60), InferenceDeadlineExceeded()])
def test_known_pre_dispatch_rejection_restores_prior_allowance(backend, failure):
    body = request(content=[image_part()])
    original = {'status': 'unknown', 'recovery_attempts': 2}
    seed(backend, body, {'0:0': original})
    vision = Vision([failure])
    with pytest.raises(type(failure)):
        add_service(backend, vision).handle(body)
    assert ledger(backend, body).value['vision']['0:0'] == original
    assert not published(backend)
    assert add_service(backend, vision).handle(body)['success'] and vision.calls == 2


def test_expired_request_does_not_change_a_legacy_checkpoint_or_call_model(backend):
    body = request(content=[image_part()])
    original = {'status': 'unknown'}
    seed(backend, body, {'0:0': original})
    vision = Vision()
    with deadline_budget(0):
        with pytest.raises(InferenceDeadlineExceeded):
            add_service(backend, vision).handle(body)
    assert vision.calls == 0
    assert ledger(backend, body).value['vision']['0:0'] == original


def test_known_failed_recovery_preserves_its_attempt_count(backend):
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    vision = Vision([VisionExtractionUnavailable(retry_safe=True)])
    with pytest.raises(VisionExtractionUnavailable):
        add_service(backend, vision).handle(body)
    assert ledger(backend, body).value['vision']['0:0'] == {
        'status': 'failed', 'recovery_attempts': 1, 'recovery_window_started_at': 1000.}
    assert add_service(backend, vision).handle(body)['success'] and vision.calls == 2


@pytest.mark.parametrize('attempts', [-1, True, '0', 4])
def test_invalid_or_exhausted_counter_cannot_reset_recovery_budget(backend, attempts):
    from mandol_competition.multimodal.vlm.openai_compat import VisionRecoveryExhausted
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown', 'recovery_attempts': attempts}})
    vision = Vision()
    with pytest.raises(VisionRecoveryExhausted):
        add_service(backend, vision).handle(body)
    assert vision.calls == 0 and not published(backend)


def test_parallel_retries_share_one_recovered_preparation_and_publication(backend):
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    entered, finish = threading.Event(), threading.Event()

    class Slow(Vision):
        def extract(self, **inputs):
            entered.set()
            assert finish.wait(5)
            return super().extract(**inputs)

    vision = Slow()
    add = add_service(backend, vision)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(add.handle, body)
        assert entered.wait(5)
        second = workers.submit(add.handle, body)
        finish.set()
        assert first.result(timeout=10)['success']
        assert second.result(timeout=10)['success']
    assert vision.calls == 1 and len(published(backend)) == 1


def test_http_recovery_returns_a_durable_idempotent_receipt(backend):
    body = request(content=[image_part()])
    seed(backend, body, {'0:0': {'status': 'unknown'}})
    vision = Vision()
    app = create_app(strict_config(backend), extractor=vision, backend=backend)
    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.post('/add', json=body.model_dump())
        replay = client.post('/add', json=body.model_dump())
        assert first.status_code == replay.status_code == 200
        assert first.json() == replay.json() == dict(success=True, request_id=body.request_id,
                                                    user_id=body.user_id, session_id=body.session_id)
    assert vision.calls == 1
