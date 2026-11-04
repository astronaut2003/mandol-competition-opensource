"""Accepted generation-body errors must not reset paid durable Add attempts."""
from dataclasses import replace
import httpx
import pytest

from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.errors import InferenceRateLimited
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient, OpenAICompatExtractor
from mandol_competition.multimodal.service import AddService, PreparationLedger
from .backend_support import request
from .test_sqlite_qdrant import image_part
from .test_vision_checkpoint_recovery import ledger, published


@pytest.mark.parametrize('operational', [False, True])
@pytest.mark.parametrize('selected_provider', ['openrouter', 'closeai'])
def test_accepted_429_cannot_refund_or_publish(backend, monkeypatch, operational, selected_provider):
    import mandol_competition.inference.vlm as module
    import mandol_competition.inference.budget as budget
    from types import SimpleNamespace
    if operational:
        monkeypatch.setattr(budget, 'from_environment', lambda: SimpleNamespace(
            seconds=10, attempts=3, model=lambda *args: None))
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', selected_provider)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-only')
    monkeypatch.setenv('CLOSEAI_API_KEY', 'synthetic-closeai-only')
    monkeypatch.setattr(PreparationLedger, '_recovery_now', lambda _: 1000.)
    seen = []
    def serve(req):
        seen.append(req)
        return httpx.Response(200, headers={'Retry-After': '60'},
            json={'error': {'code': 429, 'message': 'owned private diagnostic'}})
    monkeypatch.setattr(module, 'HTTPTransport', lambda endpoint, **kwargs:
        HTTPTransport(endpoint, transport=httpx.MockTransport(serve), **kwargs))
    config = replace(backend.config, vision_failure_policy='retry-required-v1')
    client = OpenAICompatVisionClient(model='gpt-4o-mini', api_key='unused-old',
        base_url='https://api.openai-proxy.org/v1', timeout=3)
    extractor = OpenAICompatExtractor(model_name='gpt-4o-mini', config=config, client=client)
    add = AddService(config, extractor=extractor, backend=backend)
    body = request(content=[image_part()])
    try:
        for attempt in range(2):
            with pytest.raises(InferenceRateLimited) as caught:
                add.handle(body)
            assert caught.value.vision_paid_output_failure
            saved = ledger(backend, body).value['vision']['0:0']
            assert saved['status'] == 'failed'
            assert saved['recovery_attempts'] == attempt
            assert not published(backend)
        assert len(seen) == 2
    finally:
        client.close()
