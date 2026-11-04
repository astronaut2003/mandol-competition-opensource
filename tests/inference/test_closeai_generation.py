"""CloseAI release activation retains all guards and ignores stale OR routing."""
import asyncio
from dataclasses import fields, replace
import json
from types import SimpleNamespace

import httpx
import pytest

from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable
from mandol_competition.inference.generation import generation_route, CLOSEAI_ENDPOINT
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.vlm import HTTPVLMClient
from .test_openrouter_generation import reply
from .test_openrouter_consumers import mock_shared_wire


@pytest.fixture
def closeai(monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'closeai')
    monkeypatch.setenv('CLOSEAI_API_KEY', 'synthetic-closeai-secret')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'must-never-be-forwarded')
    monkeypatch.setenv('MANDOL_OPENROUTER_PROXY', 'http://invalid-unused-proxy:1')
    monkeypatch.setenv('HTTPS_PROXY', 'http://invalid-unused-global-proxy:1')


def test_selected_provider_overrides_stale_endpoint_model_key_and_proxy(closeai, monkeypatch):
    seen = []
    mock_shared_wire(monkeypatch, lambda request: seen.append(request) or httpx.Response(200,
        json=reply(model='gpt-4o-mini')))
    client = HTTPVLMClient('https://openrouter.ai/api/v1', 'openai/gpt-4o-mini', api_key='old-or-key')
    try:
        client.complete([], request_id=None, max_completion_tokens=16384, response_format={'type': 'json_object'})
        assert client.route.provider == 'closeai' and client.model == 'gpt-4o-mini'
        assert client.http.proxy is None and not client.http.client._trust_env
        assert str(seen[0].url) == CLOSEAI_ENDPOINT + '/chat/completions'
        assert seen[0].headers['Authorization'] == 'Bearer synthetic-closeai-secret'
        body = json.loads(seen[0].content)
        assert body['max_completion_tokens'] == 16384 and body['model'] == 'gpt-4o-mini'
        assert 'provider' not in body and 'transforms' not in body
    finally:
        client.close()


def test_missing_closeai_key_never_uses_openrouter_or_argument(closeai, monkeypatch):
    monkeypatch.delenv('CLOSEAI_API_KEY')
    monkeypatch.delenv('CLOSEAI_API_KEY_FILE', raising=False)
    with pytest.raises(ValueError, match='credential'):
        HTTPVLMClient('https://openrouter.ai/api/v1', 'openai/gpt-4o-mini', api_key='old-or-key')


def test_closeai_secret_mount_and_logical_alias_mapping(closeai, monkeypatch, tmp_path):
    mount = tmp_path / 'owned-key'
    mount.write_text('owned-closeai-key\n')
    monkeypatch.delenv('CLOSEAI_API_KEY')
    monkeypatch.setenv('CLOSEAI_API_KEY_FILE', str(mount))
    route = generation_route(None, 'gpt-4o-mini-closeai')
    assert route.key('old-or-key') == 'owned-closeai-key'
    assert route.model == 'gpt-4o-mini' and route.proxy is None
    assert 'owned-closeai-key' not in repr(route)


@pytest.mark.parametrize('endpoint', ['http://api.openai-proxy.org/v1',
    'https://secret@api.openai-proxy.org/v1', 'https://api.openai-proxy.org/v1?key=secret',
    'https://openrouter.ai/v1', 'https://api.openai-proxy.org:bad/v1'])
def test_wrong_closeai_endpoint_fails_before_dispatch(closeai, monkeypatch, endpoint):
    monkeypatch.setenv('MANDOL_CLOSEAI_BASE_URL', endpoint)
    with pytest.raises(ValueError) as caught:
        generation_route(None, 'gpt-4o-mini')
    assert 'secret' not in str(caught.value)


@pytest.mark.parametrize('options', [{}, {'max_tokens': True}, {'max_tokens': 16385},
    {'max_tokens': 32, 'max_completion_tokens': 64}, {'max_tokens': 32, 'models': ['other']},
    {'max_tokens': 32, 'provider': {'only': ['OpenAI']}}, {'max_tokens': 32, 'transforms': []}])
def test_closeai_requires_bounded_budget_and_rejects_or_only_options(closeai, options):
    route = generation_route(None, 'gpt-4o-mini')
    with pytest.raises(ValueError):
        route.options(options)


@pytest.mark.parametrize('code', [429, '429', 503, None])
def test_closeai_200_errors_keep_status_budget_and_are_not_repeated(closeai, code):
    seen = []
    wire = HTTPTransport(CLOSEAI_ENDPOINT, max_attempts=3, transport=httpx.MockTransport(
        lambda request: seen.append(request) or httpx.Response(200, headers={'Retry-After': '7'},
            json={**reply(model='gpt-4o-mini'), 'error': {'code': code, 'message': 'private fixture'}})))
    try:
        with pytest.raises((InferenceRateLimited, InferenceUnavailable)) as caught:
            wire.request('POST', '/chat/completions', {'model': 'gpt-4o-mini'}, retry_unknown=False)
        assert len(seen) == 1 and caught.value.unknown_outcome
        assert 'private' not in str(caught.value)
        if str(code) == '429':
            assert caught.value.retry_after == 7 and caught.value.vision_paid_output_failure
    finally:
        wire.close()


def test_mm_preserves_baseline_closeai_cache_identity_and_all_nonprovider_fields(tmp_path, monkeypatch):
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.multimodal.service import _cached_extractor
    from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor
    original = MultimodalConfig(vlm_backend='openai_compat', data_dir=tmp_path,
        vlm_model='gpt-4o-mini-closeai', vlm_request_model='gpt-4o-mini',
        vlm_base_url=CLOSEAI_ENDPOINT, vlm_api_key_env='CLOSEAI_API_KEY')
    inner = FakeVisionExtractor({}, model=original.vlm_model, prompt_version='v1', schema_version='1')
    def cache_key(config):
        return _cached_extractor(config, 'owned-user', inner)._key('img_owned', 'caption', 0, 1)
    expected = cache_key(original)
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'closeai')
    changed = replace(original, vlm_base_url='https://openrouter.ai/api/v1',
                      vlm_request_model='openai/gpt-4o-mini', vlm_api_key_env='OPENROUTER_API_KEY')
    assert cache_key(changed) == expected
    assert changed.vlm_api_key_env == 'CLOSEAI_API_KEY' and changed.vlm_base_url == CLOSEAI_ENDPOINT
    for field in fields(original):
        assert getattr(original, field.name) == getattr(changed, field.name)
    assert changed.vlm_output_tokens == changed.vlm_max_output_tokens == 1024


def test_textual_and_mm_atomic_budgets_stay_unchanged_while_mm_vision_is_capped(closeai, monkeypatch):
    from mandol_competition.textual.core.reranker_quality import StrictQualityProvider, fingerprint
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.multimodal.atomic_guard import AtomicRecallGuard
    from mandol_competition.multimodal.vlm.openai_compat import build_vision_client
    seen = []
    def serve(request):
        seen.append(request)
        return httpx.Response(200, json=reply(model='gpt-4o-mini',
            content='{"description":"Owned square"}' if len(seen) == 2 else '{"indices":[0]}'))
    mock_shared_wire(monkeypatch, serve)
    async def textual():
        provider = StrictQualityProvider(transport=httpx.MockTransport(serve))
        try:
            result = await provider.complete([{'role': 'user', 'content': 'owned'}], 128, True)
            assert result['status'] == 'PASS' and result['request_sha256'] == fingerprint(json.loads(seen[0].content))
            assert provider.metadata['provider'] == 'closeai'
        finally:
            await provider.close()
    asyncio.run(textual())
    config = MultimodalConfig(vlm_backend='openai_compat', vlm_model='gpt-4o-mini-closeai',
        vlm_request_model='openai/gpt-4o-mini', vlm_base_url='https://openrouter.ai/api/v1',
        vlm_api_key_env='OPENROUTER_API_KEY')
    vision, guard = build_vision_client(config), AtomicRecallGuard(config)
    try:
        vision.complete(prompt='Owned JSON extraction.', image_data_uri='data:image/png;base64,QUJD')
        for selector in (guard.primary, guard.verifier):
            assert selector.select(query='q', options=[], candidates=[{'text': 'atomic fact'}], limit=1).indices == (0,)
        bodies = [json.loads(r.content) for r in seen]
        assert [b.get('max_completion_tokens', b.get('max_tokens')) for b in bodies] == [128, 1024, 600, 600]
        assert all(b['model'] == 'gpt-4o-mini' and 'provider' not in b and 'transforms' not in b for b in bodies)
        assert all(r.headers['Authorization'] == 'Bearer synthetic-closeai-secret' for r in seen)
    finally:
        vision.close()
        guard.close()


def test_grouped_launcher_explicit_closeai_drops_stale_or_key(closeai, monkeypatch):
    from mandol_competition.config import load_config
    from mandol_competition.runtime.__main__ import child_environment, profile_path
    cfg = load_config(profile_path('local'))
    # A stale configured key source must not be forwarded by the launcher.
    cfg = replace(cfg, runtime=replace(cfg.runtime, vlm_credential_env='OPENROUTER_API_KEY'))
    monkeypatch.setenv('MANDOL_LOCAL_GENERATION_PROVIDER', 'closeai')
    monkeypatch.setenv('MANDOL_LOCAL_CLOSEAI_BASE_URL', CLOSEAI_ENDPOINT)
    env = child_environment(cfg, profile_path('local'))
    assert env['MANDOL_GENERATION_PROVIDER'] == 'closeai'
    assert env['CLOSEAI_API_KEY'] == 'synthetic-closeai-secret'
    assert env['MANDOL_CLOSEAI_BASE_URL'] == CLOSEAI_ENDPOINT
    assert 'OPENROUTER_API_KEY' not in env and 'MANDOL_OPENROUTER_PROXY' not in env


def test_bailian_unchanged_under_closeai_selection(closeai):
    seen = []
    payload = {'model': 'text-embedding-v4', 'input': ['owned text']}
    wire = HTTPTransport('https://dashscope.aliyuncs.com/compatible-mode/v1', api_key='bailian-fixture',
        transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={'data': []})))
    try:
        assert wire.request('POST', '/embeddings', payload) == {'data': []}
        assert wire.proxy is None and json.loads(seen[0].content) == payload
        assert seen[0].headers['Authorization'] == 'Bearer bailian-fixture'
    finally:
        wire.close()
