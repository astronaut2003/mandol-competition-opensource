"""All generation callers use artificial responses in an isolated network."""
import asyncio
from dataclasses import fields, replace
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.generation import generation_route
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.textual.core.reranker_quality import StrictQualityProvider, fingerprint
from mandol_competition.textual.core.paired_quality import AuditedQualityProvider
from .test_openrouter_generation import reply


@pytest.fixture
def routed(monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-openrouter-only')
    monkeypatch.delenv('CLOSEAI_API_KEY', raising=False)


def mock_shared_wire(monkeypatch, handler):
    import mandol_competition.inference.vlm as module
    monkeypatch.setattr(module, 'HTTPTransport', lambda endpoint, **kwargs:
        HTTPTransport(endpoint, transport=httpx.MockTransport(handler), **kwargs))


@pytest.mark.parametrize('cls', [StrictQualityProvider, AuditedQualityProvider])
def test_textual_provider_uses_new_key_and_exact_wire_request_hash(routed, cls):
    seen = []
    def serve(request):
        seen.append(request)
        return httpx.Response(200, json=reply(content='{"facts":[]}'))
    async def run():
        provider = cls(transport=httpx.MockTransport(serve))
        try:
            assert provider.metadata['provider'] == 'openrouter'
            assert provider.metadata['actual_model'] == 'gpt-4o-mini'
            assert provider.metadata['credential_env'] == 'OPENROUTER_API_KEY'
            return await provider.complete([{'role': 'user', 'content': 'owned source'}], 128, True)
        finally:
            await provider.close()
    result = asyncio.run(run())
    assert result['status'] == 'PASS' and len(seen) == 1
    body = json.loads(seen[0].content)
    assert body['model'] == 'openai/gpt-4o-mini' and body['max_tokens'] == 128
    assert result['request_sha256'] == fingerprint(body)
    assert seen[0].headers['Authorization'] == 'Bearer synthetic-openrouter-only'


def test_mm_config_cache_identity_changes_only_for_actual_provider(tmp_path, monkeypatch):
    from mandol_competition.multimodal.service import _cached_extractor
    from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor
    legacy = MultimodalConfig(vlm_backend='openai_compat', data_dir=tmp_path,
        vlm_model='gpt-4o-mini-closeai', vlm_request_model='gpt-4o-mini',
        vlm_base_url='https://api.openai-proxy.org/v1', vlm_api_key_env='CLOSEAI_API_KEY')
    inner = FakeVisionExtractor({}, model=legacy.vlm_model, prompt_version='v1', schema_version='1')
    def key(config):
        return _cached_extractor(config, 'artificial-user', inner)._key('img_owned', 'caption', 0, 1)
    old_key = key(legacy)
    old_records = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file())
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    updated = replace(legacy)
    assert updated.vlm_base_url == 'https://openrouter.ai/api/v1'
    assert updated.vlm_request_model == 'openai/gpt-4o-mini'
    assert updated.vlm_api_key_env == 'OPENROUTER_API_KEY'
    for field in fields(legacy):
        if field.name not in {'vlm_base_url', 'vlm_request_model', 'vlm_api_key_env'}:
            assert getattr(legacy, field.name) == getattr(updated, field.name)
    assert key(updated) != old_key
    monkeypatch.setenv('OPENROUTER_API_KEY', 'rotated-artificial-key')
    monkeypatch.setenv('MANDOL_OPENROUTER_PROXY', 'http://127.0.0.1:12345')
    assert key(replace(updated)) == key(updated)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file()) == old_records


def test_mm_vision_and_both_evidence_passes_route_without_old_key(routed, monkeypatch):
    from mandol_competition.multimodal.vlm.openai_compat import build_vision_client
    from mandol_competition.multimodal.atomic_guard import AtomicRecallGuard
    seen = []
    def serve(request):
        seen.append(request)
        return httpx.Response(200, json=reply(content=(
            '{"description":"Owned red square"}' if len(seen) == 1 else '{"indices":[0]}')))
    mock_shared_wire(monkeypatch, serve)
    config = MultimodalConfig(vlm_backend='openai_compat', vlm_model='gpt-4o-mini-closeai',
        vlm_request_model='gpt-4o-mini', vlm_base_url='https://api.openai-proxy.org/v1',
        vlm_api_key_env='CLOSEAI_API_KEY')
    vision = build_vision_client(config)
    guard = AtomicRecallGuard(config)
    selectors = [guard.primary, guard.verifier]
    try:
        result = vision.complete(prompt='Describe this artificial image as JSON.',
                                 image_data_uri='data:image/png;base64,QUJD')
        assert result.payload['description'] == 'Owned red square'
        assert result.model == 'gpt-4o-mini-closeai'
        for selector in selectors:
            selected = selector.select(query='owned query', options=[],
                candidates=[{'text': 'owned atomic evidence'}], limit=1)
            assert selected.indices == (0,)
        bodies = [json.loads(request.content) for request in seen]
        assert [body['max_tokens'] for body in bodies] == [1024, 600, 600]
        assert all(body['model'] == 'openai/gpt-4o-mini' for body in bodies)
        assert all(request.headers['Authorization'] == 'Bearer synthetic-openrouter-only' for request in seen)
    finally:
        vision.close()
        for selector in selectors:
            selector.client.close()


def test_optional_textual_selector_startup_does_not_require_old_key(routed, monkeypatch):
    from mandol_competition.textual.api.competition import _generation_selector
    seen = []
    mock_shared_wire(monkeypatch, lambda request: seen.append(request) or
        httpx.Response(200, json=reply()))
    selector = _generation_selector(SimpleNamespace(selector_endpoint='https://api.openai-proxy.org/v1',
                                                    selector_api_key_env='CLOSEAI_API_KEY'))
    try:
        result = selector.select(query='q', options=[], candidates=[{'text': 'atomic fact'}], limit=1)
        assert result.indices == (0,)
        assert seen[0].headers['Authorization'] == 'Bearer synthetic-openrouter-only'
    finally:
        selector.close()


@pytest.mark.parametrize('second_finish',['stop','length'])
def test_vision_output_expands_once_with_same_source_and_bounded_openrouter_payload(routed, monkeypatch, second_finish):
    from mandol_competition.multimodal.vlm.openai_compat import build_vision_client, OpenAICompatExtractor, VisionExtractionUnavailable
    seen = []
    def serve(request):
        seen.append(json.loads(request.content))
        value = reply(content='{"description":"Owned dense document"}')
        if len(seen) == 1:
            value['choices'][0]['finish_reason'] = 'length'
            value['usage'] = {'prompt_tokens': 20, 'completion_tokens': 512, 'total_tokens': 532}
        elif second_finish == 'length':
            value['choices'][0]['finish_reason'] = 'length'
            value['usage'] = {'prompt_tokens': 20, 'completion_tokens': 1024, 'total_tokens': 1044}
        return httpx.Response(200, json=value)
    mock_shared_wire(monkeypatch, serve)
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep', lambda _: None)
    config = MultimodalConfig(vlm_backend='openai_compat', vlm_model='gpt-4o-mini-closeai',
        vlm_request_model='gpt-4o-mini', vlm_base_url='https://api.openai-proxy.org/v1',
        vlm_api_key_env='CLOSEAI_API_KEY', vision_failure_policy='retry-required-v1',
        vlm_output_tokens=512, vlm_max_output_tokens=1024)
    vision = build_vision_client(config)
    try:
        extractor = OpenAICompatExtractor(model_name=config.vlm_model, config=config, client=vision)
        inputs = dict(image_data_uri='data:image/png;base64,QUJD', context_text='owned caption', image_position=0, image_count=1)
        if second_finish == 'length':
            with pytest.raises(VisionExtractionUnavailable) as caught:
                extractor.extract(**inputs)
            assert caught.value.error_class == 'vision_output_truncated' and not caught.value.retry_safe
        else:
            assert extractor.extract(**inputs).description == 'Owned dense document'
        assert [body['max_tokens'] for body in seen] == [512, 1024]
        assert seen[0]['messages'] == seen[1]['messages']
        parts = next(message['content'] for message in seen[0]['messages'] if message['role']=='user')
        assert parts[1] == {'type':'image_url','image_url':{'url':inputs['image_data_uri']}}
        assert inputs['context_text'] in parts[0]['text']
        assert all(body['provider']['require_parameters'] is True for body in seen)
    finally:
        vision.close()


def test_downstream_textual_receipts_accept_namespace_without_allowing_other_models():
    from mandol_competition.textual.core.user_profile import checked_response
    from mandol_competition.textual.core.heterogeneous_memory import terminal_length_rejection
    from mandol_substrate.state import StateUnavailable
    record = {'status': 'PASS', 'finish_reason': 'stop',
              'response_model': 'openai/gpt-4o-mini', 'content': '{"facts":[]}'}
    assert checked_response(record) == record['content']
    assert terminal_length_rejection(dict(record, status='FAIL', finish_reason='length'))
    with pytest.raises(StateUnavailable):
        checked_response(dict(record, response_model='openai/gpt-4.1'))
    assert not terminal_length_rejection(dict(record, response_model='openai/gpt-4.1',
                                             status='FAIL', finish_reason='length'))


def test_code_experience_new_key_and_semantic_recipe_preserved(routed, monkeypatch):
    from mandol_competition.code.experience import ExperienceBuilder, FINGERPRINT
    seen = []
    mock_shared_wire(monkeypatch, lambda request: seen.append(request) or httpx.Response(200,
        json=reply(content='{"owned":"episode"}')))
    builder = ExperienceBuilder.__new__(ExperienceBuilder)
    builder.http = None
    try:
        value, receipt = builder._generate({'instruction': 'Owned extraction', 'text': 'owned'}, 'owned-request')
        assert value == {'owned': 'episode'}
        assert receipt['recipe_fingerprint'] == FINGERPRINT
        assert receipt['logical_model'] == 'gpt-4o-mini-closeai'
        assert receipt['transport_provider'] == 'openrouter'
        assert receipt['request_model'] == 'openai/gpt-4o-mini'
        assert json.loads(seen[0].content)['max_tokens'] == 1800
    finally:
        builder.http.close()


@pytest.mark.parametrize('selected_provider', ['openrouter', 'closeai'])
def test_offline_coding_facade_uses_selected_transport_and_preserves_tool_arguments(routed, monkeypatch, selected_provider):
    from mandol_competition.code.software.api_models import CodingAPIClient
    from mandol_competition.code.software.api_context import MaximumSafeBudget
    seen, closed = [], []
    wire_model = 'openai/gpt-4o-mini' if selected_provider == 'openrouter' else 'gpt-4o-mini'
    if selected_provider == 'closeai':
        monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'closeai')
        monkeypatch.setenv('CLOSEAI_API_KEY', 'synthetic-closeai-only')
    mock_shared_wire(monkeypatch, lambda request: seen.append(request) or httpx.Response(200,
        json=reply(model=wire_model, content='{"owned":"reply"}')))
    class Tokenizer:
        name = 'o200k_base'
        def encode(self, text, **kwargs):
            return list(text.encode())
    class LLM:
        def __init__(self, **kwargs):
            self.model_name = 'gpt-4o-mini-closeai'
            self.actual_model = 'gpt-4o-mini'
            self.provider_name = 'openai-proxy'
            self.context_length = 128000
            self.max_output_tokens = 16384
            self.max_context_tokens = round(128000 * kwargs['max_context_ratio'])
            self.request_timeout = kwargs['request_timeout']
            self.base_url = kwargs['base_url']
            self.tokenizer = Tokenizer()
            self.client = SimpleNamespace(close=lambda: closed.append(True))
    # Optional sibling package supplies accounting only; no SDK network calls.
    for name in ('mandol', 'mandol.llm', 'mandol.llm.llm_client', 'mandol.llm.configs'):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules['mandol.llm.llm_client'].LLMClient = LLM
    sys.modules['mandol.llm.configs'].API_PROVIDERS = {'openai-proxy': {'base_url': 'https://legacy.invalid/v1'}}
    import importlib.metadata
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name:
        'synthetic-fixture' if name in {'openai', 'tiktoken'} else real_version(name))
    client = CodingAPIClient.from_env(budget=MaximumSafeBudget(max_tokens=128))
    try:
        messages = [{'role': 'system', 'content': 'owned policy'}, {'role': 'user', 'content': 'owned task'}]
        tool = {'type': 'function', 'function': {'name': 'owned', 'parameters': {'type': 'object'}}}
        request, audit = client.prepare(messages, task_text='owned task', tools=[tool])
        result = client.complete(request, audit)
        assert closed == [True] and result['response']['model'] == wire_model
        assert json.loads(seen[0].content) == request
        assert request['messages'] == messages and request['tools'] == [tool]
        assert request['parallel_tool_calls'] is False
        assert audit['provider'] == selected_provider and audit['model_alias'] == 'gpt-4o-mini-closeai'
        if selected_provider == 'closeai':
            assert 'provider' not in request and 'transforms' not in request
            assert seen[0].headers['Authorization'] == 'Bearer synthetic-closeai-only'
    finally:
        client.llm.client.close()


def test_runtime_multimodal_startup_checks_new_credential_source(routed, monkeypatch, tmp_path):
    import mandol_competition.runtime.apps as apps
    import mandol_competition.inference.deployment as serving
    cfg = SimpleNamespace(memory_system_key='owned-system-key', allow_unauthenticated_dev=False,
        runtime=SimpleNamespace(mode='local', vlm_credential_env='CLOSEAI_API_KEY'),
        inference={'vlm': 'https://api.openai-proxy.org/v1'}, root=tmp_path)
    monkeypatch.setattr(apps, 'load_config', lambda: SimpleNamespace(
        log_level='ERROR', track=lambda _: cfg))
    monkeypatch.setattr(apps, 'claim_runtime_root', lambda *args: None)
    monkeypatch.setattr(serving, 'claim_serving_identity', lambda *args: None)
    assert apps._settings('multimodal') is cfg


def test_grouped_launcher_requires_scoped_opt_in_before_forwarding_new_key(routed, monkeypatch):
    from mandol_competition.config import load_config
    from mandol_competition.runtime.__main__ import child_environment, profile_path
    cfg = load_config(profile_path('local'))
    baseline = child_environment(cfg, profile_path('local'))
    assert 'OPENROUTER_API_KEY' not in baseline
    assert 'MANDOL_GENERATION_PROVIDER' not in baseline
    monkeypatch.setenv('MANDOL_LOCAL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('MANDOL_LOCAL_OPENROUTER_PROXY', 'http://127.0.0.1:12345')
    updated = child_environment(cfg, profile_path('local'))
    assert updated['MANDOL_GENERATION_PROVIDER'] == 'openrouter'
    assert updated['MANDOL_OPENROUTER_PROXY'] == 'http://127.0.0.1:12345'
    assert updated['OPENROUTER_API_KEY'] == 'synthetic-openrouter-only'
    assert cfg.track('textual').inference == load_config(profile_path('local')).track('textual').inference


def test_grouped_launcher_preserves_secret_mount_source_and_rejects_unknown_policy(monkeypatch, tmp_path):
    from mandol_competition.config import load_config
    from mandol_competition.runtime.__main__ import child_environment, profile_path
    cfg = load_config(profile_path('local'))
    monkeypatch.setenv('MANDOL_LOCAL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    mount = tmp_path / 'secret-mount'
    mount.write_text('artificial')
    monkeypatch.setenv('OPENROUTER_API_KEY_FILE', str(mount))
    updated = child_environment(cfg, profile_path('local'))
    assert updated['OPENROUTER_API_KEY_FILE'] == str(mount)
    assert 'OPENROUTER_API_KEY' not in updated
    monkeypatch.setenv('MANDOL_LOCAL_GENERATION_PROVIDER', 'invalid')
    with pytest.raises(ValueError, match='mode-scoped generation'):
        child_environment(cfg, profile_path('local'))


@pytest.mark.parametrize('unknown,delay,calls_expected,sleeps_expected', [
    (True, 11, 1, []), (False, 11, 3, [11, 11]), (False, 90, 1, [])])
def test_offline_coding_retry_preserves_unknown_outcomes_and_provider_cooldown(
        monkeypatch, unknown, delay, calls_expected, sleeps_expected):
    import mandol_competition.code.software.api_models as module
    from mandol_competition.code.software.api_context import MaximumSafeBudget, digest
    from mandol_competition.inference.errors import InferenceUnavailable
    calls, sleeps = [], []
    def fail(**request):
        calls.append(request)
        raise InferenceUnavailable('owned-failure', status_code=408,
                                   unknown_outcome=unknown, retry_after=delay)
    client = module.CodingAPIClient.__new__(module.CodingAPIClient)
    client.budget, client.retries, client.secret = MaximumSafeBudget(max_tokens=128), 2, ''
    client.llm = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=fail))))
    monkeypatch.setattr(module.time, 'sleep', sleeps.append)
    request = {'model': 'gpt-4o-mini', 'max_tokens': 128, 'messages': []}
    with pytest.raises(module.CodingAPIError):
        client.complete(request, {'request_sha256': digest(request)})
    assert len(calls) == calls_expected and sleeps == sleeps_expected
