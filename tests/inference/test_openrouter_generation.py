import json

import httpx
import pytest

from mandol_competition.inference.errors import InferenceIdentityError, InferenceRateLimited
from mandol_competition.inference.evidence_selector import EvidenceSelector
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.generation import generation_route, is_mini_response


def reply(model='openai/gpt-4o-mini', content='{"indices":[0]}'):
    return {'model': model, 'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 7, 'total_tokens': 17}}


def test_unified_route_maps_model_key_budget_and_proxy(monkeypatch):
    import mandol_competition.inference.vlm as module
    seen = []
    created = []
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-openrouter-secret')
    def factory(endpoint, **kwargs):
        created.append((endpoint, kwargs.copy()))
        return HTTPTransport(endpoint, transport=httpx.MockTransport(
            lambda request: seen.append(request) or httpx.Response(200, json=reply())), **kwargs)
    monkeypatch.setattr(module, 'HTTPTransport', factory)
    client = HTTPVLMClient('https://api.openai-proxy.org/v1', 'gpt-4o-mini',
                           api_key='synthetic-old-closeai-secret')
    try:
        assert client.complete([{'role': 'user', 'content': 'synthetic'}], request_id=None,
                               max_completion_tokens=64)['model'] == 'openai/gpt-4o-mini'
        payload = json.loads(seen[0].content)
        assert created[0][0] == 'https://openrouter.ai/api/v1'
        assert created[0][1]['proxy'] == 'http://127.0.0.1:17897'
        assert seen[0].url.path == '/api/v1/chat/completions'
        assert seen[0].headers['Authorization'] == 'Bearer synthetic-openrouter-secret'
        assert payload['model'] == 'openai/gpt-4o-mini'
        assert payload['max_tokens'] == 64 and 'max_completion_tokens' not in payload
        assert payload['provider']['require_parameters'] is True
        assert 'synthetic-old-closeai-secret' not in str(seen[0].headers)
    finally:
        client.close()


def test_openrouter_prefix_does_not_disable_mini_identity_check():
    class Wire:
        def request(self, *args, **kwargs):
            return reply(model='openai/gpt-4.1')
    client = HTTPVLMClient('https://openrouter.ai/api/v1', 'openai/gpt-4o-mini', transport=Wire())
    with pytest.raises(InferenceIdentityError):
        client.complete([], request_id=None, max_tokens=32)


def test_selector_accepts_namespaced_same_mini_without_changing_candidates():
    class Client:
        def complete(self, *args, **kwargs):
            return reply()
    selector = EvidenceSelector('unused', client=Client(), strict=True)
    result = selector.select(query='q', options=[], candidates=[{'text': 'original fact'}], limit=1)
    assert result.indices == (0,) and result.status == 'selected'


def test_http_200_embedded_429_preserves_retry_after_and_redaction():
    wire = HTTPTransport('https://openrouter.ai/api/v1', max_attempts=1,
        transport=httpx.MockTransport(lambda _: httpx.Response(200,
            headers={'Retry-After': '11'}, json={'error': {'code': 429, 'message': 'private source echo'}})))
    try:
        with pytest.raises(InferenceRateLimited) as error:
            wire.request('POST', '/chat/completions', {'model': 'openai/gpt-4o-mini'}, retry_unknown=False)
        assert error.value.retry_after == 11
        assert 'private' not in str(error.value)
    finally:
        wire.close()


@pytest.mark.parametrize('model', ['gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
    'openai/gpt-4o-mini', 'openai/gpt-4o-mini-2024-07-18'])
def test_exact_same_model_aliases(model):
    assert is_mini_response(model)


@pytest.mark.parametrize('model', ['gpt-4o', 'openai/gpt-4.1', 'gpt-4o-mini-evil',
                                  'other/gpt-4o-mini', None, {}, []])
def test_other_models_cannot_pass_identity_validation(model):
    assert not is_mini_response(model)


def test_missing_new_key_never_forwards_old_key(monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.delenv('OPENROUTER_API_KEY_FILE', raising=False)
    with pytest.raises(ValueError, match='credential'):
        HTTPVLMClient('https://api.openai-proxy.org/v1', 'gpt-4o-mini', api_key='old-secret')


def test_secret_file_support_and_no_key_in_route_metadata(tmp_path, monkeypatch):
    path = tmp_path / 'key'
    path.write_text('synthetic-new-secret\n')
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY_FILE', str(path))
    route = generation_route('https://api.openai-proxy.org/v1', 'gpt-4o-mini')
    assert route.key('old-secret') == 'synthetic-new-secret'
    assert 'synthetic-new-secret' not in repr(route)


@pytest.mark.parametrize('proxy', ['', 'https://127.0.0.1:17897', 'http://public.invalid:80',
    'http://secret@127.0.0.1:17897', 'http://127.0.0.1:abc', 'http://127.0.0.1:17897/?token=secret'])
def test_proxy_must_be_explicit_credential_free_loopback(proxy, monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('MANDOL_OPENROUTER_PROXY', proxy)
    with pytest.raises(ValueError) as error:
        generation_route(None, 'gpt-4o-mini')
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('endpoint', ['http://openrouter.ai/api/v1',
    'https://openrouter.ai/api/v1?key=secret', 'https://openrouter.ai:invalid/api/v1'])
def test_noncanonical_destination_is_refused(endpoint):
    with pytest.raises(ValueError) as error:
        generation_route(endpoint, 'gpt-4o-mini')
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('options', [{}, {'max_tokens': True}, {'max_tokens': 0},
    {'max_tokens': 16385}, {'max_tokens': 32, 'max_completion_tokens': 64},
    {'max_tokens': 32, 'models': ['other']}, {'max_tokens': 32, 'stream': True}])
def test_unbounded_or_fallback_requests_never_dispatch(options):
    class Wire:
        def request(self, *args, **kwargs):
            pytest.fail('invalid output budget dispatched')
    client = HTTPVLMClient('https://openrouter.ai/api/v1', 'gpt-4o-mini', transport=Wire())
    with pytest.raises(ValueError):
        client.complete([], request_id=None, **options)


@pytest.mark.parametrize('code', [400, 429, '429', 503, None, 'private-message'])
def test_accepted_body_errors_are_never_repeated_or_accepted_as_partial_success(code):
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(200, headers={'Retry-After': '9'},
            json={**reply(), 'error': {'code': code, 'message': 'private echo'}})
    wire = HTTPTransport('https://openrouter.ai/api/v1', max_attempts=3,
                          transport=httpx.MockTransport(serve))
    try:
        with pytest.raises((InferenceUnavailable, InferenceRateLimited)) as caught:
            wire.request('POST', '/chat/completions', {}, retry_unknown=False)
        assert len(calls) == 1 and caught.value.unknown_outcome
        assert 'private' not in str(caught.value)
        if str(code) == '429':
            assert caught.value.retry_after == 9
            assert caught.value.vision_paid_output_failure
    finally:
        wire.close()


def test_bailian_transport_and_body_are_unchanged_by_generation_flag(monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('HTTPS_PROXY', 'http://must-not-be-used.invalid:1')
    payload = {'model': 'text-embedding-v4', 'input': ['owned text']}
    seen = []
    wire = HTTPTransport('https://dashscope.aliyuncs.com/compatible-mode/v1', api_key='bailian-fixture',
        transport=httpx.MockTransport(lambda request: seen.append(request) or httpx.Response(200, json={'data': []})))
    try:
        assert wire.request('POST', '/embeddings', payload) == {'data': []}
        assert wire.proxy is None and json.loads(seen[0].content) == payload
        assert seen[0].headers['Authorization'] == 'Bearer bailian-fixture'
    finally:
        wire.close()


def test_proxy_connect_failure_has_no_direct_fallback_or_paid_outcome(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    seen = []
    class Proxy(BaseHTTPRequestHandler):
        def do_CONNECT(self):
            seen.append((self.path, self.headers.get('Authorization')))
            self.send_response(502)
            self.end_headers()
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'synthetic-new-key')
    monkeypatch.setenv('MANDOL_OPENROUTER_PROXY', f'http://127.0.0.1:{server.server_port}')
    client = HTTPVLMClient('https://api.openai-proxy.org/v1', 'gpt-4o-mini', timeout=2)
    try:
        with pytest.raises(InferenceUnavailable) as caught:
            client.complete([], request_id=None, max_tokens=32)
        assert caught.value.error_class == 'ProxyError'
        assert not caught.value.unknown_outcome
        assert seen == [('openrouter.ai:443', None)]
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
