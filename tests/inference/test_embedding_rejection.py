"""A permanent provider input error must not become a retryable outage."""
import json

from fastapi.testclient import TestClient
import httpx
import pytest

from mandol_competition.inference.dashscope import MODEL, official_embedding_recipes
from mandol_competition.inference.errors import InferenceInputRejected, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.runtime.http_boundary import ErrorPolicy
from mandol_competition.observability.diagnostics import _exception_metadata
from tests.inference.test_provider_recovery import Provider, bridge


@pytest.mark.parametrize('status', [400, 413, 422])
def test_bridge_keeps_permanent_status_and_does_not_poison_provider_health(monkeypatch, status):
    module = bridge(monkeypatch)
    class Rejected(Provider):
        def encode(self, texts, *, query=False):
            if self.fail:
                self.calls.append(query)
                raise InferenceUnavailable('http_' + str(status), status_code=status,
                    provider_error_code='InvalidParameter', provider_request_id='synthetic-provider-id')
            return super().encode(texts, query=query)
    provider = Rejected()
    dense, sparse = official_embedding_recipes()
    with TestClient(module.create_app(lambda: provider, recovery_delay=.01)) as client:
        provider.fail = True
        response = client.post('/v1/encode', json={'model': MODEL, 'input': ['synthetic question'],
            'input_type': 'query', 'dense_recipe_hash': dense.hash, 'sparse_recipe_hash': sparse.hash})
        assert response.status_code == status
        assert response.headers['X-Mandol-Input-Rejected'] == '1'
        assert response.headers['X-Mandol-Upstream-Code'] == 'InvalidParameter'
        assert client.get('/health').status_code == 200
        state = client.get('/ops/state').json()['provider_recovery']
        assert state['failures'] == state['recovery_probes'] == 0
        assert provider.calls == [False, True, True]


@pytest.mark.parametrize('status', [400, 413, 422])
def test_internal_encoding_transport_and_public_boundary_keep_nonretryable_status(status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={'X-Mandol-Input-Rejected': '1',
            'X-Mandol-Upstream-Code': 'InvalidParameter'}, json={'detail': 'PRIVATE CONTENT'})
    wire = HTTPTransport('http://127.0.0.1:18620', max_attempts=3, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(InferenceInputRejected) as raised:
            wire.request('POST', '/v1/encode', {'input': ['synthetic']}, retry_unknown=False)
        assert raised.value.status_code == ErrorPolicy().status(raised.value) == status
        assert raised.value.provider_error_code == 'InvalidParameter' and len(calls) == 1
        assert 'PRIVATE' not in str(raised.value)
    finally:
        wire.close()


def test_external_header_cannot_select_internal_input_error_mapping():
    wire = HTTPTransport('https://provider.invalid', transport=httpx.MockTransport(lambda _: httpx.Response(
        400, headers={'X-Mandol-Input-Rejected': '1'}, json={})))
    try:
        with pytest.raises(InferenceUnavailable) as raised:
            wire.request('POST', '/v1/encode', {})
        assert type(raised.value) is InferenceUnavailable and ErrorPolicy().status(raised.value) == 503
    finally:
        wire.close()


@pytest.mark.parametrize('status', [400, 413, 422])
@pytest.mark.parametrize('track', ['textual', 'code', 'multimodal'])
def test_public_http_boundary_exposes_permanent_input_error_without_body(status, track):
    from fastapi import FastAPI
    from mandol_competition.runtime.http_boundary import install_http_boundary
    wire = HTTPTransport('http://127.0.0.1:18620', max_attempts=3,
        transport=httpx.MockTransport(lambda _: httpx.Response(status,
            headers={'X-Mandol-Input-Rejected': '1'}, json={'message': 'PRIVATE PROVIDER BODY'})))
    app = FastAPI()
    @app.post('/search')
    def search():
        return wire.request('POST', '/v1/encode', {'input': ['synthetic']})
    install_http_boundary(app, track, ready=lambda: True)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post('/search')
        assert response.status_code == status and 'PRIVATE' not in response.text
        assert 'data' not in response.json() and 'Retry-After' not in response.headers
    finally:
        wire.close()


def test_failure_metadata_never_contains_messages_input_or_unsafe_identifiers(monkeypatch):
    module = bridge(monkeypatch)
    error = InferenceUnavailable('PRIVATE input body', status_code=400,
        provider_error_code='InvalidParameter', provider_request_id='unsafe id PRIVATE')
    metadata = module._failure_metadata(error, 'query')
    assert metadata['upstream_http_status'] == 400 and metadata['provider_error_code'] == 'InvalidParameter'
    assert 'PRIVATE' not in json.dumps(metadata)
    metadata = _exception_metadata(error)
    assert 'error_class' not in metadata and 'PRIVATE' not in json.dumps(metadata)


def test_transient_503_is_still_retryable_service_failure(monkeypatch):
    module = bridge(monkeypatch)
    provider = Provider()
    dense, sparse = official_embedding_recipes()
    with TestClient(module.create_app(lambda: provider)) as client:
        provider.fail = True
        response = client.post('/v1/encode', json={'model': MODEL, 'input': ['synthetic'],
            'input_type': 'query', 'dense_recipe_hash': dense.hash, 'sparse_recipe_hash': sparse.hash})
        assert response.status_code == 503 and 'X-Mandol-Input-Rejected' not in response.headers
        assert client.get('/ops/state').json()['provider_recovery']['failures'] == 1
