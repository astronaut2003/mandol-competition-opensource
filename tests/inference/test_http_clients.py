from dataclasses import asdict
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPRerankerClient
from mandol_competition.inference.errors import InferenceIdentityError, InferenceSchemaError, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.models import InferenceRecipe


@pytest.fixture
def recipe():
    return InferenceRecipe("test-embedding-v1", "embedding", "alias", "model", "revision", "tokenizer", dimension=3)


def transport(handler, **kwargs):
    return HTTPTransport("http://localhost", transport=httpx.MockTransport(handler), sleep=lambda _: None, **kwargs)


@pytest.mark.parametrize("payload,error", [
    ({"model": "wrong", "data": []}, InferenceIdentityError),
    ({"model": "alias", "data": []}, InferenceSchemaError),
    ({"model": "alias", "data": [{"index": 0, "embedding": [1., 2.]}]}, InferenceSchemaError),
    ({"model": "alias", "data": [{"index": 0, "embedding": [True, 2., 3.]}]}, InferenceSchemaError),
    ({"model": "alias", "data": [{"index": 0, "embedding": [0., 0., 0.]}]}, InferenceSchemaError),
])
def test_embedding_schema_rejects_invalid_results(recipe, payload, error):
    client = HTTPEmbeddingClient("unused", recipe, transport=transport(lambda _: httpx.Response(200, json=payload)))
    with pytest.raises(error):
        client.embed(["text"])
    client.close()


def test_nan_is_not_accepted(recipe):
    client = HTTPEmbeddingClient("unused", recipe, transport=transport(lambda _: httpx.Response(200,
        content=b'{"model":"alias","data":[{"index":0,"embedding":[NaN,1,2]}]}')))
    with pytest.raises(InferenceSchemaError):
        client.embed(["text"])
    client.close()


@pytest.mark.parametrize("status", [429, 503])
def test_retry_reuses_correlation_and_is_bounded(status):
    seen = []
    def handler(request):
        seen.append(request.headers["Idempotency-Key"])
        return httpx.Response(status, headers={'Retry-After': '0'}, json={"private": "must never escape"})
    client = transport(handler, max_attempts=3)
    with pytest.raises(InferenceUnavailable) as error:
        client.request("POST", "/v1/embeddings", {"input": ["private"]})
    assert len(seen) == 3 and len(set(seen)) == 1
    assert "private" not in str(error.value)
    client.close()


def test_unknown_timeout_is_not_retried_when_provider_has_no_idempotency():
    seen = []
    def handler(request):
        seen.append(request)
        raise httpx.ReadTimeout("secret upstream response", request=request)
    client = transport(handler)
    with pytest.raises(InferenceUnavailable) as error:
        client.request("POST", "/chat/completions", {}, retry_unknown=False)
    assert error.value.unknown_outcome and len(seen) == 1
    assert "secret" not in str(error.value)
    client.close()


def test_recipe_attestation_is_explicit_and_rejects_drift(recipe):
    client = HTTPEmbeddingClient("unused", recipe, transport=transport(lambda _: httpx.Response(200,
        json={"recipes": [{"recipe": asdict(recipe), "hash": recipe.hash}]})))
    assert client.validate_profile()
    assert client.attestation()["level"] == "server-declared-recipe"
    client.close()
    client = HTTPEmbeddingClient("unused", recipe, transport=transport(lambda _: httpx.Response(200, json={"recipes": []})))
    with pytest.raises(InferenceIdentityError):
        client.validate_profile()
    client.close()


def test_real_http_transport_pool_and_partial_json():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(self.headers["X-Request-ID"])
            body = b'{"ok":true}' if len(calls) == 1 else b'{partial'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HTTPTransport(f"http://127.0.0.1:{server.server_port}")
    try:
        assert client.request("POST", "/call", {}) == {"ok": True}
        with pytest.raises(InferenceSchemaError):
            client.request("POST", "/call", {})
        assert len(calls) == 2
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("url", ["http://user:key@host", "http://host?key=secret", "file:///tmp/model"])
def test_credential_free_endpoint(url):
    with pytest.raises(ValueError):
        HTTPTransport(url)

@pytest.mark.parametrize('field,value', [('sparse', {'-1': 1}), ('sparse', {'1': -1}), ('sparse', {'1': True}), ('score', 'not-numeric')])
def test_sparse_reranker_validation(recipe, field, value):
    from dataclasses import replace
    r = replace(recipe, channel='sparse' if field == 'sparse' else 'reranker')
    wire = transport(lambda _: httpx.Response(200, json={'model': r.served_alias, 'data': [{'index': 0, field: value}]}))
    client = (HTTPSparseEncoderClient if field == 'sparse' else HTTPRerankerClient)('unused', r, transport=wire)
    with pytest.raises(InferenceSchemaError):
        client.encode_documents(['x']) if field == 'sparse' else client.score('query', ['x'])
    client.close()


def test_response_bytes_bound_and_deadline():
    import time
    class Slow(httpx.SyncByteStream):
        def __iter__(self):
            time.sleep(.025)
            yield b'{"ok":true}'
    wire = transport(lambda _: httpx.Response(200, stream=Slow()), timeout=.01)
    with pytest.raises(InferenceUnavailable, match='deadline'):
        wire.request('POST', '/x', {})
    wire.close()
    wire = transport(lambda _: httpx.Response(200, content=b'0123456789'), max_response_bytes=4)
    with pytest.raises(InferenceSchemaError, match='response_too_large'):
        wire.request('GET', '/x')
    wire.close()
