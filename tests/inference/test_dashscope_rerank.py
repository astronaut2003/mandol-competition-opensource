from mandol_competition.inference.errors import InferenceInputRejected
"""Native provider response mapping, lifecycle and failure boundaries."""
import json

import httpx
import pytest

from mandol_competition.inference.dashscope_rerank import (
    DashScopeRerankerClient, ENDPOINT, MODEL, PATH, scores_in_input_order,
)
from mandol_competition.inference.errors import InferenceIdentityError, InferenceSchemaError, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport


def client(handler, **options):
    transport = HTTPTransport('https://unit.test', api_key='test-only', max_attempts=2,
        transport=httpx.MockTransport(handler), sleep=lambda _: None)
    return DashScopeRerankerClient(transport=transport, **options)


def test_sorted_provider_results_are_restored_to_input_order_without_changing_text():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'output': {'results': [
            {'index': 1, 'relevance_score': .9}, {'index': 0, 'relevance_score': .1}]},
            'request_id': 'test-id', 'usage': {'total_tokens': 42}})
    with_client = client(handler)
    try:
        assert with_client.rerank(' original query ', [' raw\n code ', 'second']) == [.1, .9]
        payload = json.loads(requests[0].content)
        assert payload == {'model': MODEL, 'input': {'query': ' original query ',
            'documents': [' raw\n code ', 'second']}, 'parameters': {'top_n': 2, 'return_documents': False}}
        assert requests[0].url.path == PATH
        assert requests[0].headers['Authorization'] == 'Bearer test-only'
        assert with_client.last_response_metadata['usage']['total_tokens'] == 42
        assert with_client.attestation()['response_model'] is None
        assert with_client.attestation()['dtype_verified'] is False
        assert 'test-only' not in str(with_client.attestation())
    finally:
        with_client.close()


@pytest.mark.parametrize('rows', [[], [{'index': 0, 'relevance_score': .1}],
    [{'index': 0, 'relevance_score': .1}] * 2,
    [{'index': 2, 'relevance_score': .1}, {'index': 0, 'relevance_score': .9}],
    [{'index': True, 'relevance_score': .1}, {'index': 0, 'relevance_score': .9}],
    [{'index': 0, 'relevance_score': float('nan')}, {'index': 1, 'relevance_score': .9}],
    [{'index': 0, 'relevance_score': -1}, {'index': 1, 'relevance_score': .9}],
    [{'index': 0, 'relevance_score': True}, {'index': 1, 'relevance_score': .9}],
    [{'index': 0, 'relevance_score': '0.1'}, {'index': 1, 'relevance_score': .9}],
])
def test_partial_duplicate_invalid_or_nonfinite_scores_fail_closed(rows):
    with pytest.raises(InferenceSchemaError):
        scores_in_input_order({'output': {'results': rows}}, 2)


def test_identity_mismatch_and_provider_error_cannot_masquerade_as_valid_results():
    with pytest.raises(InferenceIdentityError):
        scores_in_input_order({'model': 'other-model'}, 1)
    with pytest.raises(InferenceSchemaError):
        scores_in_input_order({'code': 'InvalidApiKey', 'message': 'secret echoed'}, 1)


def test_health_probes_are_bounded_cached_and_invalidated_after_scoring_failure():
    calls, now = [], [0.]
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={'output': {'results': [
            {'index': 0, 'relevance_score': .9}, {'index': 1, 'relevance_score': .1}]}})
    c = client(handler, clock=lambda: now[0])
    try:
        c.validate_profile()
        for _ in range(20):
            c.health()
        assert len(calls) == 1
        now[0] = 61
        c.health()
        assert len(calls) == 2
        # A malformed one-document score invalidates the successful observation.
        with pytest.raises(InferenceSchemaError):
            c.score('query', ['one'])
        c.health()
        assert len(calls) == 4
    finally:
        c.close()


def test_unknown_paid_outcome_is_not_replayed_and_no_secret_or_body_leaks():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(503, text='private raw document test-only secret')
    c = client(handler)
    try:
        with pytest.raises(InferenceUnavailable) as failure:
            c.score('query', ['document'])
        assert len(calls) == 1
        assert 'private' not in str(failure.value)
        assert 'test-only' not in str(failure.value)
    finally:
        c.close()


def test_no_independent_batch_score_mixing_or_silent_document_truncation():
    c = client(lambda _: pytest.fail('no HTTP dispatch expected'))
    try:
        assert c.score('query', []) == []
        with pytest.raises(InferenceInputRejected):
            c.score('query', ['document'] * 501)
    finally:
        c.close()


def test_credentials_required_and_redirects_not_followed():
    with pytest.raises(ValueError, match='credential'):
        DashScopeRerankerClient(ENDPOINT)
    c = client(lambda _: httpx.Response(302, headers={'Location': 'https://other.test'}))
    try:
        with pytest.raises(InferenceUnavailable, match='http_302'):
            c.score('query', ['document'])
    finally:
        c.close()
