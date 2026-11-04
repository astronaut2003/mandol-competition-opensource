"""Synthetic Search boundary, HTTP contract and privacy regression tests."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import logging
import math
import re
import time

from fastapi.testclient import TestClient
import pytest

from mandol_competition.textual.api.app import create_app
from mandol_competition.textual.api.config import Settings
from mandol_competition.observability.diagnostics import diagnostic_call
from mandol_substrate.state import Conflict, Mutation, Scope, StateUnavailable
from tests.common.test_publication import coordinator


@pytest.mark.parametrize('statistics', ['scope-v1', 'candidate-subset-v1'])
@pytest.mark.parametrize('term_count', [512, 513, 700, 20000])
def test_full_query_exact_bm25_and_user_isolation(store, profile, scope, memory, tmp_path, statistics, term_count):
    backend = store.lexical
    backend.statistics_policy = statistics
    c, _ = coordinator(store, profile, tmp_path)
    texts = ['earlyword earlyword earlyword', 'lateword lateword', 'other words']
    docs = [replace(memory, logical_uid='synthetic-' + str(i), text_content=text,
                    raw_data={'text_content': text}) for i, text in enumerate(texts)]
    c.apply(scope, 'r', 'r', Mutation(memories=tuple(docs)), {})
    foreign = Scope(scope.namespace, 'synthetic-foreign-user')
    c.apply(foreign, 'r', 'r', Mutation(memories=(replace(docs[0], text_content='lateword ' * 30),)), {})
    # The only matching late term is beyond the former 512-term boundary.
    words = ['earlyword'] + ['syntheticterm%05d' % i for i in range(term_count - 2)] + ['lateword'] * 4
    query = ' '.join(words)
    candidates = [x.logical_uid for x in docs] if statistics == 'candidate-subset-v1' else None
    hits = backend.search(scope, query, limit=100, candidate_uids=candidates)
    normalized = [backend.processor.document(text).split() for text in texts]
    average = sum(map(len, normalized)) / len(texts)
    expected = {}
    for doc, tokens in zip(docs, normalized, strict=True):
        counts = Counter(tokens)
        score = 0.0
        for term, qtf in Counter(words).items():
            tf = counts[term]
            df = sum(term in other for other in normalized)
            if tf:
                score += qtf * math.log(1 + (len(texts) - df + .5) / (df + .5)) * (
                    tf * (backend.k1 + 1) / (tf + backend.k1 * (1 - backend.b + backend.b * len(tokens) / average)))
        if score > backend.minimum_score:
            expected[doc.logical_uid] = score
    assert [h.logical_uid for h in hits] == sorted(expected, key=lambda uid: (-expected[uid], uid))
    assert [h.score for h in hits] == pytest.approx([expected[h.logical_uid] for h in hits], rel=1e-13)
    assert all(math.isfinite(h.score) for h in hits)
    if term_count > 512:
        sql, params = backend._query(scope, query, 100, candidate_uids=candidates)
        assert 'json_each(?)' in sql and len(params) < 20
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert all(result == hits for result in pool.map(
            lambda _: backend.search(scope, query, limit=100, candidate_uids=candidates), range(16)))


class Memory:
    ready = False
    result = None
    failure = None

    def start(self): self.ready = True
    def shutdown(self): self.ready = False
    def health(self): return True
    def purge_expired(self, *args, **kwargs): return 0
    def add_messages(self, **kwargs): return len(kwargs['messages'])

    def search(self, **kwargs):
        if self.failure is not None:
            def boundary(): raise self.failure
            return diagnostic_call('embedding', boundary)
        if self.result is not None:
            return self.result[:kwargs['top_k']]
        return [dict(id='synthetic-id', content='Synthetic memory', score=.5)] if kwargs['user_id'] == 'synthetic-user' else []


@pytest.fixture
def api(tmp_path):
    memory = Memory()
    app = create_app(Settings(competition_run_root=str(tmp_path), evaluation_namespace='synthetic-test',
        memory_system_key='synthetic-regression-key'), memory_service=memory)
    logger = logging.getLogger('mandol_competition.textual.diagnostics')
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            client.headers.update({'X-Api-Key': 'synthetic-regression-key', 'X-Request-ID': 'untrusted-marker'})
            until = time.monotonic() + 5
            while client.get('/health').status_code != 200:
                assert time.monotonic() < until
                time.sleep(.01)
            yield client, memory, stream
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize('options', [None, [], ['Artificial choice one', 'Artificial choice two']])
@pytest.mark.parametrize('top_k', [1, 100])
@pytest.mark.parametrize('user', ['synthetic-user', 'synthetic-empty'])
def test_http_200_contract(api, options, top_k, user):
    client, _, _ = api
    body = dict(user_id=user, query='Artificial query', top_k=top_k)
    if options is not None: body['options'] = options
    response = client.post('/search', json=body)
    assert response.status_code == 200
    payload = json.loads(response.text, parse_constant=lambda value: pytest.fail('nonfinite JSON number'))
    assert set(payload) == {'data'} and isinstance(payload['data'], list)
    assert len(payload['data']) <= top_k
    assert bool(payload['data']) == (user == 'synthetic-user')
    assert all(isinstance(row['id'], str) and row['id'] and isinstance(row['content'], str) and row['content']
               and math.isfinite(row['score']) for row in payload['data'])


@pytest.mark.parametrize('failure,status', [(ValueError, 500), (RuntimeError, 500), (StateUnavailable, 503), (Conflict, 409)])
def test_http_internal_failure_is_not_client_422_and_is_private(api, failure, status):
    client, memory, stream = api
    marker = 'DO-NOT-LOG-API-KEY-QUERY-ANSWER'
    memory.failure = failure(marker)
    response = client.post('/search', json=dict(user_id='synthetic-user', query=marker, top_k=100))
    assert response.status_code == status
    assert marker not in response.text + stream.getvalue()
    if status != 409:
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row['exception_type'] == failure.__name__ and row['stage'] == 'embedding'
        assert re.fullmatch('[0-9a-f]{32}', row['correlation_id'])
        assert response.headers['X-Request-ID'] == row['correlation_id']
        assert row['stack'][-1]['function'] == 'boundary'
        assert all(set(frame) == {'file', 'function', 'line'} for frame in row['stack'])
    assert 'untrusted-marker' not in stream.getvalue()


@pytest.mark.parametrize('patch,error_type', [({'top_k': 0}, 'greater_than_equal'),
    ({'top_k': '100'}, 'int_type'), ({'query': '  '}, 'value_error'),
    ({'options': {'secret-field': 'PRIVATE-MARKER'}}, 'list_type')])
def test_http_client_validation_422_metadata_only(api, patch, error_type):
    client, _, stream = api
    response = client.post('/search', json=dict(user_id='synthetic-user', query='PRIVATE-MARKER', top_k=100) | patch)
    assert response.status_code == 422 and response.json()['detail']['reason'] == 'malformed request'
    row = json.loads(stream.getvalue().splitlines()[-1])
    assert 'stack' not in row and row['stage'] == 'request_validation'
    assert row['validation'][0]['type'] == error_type
    assert all(set(item) == {'loc', 'type'} for item in row['validation'])
    assert 'PRIVATE-MARKER' not in response.text + stream.getvalue()
    assert 'secret-field' not in stream.getvalue()


@pytest.mark.parametrize('result', [
    [dict(id='id', content='secret-answer', score=float('nan'))],
    [dict(id='id', content='secret-answer', score=float('inf'))],
    [dict(id='', content='secret-answer', score=.1)],
    [dict(id='id', content='', score=.1)],
    [dict(id='id', content=None, score=.1)],
])
def test_http_invalid_internal_result_is_safe_500(api, result):
    client, memory, stream = api
    memory.result = result
    response = client.post('/search', json=dict(user_id='synthetic-user', query='private-query', top_k=100))
    assert response.status_code == 500
    assert 'secret-answer' not in response.text + stream.getvalue()
    assert 'private-query' not in response.text + stream.getvalue()
    row = json.loads(stream.getvalue().splitlines()[-1])
    assert row['stage'] == 'response'
    if row['exception_type'] == 'ValidationError':
        assert 'stack' not in row and row['validation']


def test_add_existing_size_bound_remains_client_validation(api):
    client, _, _ = api
    response = client.post('/add', json=dict(user_id='synthetic-user', request_id='r', session_id='s',
        messages=[dict(role='user', content='Synthetic message')] * 1025))
    assert response.status_code == 422
