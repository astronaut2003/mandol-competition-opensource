"""Opt-in contract checks against real deployed APIs and inference dependencies.

MANDOL_AML_REAL=1, MANDOL_AML_KEY_FILE and the three MANDOL_AML_<TRACK>_URL
origins are required. Use a disposable deployment authorized for test writes.
No tests delete data, log secrets/content, or consume official Smoke quota.
"""
import base64
import io
import json
import math
import os
from pathlib import Path
import uuid

import httpx
from PIL import Image
import pytest

pytestmark = pytest.mark.skipif(os.getenv('MANDOL_AML_REAL') != '1', reason='explicit real deployment opt-in required')


class Secret(str):
    """Prevent pytest fixture/assertion repr from exposing a real credential."""
    def __repr__(self):
        return '<redacted credential>'
    def __radd__(self, other):
        return Secret(str(other) + str(self))


@pytest.fixture(params=['textual', 'code', 'multimodal'])
def api(request):
    track = request.param
    origin = os.environ['MANDOL_AML_' + track.upper() + '_URL']
    key = Secret(Path(os.environ['MANDOL_AML_KEY_FILE']).read_text().strip())
    with httpx.Client(base_url=origin, timeout=1800, trust_env=False, follow_redirects=False) as client:
        yield track, client, key


def validate_search(response, top_k):
    assert response.status_code == 200, 'Search HTTP status: ' + str(response.status_code)
    value = response.json()
    assert isinstance(value, dict) and isinstance(value.get('data'), list)
    assert len(value['data']) <= top_k
    for row in value['data']:
        assert isinstance(row['id'], str) and row['id'] and row['content']
        if 'score' in row:
            assert type(row['score']) in (float, int) and math.isfinite(row['score'])
    return value['data']


def test_real_health_auth_sync_visibility_replay_conflict_and_isolation(api):
    track, client, key = api
    health = client.get('/health')
    assert 200 <= health.status_code < 300
    assert health.json() == {'status': 'ok'}, 'health must not reveal internal configuration'
    suffix = uuid.uuid4().hex
    body = {'request_id': 'contract-' + suffix, 'user_id': 'contract-' + suffix,
        'session_id': 'session-1', 'messages': [{'role': 'user', 'content':
        'The qualification database uses SQLite WAL.\n北京部署使用事务日志。'},
        {'role': 'assistant', 'content': 'def recover(request_id): return ledger.replay(request_id)'}]}
    search = {'query': 'Which database mode is used?', 'user_id': body['user_id'], 'top_k': 100,
              'options': ['SQLite WAL', 'other']}
    assert client.post('/add', json=body).status_code == 401
    assert client.post('/search', json=search, headers={'Authorization': 'Bearer invalid'}).status_code == 401
    for headers in ({'Authorization': 'Token ' + key}, {'Authorization': 'Bearer ' + key}, {'X-Api-Key': key}):
        response = client.post('/add', json=body, headers=headers)
        assert response.status_code == 200, 'Add HTTP status: ' + str(response.status_code)
        assert response.json() == {'success': True, **{k: body[k] for k in ('request_id', 'user_id', 'session_id')}}
    headers = {'Authorization': 'Bearer ' + key}
    hits = validate_search(client.post('/search', json=search, headers=headers), 100)
    assert hits, 'synchronous Add must be immediately searchable'
    original = [(h['id'], h.get('score')) for h in hits]
    assert client.post('/add', json=body | {'session_id': 'conflicting-session'}, headers=headers).status_code == 409
    replayed = validate_search(client.post('/search', json=search, headers=headers), 100)
    assert [h['id'] for h in replayed] == [h[0] for h in original]
    assert len({h['id'] for h in replayed}) == len(replayed)
    assert len(validate_search(client.post('/search', json=search | {'top_k': 1}, headers=headers), 1)) == 1
    empty = client.post('/search', json=search | {'user_id': 'unknown-' + suffix}, headers=headers)
    assert empty.status_code == 200 and empty.json() == {'data': []}
    for field in ('request_id', 'messages', 'user_id', 'session_id'):
        malformed = {k: v for k, v in body.items() if k != field}
        invalid = client.post('/add', json=malformed, headers=headers)
        assert invalid.status_code in (400, 422)
        assert body['messages'][0]['content'] not in invalid.text and key not in invalid.text
    assert client.post('/search', json=search | {'top_k': 0}, headers=headers).status_code in (400, 422)
    if track == 'multimodal':
        part_search = search | {'query': [{'type': 'text', 'text': search['query']}]}
        parts = validate_search(client.post('/search', json=part_search, headers=headers), 100)
        assert [h['id'] for h in parts] == [h['id'] for h in hits]


@pytest.mark.parametrize('format,mime', [('JPEG', 'image/jpeg'), ('PNG', 'image/png'), ('WEBP', 'image/webp')])
def test_real_mm_inline_images_and_order(format, mime):
    key = Secret(Path(os.environ['MANDOL_AML_KEY_FILE']).read_text().strip())
    origin = os.environ['MANDOL_AML_MULTIMODAL_URL']
    suffix = uuid.uuid4().hex
    stream = io.BytesIO()
    Image.new('RGB', (16, 16), 'red').save(stream, format)
    image = {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(stream.getvalue()).decode()}}
    parts = [{'type': 'text', 'text': 'Before: a red qualification image.'}, image,
             {'type': 'text', 'text': 'After: preserve this source order.'}]
    body = {'request_id': 'image-' + suffix, 'user_id': 'image-' + suffix, 'session_id': 's',
            'messages': [{'role': 'user', 'content': parts}]}
    headers = {'Authorization': 'Bearer ' + key}
    with httpx.Client(base_url=origin, timeout=1800, trust_env=False) as client:
        for _ in range(2):
            assert client.post('/add', json=body, headers=headers).status_code == 200
        hits = validate_search(client.post('/search', headers=headers, json={'query': 'red qualification image',
            'user_id': body['user_id'], 'top_k': 100}), 100)
        assert hits and hits[0]['content'][:3] == parts
        invalid = body | {'request_id': 'invalid-' + suffix,
            'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'https://example.invalid/image.png'}}]}]}
        assert client.post('/add', json=invalid, headers=headers).status_code in (400, 422)


def test_real_mm_multi_image_and_image_query():
    key = Secret(Path(os.environ['MANDOL_AML_KEY_FILE']).read_text().strip())
    origin = os.environ['MANDOL_AML_MULTIMODAL_URL']
    suffix = uuid.uuid4().hex
    images = []
    for color in ('red', 'blue'):
        stream = io.BytesIO(); Image.new('RGB', (16, 16), color).save(stream, 'PNG')
        images.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}})
    parts = [{'type':'text','text':'Before the red square.'},images[0],
             {'type':'text','text':'Between the two qualification images.'},images[1],
             {'type':'text','text':'After the blue square.'}]
    body = {'request_id':'multi-'+suffix,'user_id':'multi-'+suffix,'session_id':'s',
            'messages':[{'role':'user','content':parts}]}
    with httpx.Client(base_url=origin,timeout=1800,trust_env=False,headers={'Authorization':'Bearer '+key}) as client:
        assert client.post('/add',json=body).status_code == 200
        for query in ([images[0]], [{'type':'text','text':'Find the red qualification square.'},images[0]]):
            hits = validate_search(client.post('/search',json={'query':query,'user_id':body['user_id'],'top_k':100}),100)
            assert hits and hits[0]['content'][:5] == parts
        assert client.post('/add',json=body).status_code == 200
