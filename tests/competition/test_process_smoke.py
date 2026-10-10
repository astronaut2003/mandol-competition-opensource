"""Three real Uvicorn processes, three SQLite files, one real scratch Qdrant.

Inference uses deterministic HTTP fixtures. No model-quality claim is made.
"""
from contextlib import ExitStack
import base64
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import httpx
from PIL import Image
from tests.support.http_inference import fake_inference


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); return sock.getsockname()[1]


def test_three_independent_aml_services(tmp_path):
    qdrant = os.environ['MANDOL_TEST_QDRANT']
    suffix = uuid.uuid4().hex
    key = 'smoke-key-' + suffix
    processes, logs, roots, collections = [], [], {}, []
    with ExitStack() as stack, httpx.Client(timeout=30, trust_env=False) as http:
        dense, dense_calls = stack.enter_context(fake_inference('dense'))
        sparse, _ = stack.enter_context(fake_inference('sparse'))
        reranker, _ = stack.enter_context(fake_inference('reranker'))
        mm, _ = stack.enter_context(fake_inference('multimodal'))
        vlm, vlm_calls = stack.enter_context(fake_inference('vlm'))
        env = dict(os.environ, MANDOL_RUNTIME_ROOT=str(tmp_path / 'data'), MANDOL_MEMORY_SYSTEM_KEY=key,
            MANDOL_QDRANT_ENDPOINT=qdrant, MANDOL_MM_VLM_BASE_URL=vlm + '/v1', OPENAI_API_KEY='fixture-only',
            MANDOL_TEXTUAL_EMBEDDING_ENDPOINT=dense, MANDOL_CODE_EMBEDDING_ENDPOINT=dense,
            MANDOL_TEXTUAL_SPARSE_ENDPOINT=sparse, MANDOL_TEXTUAL_RERANKER_ENDPOINT=reranker,
            MANDOL_MULTIMODAL_EMBEDDING_ENDPOINT=mm, MANDOL_MULTIMODAL_SPARSE_ENDPOINT=mm,
            MANDOL_MULTIMODAL_RERANKER_ENDPOINT=mm)
        env.pop('PYTHONPATH', None); env.pop('MANDOL_CONFIG', None)
        try:
            for track in ('textual', 'code', 'multimodal'):
                collection = track + '_v1'
                assert http.get(f'{qdrant}/collections/{collection}').status_code == 404, 'scratch collection already exists; refusing to adopt/delete it'
                collections.append((collection, 'smoke_' + suffix + '_' + track))
                local_env = env | {'MANDOL_' + track.upper() + '_NAMESPACE': collections[-1][1]}
                address = 'http://127.0.0.1:' + str(port()); roots[track] = address
                log = (tmp_path / (track + '.log')).open('w'); logs.append(log)
                process = subprocess.Popen([os.getenv('MANDOL_TEST_' + track.upper() + '_PYTHON', sys.executable), '-m', 'uvicorn', 'mandol_competition.runtime.apps:' + track,
                    '--factory', '--host', '127.0.0.1', '--port', address.rsplit(':', 1)[1], '--workers', '1', '--no-access-log'],
                    env=local_env, stdout=log, stderr=subprocess.STDOUT)
                processes.append(process)
                end = time.monotonic() + 25
                while time.monotonic() < end:
                    assert process.poll() is None, (tmp_path / (track + '.log')).read_text()
                    try:
                        if http.get(address + '/readyz', timeout=1).status_code == 200: break
                    except httpx.RequestError: pass
                    time.sleep(.1)
                else: raise AssertionError('startup unavailable: ' + (tmp_path / (track + '.log')).read_text()[:2000])
            stream = io.BytesIO(); Image.new('RGB', (3, 3), 'red').save(stream, 'PNG')
            image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()}}
            for track, address in roots.items():
                fixture = json.loads((Path(__file__).parent / 'fixtures' / (track + '.json')).read_text())
                body, query = fixture['add'], fixture['search']
                content = body['messages'][0]['content']
                assert http.get(address + '/health').status_code == 200
                assert http.get(address + '/livez').status_code == 200
                assert http.post(address + '/add', json=body).status_code == 401
                for header in ({'Authorization': 'Token ' + key}, {'Authorization': 'Bearer ' + key}, {'X-Api-Key': key}):
                    result = http.post(address + '/add', json=body, headers=header)
                    assert result.status_code == 200, result.text
                    assert result.json() == {'success': True, **{k: body[k] for k in ('request_id', 'user_id', 'session_id')}}
                header = {'X-Api-Key': key}
                changed = body | {'session_id': 'other'}
                assert http.post(address + '/add', json=changed, headers=header).status_code == 409
                assert http.post(address + '/add', json={k: v for k, v in body.items() if k != 'user_id'}, headers=header).status_code == 422
                assert http.post(address + '/search', json=query, headers={'X-Api-Key': 'invalid'}).status_code == 401
                result = http.post(address + '/search', json=query, headers=header)
                assert result.status_code == 200, result.text
                data = result.json()['data']; assert 0 < len(data) <= 100
                assert all(row['id'] and row['content'] for row in data)
                again = http.post(address + '/search', json=query | {'top_k': 1}, headers=header).json()['data']
                assert len(again) <= 1 and again[0]['id'] == data[0]['id']
                empty = http.post(address + '/search', json=query | {'user_id': 'another'}, headers=header)
                assert empty.status_code == 200 and empty.json() == {'data': []}
                if track == 'multimodal': assert data[0]['content'][:2] == content
                metrics = http.get(address + '/metrics').text
                assert 'request_duration_seconds_count' in metrics and body['user_id'] not in metrics and key not in metrics
            assert len([p for p in (tmp_path / 'data').rglob('state.sqlite3')]) == 3
            assert len(vlm_calls) == 1, 'duplicate Add repeated VLM'
            # Explicitly opt in to restart only the known disposable server.
            container = os.getenv('MANDOL_SCRATCH_CONTAINER')
            if container:
                assert container.startswith('mandol-unified-qualification-')
                subprocess.run(['docker', 'restart', container], check=True, capture_output=True)
                for address in roots.values():
                    end = time.monotonic() + 20
                    while time.monotonic() < end:
                        if http.get(address + '/readyz').status_code == 200: break
                        time.sleep(.2)
                    else: raise AssertionError('service failed to recover shared Qdrant restart')
            report = {'services': 3, 'sqlite_files': 3, 'qdrant': 'real scratch one server',
                      'inference': 'fake HTTP', 'aml_wire_smoke': 'passed', 'qdrant_restart': bool(container)}
            (tmp_path / 'result.json').write_text(json.dumps(report, indent=2))
        finally:
            for process in processes: process.terminate()
            for process in processes:
                try: process.wait(timeout=25)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            for log in logs: log.close()
            for collection, namespace in collections:
                try:
                    response = http.get(f'{qdrant}/collections/{collection}')
                except httpx.RequestError:
                    continue  # retain scratch evidence; do not hide the primary failure
                if response.status_code == 200 and response.json()['result']['config'].get('metadata', {}).get('namespace') == namespace:
                    http.delete(f'{qdrant}/collections/{collection}').raise_for_status()
