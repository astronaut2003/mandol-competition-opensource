"""Real adapter/store publication with HTTP timeout before/after durable work."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import threading
import time
import uuid
import httpx
import pytest
from fastapi.testclient import TestClient
from mandol_competition.runtime import apps
from tests.support.http_inference import fake_inference


@pytest.mark.parametrize('track', ['textual', 'code', 'multimodal'])
@pytest.mark.parametrize('pause_at', ['before_commit', 'after_commit'])
def test_timeout_retry_keeps_single_logical_result(tmp_path, monkeypatch, track, pause_at):
    endpoint = os.environ['MANDOL_TEST_QDRANT']; collection = track + '_v1'
    assert httpx.get(endpoint + '/collections/' + collection, trust_env=False).status_code == 404
    namespace = 'timeout_' + uuid.uuid4().hex
    with ExitStack() as stack:
        urls = {kind: stack.enter_context(fake_inference(kind)) for kind in ('dense', 'sparse', 'reranker', 'multimodal', 'vlm')}
        for key,value in {'MANDOL_RUNTIME_ROOT': str(tmp_path), 'MANDOL_QDRANT_ENDPOINT': endpoint,
            'MANDOL_' + track.upper() + '_NAMESPACE': namespace, 'MANDOL_MEMORY_SYSTEM_KEY': 'fixture',
            'AML_ADD_TIMEOUT_SECONDS': '0.3', 'MANDOL_MM_ADD_TIMEOUT_SECONDS': '0.3',
            'MANDOL_MM_VLM_BASE_URL': urls['vlm'][0] + '/v1', 'OPENAI_API_KEY': 'fixture-only'}.items(): monkeypatch.setenv(key, value)
        for key,channel in [('TEXTUAL_EMBEDDING','dense'), ('CODE_EMBEDDING','dense'), ('TEXTUAL_SPARSE','sparse'), ('TEXTUAL_RERANKER','reranker'), ('MULTIMODAL_EMBEDDING','multimodal'), ('MULTIMODAL_SPARSE','multimodal'), ('MULTIMODAL_RERANKER','multimodal')]:
            monkeypatch.setenv('MANDOL_' + key + '_ENDPOINT', urls[channel][0])
        from mandol_competition.multimodal.config import get_config
        get_config.cache_clear()
        released, paused, completed, finished = (threading.Event() for _ in range(4))
        try:
            app = getattr(apps, track)()
            obj = app.state.add_service if track == 'multimodal' else app.state.memory_service
            name = 'handle' if track == 'multimodal' else 'add_messages'
            original = getattr(obj, name)
            count = []
            def slow(*args, **kwargs):
                try:
                    count.append(1)
                    if len(count) == 1 and pause_at == 'before_commit':
                        paused.set(); assert released.wait(10)
                    result = original(*args, **kwargs)
                    if len(count) == 1 and pause_at == 'after_commit':
                        paused.set(); assert released.wait(10)
                    completed.set()
                    return result
                finally: finished.set()
            setattr(obj, name, slow)
            with TestClient(app) as client:
                until = time.monotonic() + 15
                while client.get('/readyz').status_code != 200:
                    assert time.monotonic() < until, getattr(obj, '_init_error', 'startup not ready')
                    time.sleep(.05)
                fixture = json.loads((Path(__file__).parent / 'fixtures' / (track + '.json')).read_text())
                headers = {'X-Api-Key': 'fixture'}
                first = client.post('/add', json=fixture['add'], headers=headers)
                assert first.status_code == 503, first.text
                assert paused.wait(10)
                released.set(); assert finished.wait(10)
                # With an end-to-end budget, work paused before publication
                # may stop before a new inference dispatch. Its durable ledger
                # must allow the same request to complete on retry. A committed
                # write must still finish and replay its original receipt.
                if pause_at == 'after_commit': assert completed.is_set()
                replay = client.post('/add', json=fixture['add'], headers=headers)
                assert replay.status_code == 200 and replay.json()['success'] is True, replay.text
                assert client.post('/search', json=fixture['search'], headers=headers).json()['data']
                state = app.state.backend.state if track == 'multimodal' else (obj.store.state if track == 'code' else obj.state)
                assert state._rows('SELECT COUNT(*) AS n FROM requests')[0]['n'] == 1
                assert state._rows('SELECT COUNT(*) AS n FROM memories')[0]['n'] == 1
                if track == 'multimodal': assert len(urls['vlm'][1]) == 1
        finally:
            released.set(); get_config.cache_clear()
            response = httpx.get(endpoint + '/collections/' + collection, trust_env=False)
            if response.status_code == 200 and response.json()['result']['config']['metadata']['namespace'] == namespace:
                httpx.delete(endpoint + '/collections/' + collection, trust_env=False).raise_for_status()
