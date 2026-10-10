from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import os
import uuid
import pytest
from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.config import Settings
from mandol_competition.code.projection import prepare_channel
from mandol_competition.code.api.remote_embeddings import RemoteDenseEncoder
from mandol_competition.runtime.storage import claim_runtime_root
from mandol_competition.runtime.maintenance import maintain
from tests.support.http_inference import fake_inference


def test_offline_code_verify_purge_and_writer_exclusion(tmp_path, monkeypatch):
    endpoint = os.environ['MANDOL_TEST_QDRANT']
    # A shared qualification server already has code_v1. The maintenance
    # implementation is injected with an explicit test binding; never adopt or
    # delete the running application's collection.
    import httpx
    collection = 'code_maintenance_' + uuid.uuid4().hex
    assert httpx.get(endpoint + '/collections/' + collection, trust_env=False).status_code == 404
    namespace = 'maintenance_' + uuid.uuid4().hex
    monkeypatch.setenv('MANDOL_RUNTIME_ROOT', str(tmp_path))
    monkeypatch.setenv('MANDOL_CODE_NAMESPACE', namespace)
    monkeypatch.setenv('MANDOL_QDRANT_ENDPOINT', endpoint)
    root = claim_runtime_root(tmp_path / 'code', 'code')
    from mandol_competition.config import load_config
    import mandol_competition.runtime.maintenance as maintenance_module
    cfg = load_config()
    test_binding = replace(cfg.track('code'), collection=collection)
    monkeypatch.setattr(maintenance_module, 'load_config', lambda: SimpleNamespace(
        track=lambda name: test_binding if name == 'code' else cfg.track(name)))
    with fake_inference('dense') as (url, _):
        monkeypatch.setenv('MANDOL_CODE_EMBEDDING_ENDPOINT', url)
        service = CodingMemoryService(Settings(coding_db_path=str(root / 'state.sqlite3'), namespace=namespace,
                                              qdrant_endpoint=endpoint, qdrant_collection=collection))
        service.start()
        try:
            service._encoder = RemoteDenseEncoder(url)
            service.add_messages('u', 's', 'r', [{'role':'user','content':'Orion evidence'}])
            prepare_channel(service, 'u', {m.logical_uid: m.text_content for batch in service.store.state.iter_memories(service.store.scope('u')) for m in batch})
            with pytest.raises(BlockingIOError): maintain('code', 'verify')
        finally: service.shutdown()
        try:
            assert maintain('code', 'verify')['projection_servable']
            assert maintain('code', 'purge', user_id='u')['projection']['eligible'] == 0
            assert maintain('code', 'gc')['projection']['clean']
        finally:
            value = httpx.get(endpoint + '/collections/' + collection, trust_env=False).json()
            if value.get('result', {}).get('config', {}).get('metadata', {}).get('namespace') == namespace:
                httpx.delete(endpoint + '/collections/' + collection, trust_env=False).raise_for_status()
