from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

from mandol_competition.multimodal.media_store import UserMediaStore
from mandol_competition.multimodal.service import SearchService
from .test_search_service import setup, image, add_request, search_request


def test_request_media_reuse_preserves_order_budget_and_does_not_survive_next_request(tmp_path, monkeypatch):
    raw, uri = image('red')
    add, legacy, graphs, extractor = setup(tmp_path, {})
    for index in range(3):
        add.handle(add_request('alice', uri, index))
    units = list(graphs['alice'].units.values())
    backend = SimpleNamespace(profile=SimpleNamespace(options={}), state=None,
        projection=SimpleNamespace(policy=SimpleNamespace(physical_search_budget=100)),
        user_operation=lambda user, *, write=True: nullcontext(), assert_searchable=lambda user: None,
        search=lambda user, query, top_k: [(u, 1.) for u in units][:top_k])
    config = replace(legacy.config, image_byte_budget=len(raw)*2)
    search = SearchService(config, extractor=extractor, backend=backend)
    request = search_request('alice', [{'type': 'text', 'text': 'photo'}])
    calls = []
    original = UserMediaStore.get
    def read(store, media_id):
        calls.append(media_id)
        return original(store, media_id)
    monkeypatch.setattr(UserMediaStore, 'get', read)
    baseline = search.handle(request)
    assert len(calls) == 3
    calls.clear()
    backend.profile.options['reuse_media_within_response'] = True
    assert search.handle(request) == baseline
    assert len(calls) == 1
    assert sum(p['type']=='image_url' for h in baseline['data'] for p in h['content']) == 2
    # No cross-request cache: later corruption/missing media is observed.
    monkeypatch.setattr(UserMediaStore, 'get', lambda *args: None)
    after = search.handle(request)
    assert all(p['type']!='image_url' for h in after['data'] for p in h['content'])
    legacy.registry.close()
    legacy.store.close()
