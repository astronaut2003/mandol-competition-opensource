from dataclasses import replace
from types import SimpleNamespace

import pytest
from mandol_substrate.state import CanonicalMemory, Mutation

from mandol_competition.multimodal.episodes import (
    RECIPE, TYPE, prepare_navigation, route, supported,
)
from .backend_support import services, request, query
from .test_sqlite_qdrant import image_part


def test_visual_source_window_is_atomic_replayable_and_returns_original_evidence(backend):
    backend.visual_policy = {'version': RECIPE, 'episode': True, 'adaptive': False}
    add, search = services(backend)
    body = request(content=[{'type':'text','text':'During my trip I visited a red station.'},
                            image_part('red'), image_part('blue')])
    expected = add.handle(body)
    scope = backend.scope('alice')
    memories = {m.logical_uid:m for batch in backend.state.iter_memories(scope) for m in batch}
    derived = next(m for m in memories.values() if m.memory_type == TYPE)
    raw = next(m for m in memories.values() if m.memory_type != TYPE)
    assert supported(derived, memories)
    assert derived.commit_seq == raw.commit_seq
    assert 'data:image' not in str(derived.raw_data)
    calls = backend.encoders.documents
    assert add.handle(body) == expected
    assert backend.encoders.documents == calls
    results = search.handle(query(text='What happened during my trip?'))['data']
    assert results and {h['id'] for h in results} == {raw.logical_uid}
    assert search.handle(query(user='bob',text='trip')) == {'data':[]}
    changed = replace(raw, version=2, text_content='Corrected station description')
    backend.coordinator.apply(scope, 'source-correction', 'correction', Mutation(memories=(changed,)), {})
    with backend.coordinator.search_view(scope) as view:
        routed = route(backend.state, scope, 'during my trip', view, enabled=True)
    assert routed.support_uids == ()
    assert derived.logical_uid not in routed.allowed_uids
    assert {h['id'] for h in search.handle(query(text='station'))['data']} == {raw.logical_uid}
    backend.purge('alice')
    assert not list(backend.state.iter_memories(scope))


def test_window_includes_new_undated_sources_and_remains_bounded():
    from mandol_substrate.state import Scope
    old = [CanonicalMemory(str(i), 'old '+str(i), session_id='s', commit_seq=i+1,
           raw_data={'content_parts':[{'type':'image','media_id':'old-'+str(i)}]}) for i in range(20)]
    incoming = CanonicalMemory('new', 'new observation', session_id='s',
        raw_data={'content_parts':[{'type':'image','media_id':'new-image'}]})
    state = SimpleNamespace(iter_memories=lambda _: [old])
    result = prepare_navigation(state, Scope('multimodal','alice'),
        SimpleNamespace(session_id='s',user_id='alice',request_id='r'), (incoming,))
    nav = result.memories[-1]
    assert len(nav.metadata['source_refs']) == 8
    assert nav.metadata['source_refs'][-1]['uid'] == 'new'
    assert result.jobs[0].status == 'completed'


def test_corrupt_source_span_falls_back_instead_of_becoming_truth():
    source = CanonicalMemory('raw','station')
    nav = CanonicalMemory('derived','navigation',metadata={'recipe':RECIPE,
        'source_refs':[{'uid':'raw','span':'not a span'}]})
    assert not supported(nav, {'raw':source})


def test_unversioned_visual_policy_rejected_before_model_loading(tmp_path):
    from mandol_competition.multimodal.episodes import quality_policy
    with pytest.raises(ValueError):
        quality_policy(SimpleNamespace(options={'visual_memory_policy':{'episode':True}}))


