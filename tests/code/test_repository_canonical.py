from mandol_competition.code.api.coding_store import CodingEvidenceStore
from mandol_competition.code.ingestion import prepare_repository


def test_repository_output_roundtrip_and_incremental_sources(tmp_path):
    repo = tmp_path / 'input'; repo.mkdir()
    source = repo / 'calc.py'
    source.write_text('def add(a, b):\n    # preserved whitespace\n    return a + b\n')
    store = CodingEvidenceStore(tmp_path / 'state.sqlite3'); store.initialize()
    scope = store.scope('u'); store.state.ensure_scope(scope)
    def prepare(request):
        return prepare_repository(repo, state=store.state, scope=scope, request_id=request,
            payload_digest=request, cache_path=tmp_path / 'cache.json', include_git_history=False)
    first = prepare('r1')
    assert first.mutation.memories and first.mutation.edges
    store.coordinator.apply(scope, first.request_id, first.payload_digest, first.mutation, first.result)
    memories = [m for batch in store.state.iter_memories(scope) for m in batch]
    functions = [m for m in memories if m.memory_type == 'function']
    assert functions and functions[0].raw_data['start_line'] == 1
    assert functions[0].raw_data['file_path'] == 'calc.py'
    assert 'return a + b' in str(functions[0].raw_data)
    source.write_text('def add(a, b):\n    return a - b\n')
    second = prepare('r2')
    assert any(m.version == 2 and m.logical_uid == functions[0].logical_uid for m in second.mutation.memories)
    store.coordinator.apply(scope, second.request_id, second.payload_digest, second.mutation, second.result)
    store.close()
    restored = CodingEvidenceStore(tmp_path / 'state.sqlite3'); restored.initialize()
    assert 'return a - b' in str(restored.state.get_memories(scope, [functions[0].logical_uid]))
    assert not (repo / '.mandol').exists()
    restored.close()
