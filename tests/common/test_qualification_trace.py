import json
from types import SimpleNamespace

from mandol_competition.observability.qualification import install_qualification_trace


def test_trace_preserves_output_and_omits_content_and_user(tmp_path, monkeypatch):
    path = tmp_path / 'trace.jsonl'
    monkeypatch.setenv('MANDOL_QUALIFICATION_TRACE_PATH', str(path))
    returned = [{'content': 'private response', 'score': .3}]
    memory = SimpleNamespace(add_messages=lambda **kwargs: 1, search=lambda **kwargs: returned,
                             start=lambda: None, _last_retrieval_path='LEXICAL_FALLBACK')
    app = SimpleNamespace(state=SimpleNamespace(memory_service=memory))
    install_qualification_trace(app, 'code')
    assert memory.search(user_id='private user', query='private query', top_k=100) is returned
    record = json.loads(path.read_text())
    assert record['final_count'] == 1 and record['retrieval_path'] == 'LEXICAL_FALLBACK'
    assert 'private' not in path.read_text()


def test_trace_disabled_keeps_methods(monkeypatch):
    monkeypatch.delenv('MANDOL_QUALIFICATION_TRACE_PATH', raising=False)
    sentinel = lambda: None
    app = SimpleNamespace(state=SimpleNamespace(memory_service=SimpleNamespace(search=sentinel)))
    assert install_qualification_trace(app, 'textual') is app
    assert app.state.memory_service.search is sentinel


def test_trace_write_failure_does_not_change_success(tmp_path, monkeypatch):
    # A directory at the output path simulates an unwritable trace destination.
    monkeypatch.setenv('MANDOL_QUALIFICATION_TRACE_PATH', str(tmp_path))
    returned = [{'content': 'evidence'}]
    memory = SimpleNamespace(add_messages=lambda **kwargs: 1, search=lambda **kwargs: returned,
                             start=lambda: None, _last_retrieval_path='DENSE')
    app = SimpleNamespace(state=SimpleNamespace(memory_service=memory))
    install_qualification_trace(app, 'code')
    assert memory.search() is returned
