from dataclasses import asdict, replace

import pytest

from mandol_substrate.contracts import ResourceIdentity, SpacePolicy
from mandol_substrate.state import Conflict, ProfileMismatch, SQLiteStateStore


def test_manifest_binding_and_explicit_empty_spaces(tmp_path, profile, scope):
    path = tmp_path / "state.sqlite3"
    identity = ResourceIdentity("test", scope.namespace, "test_v1")
    with_store = SQLiteStateStore(path, profile, space_policy=SpacePolicy(), identity=identity)
    with_store.ensure_scope(scope)
    assert with_store._rows("SELECT * FROM memory_spaces") == []
    with_store.close()
    with pytest.raises(ProfileMismatch):
        SQLiteStateStore(path, profile, space_policy=SpacePolicy(),
                         identity=replace(identity, track="other"))
    reopened = SQLiteStateStore(path, profile, space_policy=SpacePolicy(), identity=identity)
    reopened.close()


def test_preparation_replay_survives_restart_and_conflicts(tmp_path, profile, scope):
    path = tmp_path / "state.sqlite3"
    state = SQLiteStateStore(path, profile)
    assert state.ledger.reserve(scope, "request", "digest") == {}
    state.ledger.checkpoint(scope, "request", "digest", {"prepared": ["model output"]})
    state.close()
    state = SQLiteStateStore(path, profile)
    assert state.ledger.reserve(scope, "request", "digest") == {"prepared": ["model output"]}
    with pytest.raises(Conflict):
        state.ledger.reserve(scope, "request", "different")
    with pytest.raises(ValueError):
        state.ledger.checkpoint(scope, "request", "digest", {"image": "data:image/png;base64,AAAA"})
    state.begin_purge(scope, {})
    state.finish_purge(scope)
    assert state._rows("SELECT * FROM request_preparations") == []
    state.close()


def test_degradation_is_durable_without_payloads(store, scope):
    store.record_event(scope, "search", "dense", "Timeout")
    store.record_event(scope, "search", "dense", "Timeout")
    assert store._rows("SELECT count FROM runtime_events")[0]["count"] == 2
    with pytest.raises(ValueError):
        store.record_event(scope, "search", "dense", "raw response contains secret")


@pytest.mark.parametrize('text', ['data:image/', 'data:image/png;base64,AAAA',
    'Markdown example: ![example](data:image/png;base64,AAAA)',
    '{"image_url":{"url":"data:image/png;base64,AAAA"}}'])
def test_checkpoint_preserves_source_text_containing_image_uri_syntax(store, scope, text):
    value = {'units': [{'raw_data': {'original_content': text, 'text_content': text,
        'content_parts': [{'type':'text','text':text}]}, 'metadata': {'description':text}}]}
    store.ledger.reserve(scope, 'source-text', 'source-digest')
    store.ledger.checkpoint(scope, 'source-text', 'source-digest', value)
    assert store.ledger.reserve(scope, 'source-text', 'source-digest') == value


@pytest.mark.parametrize('value', [
    {'image':'data:image/png;base64,AAAA'},
    {'image_data_uri':'data:image/png;base64,AAAA'},
    {'content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,AAAA'}}]},
    {'nested':{'image_url':' DATA:IMAGE/PNG;BASE64,AAAA'}},
])
def test_checkpoint_still_rejects_actual_structured_inline_media(store, scope, value):
    store.ledger.reserve(scope, 'media', 'digest')
    with pytest.raises(ValueError, match='inline media'):
        store.ledger.checkpoint(scope, 'media', 'digest', value)
    assert store.ledger.reserve(scope, 'media', 'digest') == {}


def test_contracts_do_not_change_frozen_semantic_serialization(profile):
    from mandol_substrate.state.models import digest

    assert profile.hash == digest(asdict(profile))
    assert "track" not in asdict(profile)
