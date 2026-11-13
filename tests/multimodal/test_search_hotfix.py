"""Synthetic wire failures and fast restart of owned fixture databases only."""
from dataclasses import replace
from types import SimpleNamespace
import json
import uuid

import pytest

from mandol_competition.inference.evidence_selector import EvidenceSelector, Selection
from mandol_competition.inference.errors import (
    InferenceSchemaError, InferenceRateLimited, InferenceUnavailable, InferenceIdentityError)
from mandol_competition.multimodal import evidence_review as review
from mandol_competition.multimodal.grounded_delivery import deliver
from mandol_competition.multimodal.config import MultimodalConfig, ConfigurationError, get_config
from mandol_competition.multimodal.backend import MultimodalBackend
from mandol_substrate.state import StateUnavailable, SQLiteStateStore, Mutation
from .backend_support import Encoders, profile, runtime, services, request, query, OWNED_COLLECTIONS
from .test_grounded_evidence import memory


def wire(indices):
    return SimpleNamespace(complete=lambda *a, **k: dict(model='gpt-4o-mini', choices=[
        dict(finish_reason='stop', message=dict(content=json.dumps(dict(indices=indices))))]))


@pytest.mark.parametrize('indices, expected', [([0,0,1],(0,1)), ([2,1,0],(2,1)), ([],())])
def test_only_valid_duplicate_and_excess_indices_are_normalized(indices, expected):
    selector=EvidenceSelector('https://synthetic.invalid', client=wire(indices),
        strict=True, normalize_indices=True)
    result=selector.select(query='Artificial label?', options=[],
        candidates=[{'text':str(i)} for i in range(3)], limit=2)
    assert result.indices == expected


@pytest.mark.parametrize('indices', [[True],[-1],[3],['0'],[None],{}])
def test_invalid_indices_are_never_guessed(indices):
    selector=EvidenceSelector('https://synthetic.invalid', client=wire(indices),
        strict=True, normalize_indices=True)
    with pytest.raises(InferenceSchemaError):
        selector.select(query='q', options=[], candidates=[{'text':'fact'}], limit=1)


def test_other_tracks_keep_strict_duplicate_rejection_by_default():
    selector=EvidenceSelector('https://synthetic.invalid', client=wire([0,0]), strict=True)
    with pytest.raises(InferenceSchemaError):
        selector.select(query='q', options=[], candidates=[{'text':'fact'}], limit=1)


@pytest.mark.parametrize('phase', ['primary','verification'])
@pytest.mark.parametrize('error', ['selector_invalid_indices','selector_invalid_response',
    'selector_invalid_schema','selector_incomplete_output'])
def test_bad_optional_review_keeps_all_ranks_scores_and_originals(phase,error):
    pool=[(memory('a','Printed label ORBIT-47'),.9),(memory('b','Original correction'),.2)]
    calls=[]
    def select(**inputs):
        calls.append(1)
        if phase=='primary' or len(calls)>1:
            raise InferenceSchemaError(error)
        return Selection((), 'selected', 0.)
    guard=SimpleNamespace(select=select, confirm_absence=select)
    plan=review.review_sources(guard,query(),pool,SimpleNamespace(get=lambda _:None))
    assert plan.hits == pool and all(a is b for a,b in zip(plan.hits,pool))
    assert not plan.complete_absence_review and plan.preferred_sources == ()
    assert plan.model_calls == len(calls) <= 2
    result=deliver(MultimodalConfig(),plan,SimpleNamespace(get=lambda _:None))
    assert [r['id'] for r in result['data']] == ['a','b']
    assert [r['score'] for r in result['data']] == [.9,.2]
    assert 'Review incomplete' in result['data'][0]['content'][-1]['text']


@pytest.mark.parametrize('error', [InferenceRateLimited(retry_after=60),
    InferenceUnavailable('ReadTimeout',unknown_outcome=True), InferenceIdentityError('wrong identity')])
def test_rate_network_and_identity_errors_are_not_hidden(error):
    def fail(**_): raise error
    with pytest.raises(type(error)):
        review.review_sources(SimpleNamespace(select=fail),query(),
            [(memory('a','fact'),.7)],SimpleNamespace(get=lambda _:None))


def test_full_audit_option_is_explicit_and_default_is_safe(monkeypatch):
    get_config.cache_clear()
    try:
        monkeypatch.delenv('MANDOL_MM_FULL_DATABASE_AUDIT',raising=False)
        assert get_config().full_database_audit is True
        get_config.cache_clear();monkeypatch.setenv('MANDOL_MM_FULL_DATABASE_AUDIT','0')
        assert get_config().full_database_audit is False
        get_config.cache_clear();monkeypatch.setenv('MANDOL_MM_FULL_DATABASE_AUDIT','typo')
        with pytest.raises(ConfigurationError):get_config()
    finally:get_config.cache_clear()


def test_fast_restart_and_search_never_scan_existing_corpus(tmp_path,qdrant_server,monkeypatch):
    collection='hotfix_'+uuid.uuid4().hex
    original=runtime(tmp_path,qdrant_server.uri,collection)
    original.start()
    add,_=services(original)
    add.handle(request(content='red backpack'))
    config=replace(original.config,full_database_audit=False)
    original.close()
    def forbidden(*a,**kw):raise AssertionError('unexpected whole database audit')
    monkeypatch.setattr(SQLiteStateStore,'integrity_check',forbidden)
    import mandol_competition.multimodal.backend as module
    monkeypatch.setattr(module,'verify_projection',forbidden)
    restored=MultimodalBackend(config,profile=profile(),encoders=Encoders(forbid_documents=True))
    OWNED_COLLECTIONS.add((qdrant_server.uri,collection))
    try:
        monkeypatch.setattr(restored.state.lexical,'verify_scope',forbidden)
        monkeypatch.setattr(restored.projection,'audit_rows',forbidden)
        monkeypatch.setattr(restored.state,'iter_memories',forbidden)
        restored.start()
        _,search=services(restored)
        assert search.handle(query())['data']
        assert search.handle(query(user='other')) == {'data':[]}
        scope=restored.scope('alice')
        restored.state.prepare(scope,'unpublished','different',Mutation(),{})
        with pytest.raises(StateUnavailable):search.handle(query())
    finally:restored.close()


def test_fast_start_does_not_publish_an_interrupted_mutation(tmp_path,qdrant_server):
    collection='hotfix_'+uuid.uuid4().hex
    original=runtime(tmp_path,qdrant_server.uri,collection)
    original.start()
    scope=original.scope('alice');original.state.ensure_scope(scope)
    original.state.prepare(scope,'pending','digest',Mutation(),{})
    config=replace(original.config,full_database_audit=False)
    original.close()
    restored=MultimodalBackend(config,profile=profile(),encoders=Encoders(forbid_documents=True))
    OWNED_COLLECTIONS.add((qdrant_server.uri,collection))
    try:
        with pytest.raises(StateUnavailable):restored.start()
    finally:restored.close()
