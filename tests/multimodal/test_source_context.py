"""Actual scoped SQLite/Qdrant with artificial original-source contexts."""
from copy import deepcopy
import hashlib
import json
import pytest

from mandol_substrate.representation import MemoryUnit
from mandol_competition.multimodal.protocol import AddRequest
from mandol_competition.multimodal.source_context import collect_context, MAX_CONTEXT_SOURCES


def publish(backend,uid,text,*,user='alice',session='s1',request_id=None,turn=0,role='user',entity=None):
    request_id=request_id or uid
    raw={'text_content':text,'original_content':text,'content_parts':[{'type':'text','text':text}],
         'multimodal_info':{'images':[{'entities':[{'name':entity}]}] if entity else []}}
    unit=MemoryUnit(uid,raw,dict(user_id=user,session_id=session,request_id=request_id,
        message_index=turn,role=role,timestamp='2026-02-01'))
    request=AddRequest(user_id=user,request_id=request_id,session_id=session,
        messages=[{'role':role,'content':text}])
    backend.publish(request,hashlib.sha256(json.dumps(request.model_dump()).encode()).hexdigest(),[unit],{'ok':True})
    return unit


def test_neighbor_read_is_user_scoped_published_and_does_not_change_memory_or_frontier(backend):
    first=publish(backend,'seed','I selected a folder in the first exchange.')
    context=publish(backend,'support','The selected folder has label ORBIT-47.',role='assistant')
    publish(backend,'secret','Another user private folder.',user='bob')
    scope=backend.scope('alice')
    before=deepcopy(backend.state.scope_state(scope))
    rows=deepcopy(backend.state._rows('SELECT * FROM memories ORDER BY scope,logical_uid'))
    got=collect_context(backend,'alice',[(first,.9)],'Which folder was it from the previous exchange?')
    assert len(got)==1 and got[0].unit.uid==context.uid and got[0].relation=='same-session-neighbor'
    assert got[0].unit.metadata['role']=='assistant' and got[0].unit.metadata['user_id']=='alice'
    assert backend.state.scope_state(scope)==before
    assert backend.state._rows('SELECT * FROM memories ORDER BY scope,logical_uid')==rows


def test_exact_entity_anchor_finds_an_original_in_another_session(backend):
    seed=publish(backend,'seed','Nora Vale selected the violet folder.',session='choice',entity='Nora Vale')
    publish(backend,'bridge','Nora Vale later gave me the folder labeled ORBIT-47.',session='gift')
    publish(backend,'near-name','Nora Valen bought something unrelated.',session='distractor')
    publish(backend,'other-user','Nora Vale private note.',user='bob',session='other')
    got=collect_context(backend,'alice',[(seed,.9)],'Which folder did she give me later?')
    assert [c.unit.uid for c in got]==['bridge']
    assert got[0].relation=='shared-explicit-text-anchor' and got[0].unit.metadata['session_id']=='gift'


def test_atomic_question_does_not_launch_extra_context_reads(backend,monkeypatch):
    seed=publish(backend,'seed','Label ORBIT-47.')
    def forbidden(*args,**kwargs):raise AssertionError('unneeded supplemental reads')
    monkeypatch.setattr(backend.state,'get_memories',forbidden)
    assert collect_context(backend,'alice',[(seed,.9)],'What printed label is visible?')==()


def test_generic_entity_and_unrelated_session_do_not_create_an_identity_link(backend):
    seed=publish(backend,'seed','A woman holds a folder.',entity='woman')
    publish(backend,'other','A woman has a different folder.',session='unrelated')
    assert collect_context(backend,'alice',[(seed,.9)],'Which folder did she give me?')==()


def test_context_full_text_and_count_budgets_do_not_mutate_baseline(backend):
    seeds=[]
    for index in range(12):
        seed=publish(backend,f'seed-{index}',f'Entity Marker{index} selected a folder.',session=f's-{index}',entity=f'Marker{index}')
        seeds.append((seed,.9-index/100))
        publish(backend,f'support-{index}','Stored correction: '+'large original '*2000,session=f's-{index}')
    before=deepcopy(seeds)
    got=collect_context(backend,'alice',seeds,'Which folder was selected before?')
    assert len(got)<=MAX_CONTEXT_SOURCES and all(len(c.unit.raw_data['text_content'].encode())<12*1024 for c in got)
    assert seeds==before


def test_derived_summary_cannot_be_mislabeled_as_an_original_neighbor(backend):
    from dataclasses import replace
    from mandol_competition.multimodal.backend import canonical_memory
    from mandol_substrate.state import Mutation
    seed=publish(backend,'seed','Nora Vale selected a folder.',entity='Nora Vale')
    source=MemoryUnit('summary',{'text_content':'Nora Vale inferred a folder identity.',
        'original_content':'','content_parts':[{'type':'text','text':'Derived text'}]},
        dict(user_id='alice',session_id='s1',request_id='derived',message_index=1,role='derived'))
    derived=replace(canonical_memory(source,'alice'),memory_type='summary',memory_level='L1_Episode')
    backend.coordinator.apply(backend.scope('alice'),'derived','owned-hash',Mutation(memories=(derived,)),{'ok':True})
    assert collect_context(backend,'alice',[(seed,.9)],'Which folder did she give me later?')==()


@pytest.mark.parametrize('info',[None,[],{'images':None},
    {'images':[None,{'entities':None},{'entities':[False,{},{'name':123}]}]}])
def test_malformed_optional_entity_hints_cannot_break_baseline_search(backend,info):
    seed=publish(backend,'seed','Existing original folder fact.')
    seed.raw_data['multimodal_info']=info
    assert collect_context(backend,'alice',[(seed,.9)],'Which folder was it?')==()


def test_context_budget_includes_actual_provenance_not_only_body_text(backend):
    session='owned-'+'A'*13000
    seed=publish(backend,'seed','A selected folder.',session=session)
    publish(backend,'context','The previous folder label was ORBIT-47.',session=session)
    assert collect_context(backend,'alice',[(seed,.9)],'Which folder was it before?')==()
