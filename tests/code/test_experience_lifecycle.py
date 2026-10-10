from pathlib import Path
from types import SimpleNamespace
import json

import pytest
from mandol_competition.code.api.coding_store import CodingEvidenceStore, RequestConflict
from mandol_competition.code.experience import ExperienceBuilder, source_units
from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder


MESSAGES=[{'role':'user','content':'  JWT issuer validation fails with the old staging configuration.\n'*4,'timestamp':1700000000000},
          {'role':'assistant','content':'Changing the parser did not help. Update the issuer configuration; tests are only proposed.\n'*3}]


def store(path, generator):
    s=CodingEvidenceStore(path);s.initialize()
    b=ExperienceBuilder(SimpleNamespace(store=s),generator=generator)
    s.experience_builder=b
    return s,b


def generator(calls):
    def generate(prepared, request_id):
        calls.append(request_id);e=prepared['evidence'][0]
        return {'task_summary':{'text':'Historical JWT issuer configuration failure',
                 'evidence':[{'source_uid':e['source_uid'],'quote':e['text'][2:58]}]},
                'diagnosis':[],'resolution':[],'knowledge':[]}, {'response_model':'UNIT_FAKE'}
    return generate


def test_raw_and_derived_publish_together_replay_restart_and_conflict(tmp_path):
    calls=[];s,b=store(tmp_path/'state.sqlite3',generator(calls))
    assert s.add('u','s','r',MESSAGES)==2
    assert s.add('u','s','r',MESSAGES)==2 and len(calls)==1
    with pytest.raises(RequestConflict):s.add('u','s','r',MESSAGES[:1])
    _,rows=s.snapshot('u')
    assert [json.loads(r['payload'])['content'] for r in rows]==[m['content'] for m in MESSAGES]
    memories=[m for batch in s.state.iter_memories(s.scope('u')) for m in batch]
    assert len(memories)==3
    assert len({m.commit_seq for m in memories})==1
    derived=next(m for m in memories if m.memory_type=='engineering_experience')
    assert derived.metadata['source_versions']=={r['uid']:1 for r in rows}
    assert derived.raw_data['episode']['outcome']=='unknown'
    sources={u.uid:u for u in source_units(rows,'u')}
    raw=EngineeringEpisodeBuilder().build(list(sources.values()),user_id='u',session_id='s')
    assert b.overlay('u',[raw],sources)[0].task_summary_claim is not None
    assert b.overlay('other',[raw],sources)[0].task_summary_claim is None
    s.close();s,b=store(tmp_path/'state.sqlite3',generator(calls))
    assert s.add('u','s','r',MESSAGES)==2 and len(calls)==1
    s.close()


def test_pre_t1_restart_uses_completed_provider_preparation(tmp_path):
    calls=[];s,b=store(tmp_path/'state.sqlite3',generator(calls))
    def fault(point):
        if point=='before_T1':raise RuntimeError('unit interruption')
    s.coordinator.fault=fault
    with pytest.raises(RuntimeError):s.add('u','s','r',MESSAGES)
    assert len(calls)==1 and s.state.request(s.scope('u'),'r') is None
    s.close();s,b=store(tmp_path/'state.sqlite3',generator(calls))
    s.add('u','s','r',MESSAGES)
    assert len(calls)==1 and s.state.scope_state(s.scope('u'))['canonical_seq']==s.state.scope_state(s.scope('u'))['published_seq']
    s.close()


def test_unknown_provider_outcome_preserves_raw_ack_without_repeated_calls(tmp_path):
    calls=[]
    def broken(*args):calls.append(1);raise TimeoutError('unknown')
    s,b=store(tmp_path/'state.sqlite3',broken)
    assert s.add('u','s','r',MESSAGES)==2
    assert s.add('u','s','r',MESSAGES)==2 and len(calls)==1
    assert s.state.pending_count()==0 and len(s.snapshot('u')[1])==2
    assert not [m for batch in s.state.iter_memories(s.scope('u')) for m in batch if m.memory_type=='engineering_experience']
    s.close()


def test_source_append_replaces_derived_and_purge_erases_lineage(tmp_path):
    calls=[];s,b=store(tmp_path/'state.sqlite3',generator(calls))
    s.add('u','s','r',MESSAGES)
    s.add('u','s','r2',[{'role':'user','content':'The earlier fix failed; investigation continues.'}])
    derived=[m for batch in s.state.iter_memories(s.scope('u')) for m in batch if m.memory_type=='engineering_experience']
    assert len(derived)==1 and len(derived[0].metadata['source_versions'])==3 and len(calls)==2
    s._erase(s.scope('u'))
    assert not [m for batch in s.state.iter_memories(s.scope('u')) for m in batch]
    assert not s.state._rows('SELECT 1 FROM build_jobs WHERE scope=?',(s.scope('u').key,))
    s.close()


def test_source_revision_cannot_keep_a_cached_derived_episode(tmp_path, monkeypatch):
    from dataclasses import replace
    from mandol_substrate.state import Mutation
    from mandol_competition.code.api.coding_memory import CodingMemoryService
    from mandol_competition.code.api.config import Settings
    profile=tmp_path/'candidate.json'
    profile.write_text(json.dumps({'schema':'mandol-code-experience-candidate-v1','id':'unit',
                                   'recipe':'code-source-cited-experience-v1','distill':True,'route':'raw'}))
    monkeypatch.setenv('MANDOL_CODE_QUALITY_CANDIDATE',str(profile))
    memory=CodingMemoryService(Settings(coding_db_path=str(tmp_path/'state.sqlite3')))
    memory.start();calls=[];memory.experience_builder.generate=generator(calls)
    memory.add_messages('u','s','r',MESSAGES);memory.search('u','issuer',10)
    assert any(e.task_summary_claim for retriever,_ in memory._cache.values() for e in retriever.episodes.values())
    scope=memory.store.scope('u')
    original=next(m for batch in memory.store.state.iter_memories(scope) for m in batch if m.memory_type=='engineering_raw' and m.role=='user')
    text='The issuer explanation was corrected; the parser is now the investigation focus.'
    raw=dict(original.raw_data);raw['text_content']=text;raw['message']=dict(raw['message'],content=text)
    corrected=replace(original,version=2,text_content=text,raw_data=raw)
    memory.store.coordinator.apply(scope,'unit-revision','unit-digest',Mutation(memories=(corrected,)),{})
    memory.search('u','parser',10)
    assert not any(e.task_summary_claim for retriever,_ in memory._cache.values() for e in retriever.episodes.values())
    assert len(calls)==1  # Search never regenerates experience.
    memory.shutdown();memory.store.close()
