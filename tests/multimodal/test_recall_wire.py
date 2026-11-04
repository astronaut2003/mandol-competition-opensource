"""Private HTTP contracts for the candidate method and real provider adapter."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from mandol_competition.inference.evidence_selector import EvidenceSelector
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.multimodal.api import create_app
from .backend_support import request
from .test_grounded_wire import Vision
from .test_partitioned_evidence import Guard
from .test_sqlite_qdrant import image_part


def app_for(backend,monkeypatch,guard):
    guard.close=lambda:None
    monkeypatch.setattr('mandol_competition.multimodal.evidence.create_selector',lambda _:guard)
    config=replace(backend.config,vlm_backend='openai_compat',vlm_base_url='https://synthetic.invalid',
        evidence_policy='source-grounded-recall-v4',vision_failure_policy='retry-required-v1')
    return create_app(config,backend=backend,extractor=Vision())


def test_real_add_replay_user_isolation_options_and_16_parallel_searches(backend,monkeypatch):
    guard=Guard(lambda text:'ORBIT-47' in text)
    body=request(content=[{'type':'text','text':'Owned folder label ORBIT-47.'},image_part('blue')]).model_dump()
    with TestClient(app_for(backend,monkeypatch,guard)) as client:
        first=client.post('/add',json=body)
        assert first.status_code==200 and client.post('/add',json=body).json()==first.json()
        frontier=backend.state.scope_state(backend.scope('alice'))['published_seq']
        query={'user_id':'alice','query':'What printed label is visible?','top_k':100}
        for options in ({},{'options':[]},{'options':['ORBIT-47','ATLAS-92']}):
            response=client.post('/search',json={**query,**options})
            assert response.status_code==200 and len(response.json()['data'])==1
            wire=json.dumps(response.json(),allow_nan=False)
            assert 'ORBIT-47' in wire and image_part('blue')['image_url']['url'] in wire
        with ThreadPoolExecutor(max_workers=16) as workers:
            replies=list(workers.map(lambda _:client.post('/search',json=query),range(16)))
        assert all(r.status_code==200 for r in replies)
        assert all(r.json()==replies[0].json() for r in replies)
        assert client.post('/search',json={**query,'user_id':'bob'}).json()=={'data':[]}
        assert backend.state.scope_state(backend.scope('alice'))['published_seq']==frontier
        assert backend.state._rows('SELECT count(*) n FROM memories')[0]['n']==1


@pytest.mark.parametrize('envelope', [None,[],{'model':'gpt-4o-mini','choices':None},
    {'model':'gpt-4o-mini','choices':[None]},
    {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':None}]},
    {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':[]}]},
    {'model':'gpt-4o-mini','choices':[{'finish_reason':'length','message':{'content':'{"indices":[]}'}}]},
    {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':{'content':'not-json'}}]}])
def test_bad_provider_wire_is_503_once_and_cannot_clear_or_modify_existing_evidence(backend,monkeypatch,envelope):
    class Wire:
        calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            return envelope
        def close(self):pass
    wire=Wire()
    model=HTTPVLMClient('https://synthetic.invalid','gpt-4o-mini',transport=wire)
    selector=EvidenceSelector('https://synthetic.invalid',strict=True,client=model)
    guard=SimpleNamespace(select=selector.select,confirm_absence=selector.select,close=selector.close)
    body=request(content='Owned label ORBIT-47.').model_dump()
    with TestClient(app_for(backend,monkeypatch,guard),raise_server_exceptions=False) as client:
        assert client.post('/add',json=body).status_code==200
        before=backend.state.scope_state(backend.scope('alice'))['published_seq']
        response=client.post('/search',json={'user_id':'alice','query':'Printed folder label?','top_k':100})
        assert response.status_code==503 and wire.calls==1
        assert response.json()=={'detail':{'reason':'dependency unavailable; completion may be unknown'}}
        assert backend.state.scope_state(backend.scope('alice'))['published_seq']==before
        assert backend.state._rows('SELECT count(*) n FROM memories')[0]['n']==1


def test_complete_twice_rejected_unrelated_sources_return_a_valid_empty_response(backend,monkeypatch):
    guard=Guard()
    with TestClient(app_for(backend,monkeypatch,guard)) as client:
        assert client.post('/add',json=request(content='A lake and trees.').model_dump()).status_code==200
        response=client.post('/search',json={'user_id':'alice','query':'Printed folder label?','top_k':100})
        assert response.status_code==200 and response.json()=={'data':[]}
        assert [phase for phase,_ in guard.calls]==['primary','verification']


def test_cross_session_original_can_be_delivered_even_when_missing_from_the_baseline_pool(backend,monkeypatch):
    from mandol_competition.multimodal.backend import canonical_memory
    from .test_source_context import publish
    seed=publish(backend,'seed','Nora Vale selected a folder.',session='choice',entity='Nora Vale')
    publish(backend,'bridge','Nora Vale gave me the folder with label ORBIT-47.',session='gift')
    # Hold the ranked baseline constant so this tests actual supplemental
    # retrieval, not a lucky dense/sparse match that already included both.
    monkeypatch.setattr(backend.retrieval,'search',lambda *a,**k:[(canonical_memory(seed,'alice'),.13)])
    guard=Guard(lambda text:'ORBIT-47' in text)
    with TestClient(app_for(backend,monkeypatch,guard)) as client:
        response=client.post('/search',json={'user_id':'alice','query':'Which folder did she give me later?','top_k':1})
        assert response.status_code==200
        wire=response.json()['data']
        assert len(wire)==1 and wire[0]['id']=='seed' and wire[0]['score']==.13
        text=json.dumps(wire)
        assert 'ORBIT-47' in text and 'Additional source context' in text and 'gift' in text
        assert backend.state._rows('SELECT count(*) n FROM memories')[0]['n']==2


def test_bad_verifier_output_is_an_error_instead_of_an_empty_verdict(backend,monkeypatch):
    class Wire:
        calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            message={'content':'{"indices":[]}'} if self.calls==1 else None
            return {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':message}]}
        def close(self):pass
    wire=Wire()
    selector=EvidenceSelector('https://synthetic.invalid',strict=True,
        client=HTTPVLMClient('https://synthetic.invalid','gpt-4o-mini',transport=wire))
    guard=SimpleNamespace(select=selector.select,confirm_absence=selector.select,close=selector.close)
    with TestClient(app_for(backend,monkeypatch,guard),raise_server_exceptions=False) as client:
        assert client.post('/add',json=request(content='Original fact ORBIT-47.').model_dump()).status_code==200
        reply=client.post('/search',json={'user_id':'alice','query':'Printed label?','top_k':100})
        assert reply.status_code==503 and wire.calls==2
        assert backend.state._rows('SELECT count(*) n FROM memories')[0]['n']==1
