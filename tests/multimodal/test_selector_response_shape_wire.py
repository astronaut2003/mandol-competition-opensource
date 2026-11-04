"""A malformed synthetic provider response stays a typed HTTP dependency error."""
from dataclasses import replace
from types import SimpleNamespace
from fastapi.testclient import TestClient
import pytest
from mandol_competition.inference.evidence_selector import EvidenceSelector
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.multimodal.api import create_app
from .backend_support import request
from .test_grounded_wire import Vision

@pytest.mark.parametrize('message',[None,[]])
def test_current_delivery_search_invalid_provider_message_is_503_and_keeps_memory(backend,monkeypatch,message):
    class Wire:
        calls=0
        def request(self,*args,**kwargs):
            self.calls+=1
            return {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':message}]}
        def close(self):pass
    wire=Wire()
    model=HTTPVLMClient('https://synthetic.invalid','gpt-4o-mini',transport=wire)
    selector=EvidenceSelector('https://synthetic.invalid',strict=True,client=model)
    guard=SimpleNamespace(select=selector.select,confirm_absence=selector.select,close=selector.close)
    monkeypatch.setattr('mandol_competition.multimodal.evidence.create_selector',lambda _:guard)
    config=replace(backend.config,vlm_backend='openai_compat',vlm_base_url='https://synthetic.invalid',
        evidence_policy='source-grounded-delivery-v3',vision_failure_policy='retry-required-v1')
    body=request(content='Synthetic folder label ORBIT-47.').model_dump()
    with TestClient(create_app(config,backend=backend,extractor=Vision()),raise_server_exceptions=False) as client:
        assert client.post('/add',json=body).status_code==200
        reply=client.post('/search',json={'user_id':body['user_id'],'query':'Which folder label?','top_k':100})
        assert reply.status_code==503
        assert reply.json()=={'detail':{'reason':'dependency unavailable; completion may be unknown'}}
        assert wire.calls==1
        assert backend.state._rows('SELECT count(*) n FROM memories')[0]['n']==1
        assert client.post('/add',json=body).status_code==200
