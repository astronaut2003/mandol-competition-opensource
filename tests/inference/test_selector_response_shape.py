"""Synthetic provider envelope failures; no network, credentials or corpus."""
import pytest
from mandol_competition.inference.evidence_selector import EvidenceSelector
from mandol_competition.inference.errors import InferenceSchemaError
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.runtime.http_boundary import exception_status

class Wire:
    def __init__(self,message):self.message=message;self.calls=0
    def request(self,*args,**kwargs):
        self.calls+=1
        return {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':self.message}]}
    def close(self):pass

def selector(message,strict):
    wire=Wire(message)
    model=HTTPVLMClient('https://synthetic.invalid','gpt-4o-mini',transport=wire)
    return EvidenceSelector('https://synthetic.invalid',strict=strict,client=model),wire

def invoke(client):
    return client.select(query='Synthetic label?',options=[],candidates=[{'text':'Synthetic label ORBIT-47'}],limit=1)

@pytest.mark.parametrize('message',[None,[],42,'invalid-envelope'])
def test_strict_malformed_message_is_typed_503_not_internal_500(message):
    client,wire=selector(message,True)
    try:
        with pytest.raises(InferenceSchemaError) as raised:invoke(client)
        assert exception_status(raised.value)==503 and wire.calls==1
    finally:client.close()

@pytest.mark.parametrize('message',[None,[],42,'invalid-envelope'])
def test_nonstrict_malformed_message_keeps_existing_baseline_contract(message):
    client,wire=selector(message,False)
    try:
        result=invoke(client)
        assert result.status=='fallback' and result.indices==(0,) and wire.calls==1
    finally:client.close()

@pytest.mark.parametrize('indices',[[0],[]])
def test_normal_selection_and_absence_contract_unchanged(indices):
    import json
    client,wire=selector({'content':json.dumps({'indices':indices})},True)
    try:
        result=invoke(client)
        assert result.status=='selected' and result.indices==tuple(indices) and wire.calls==1
    finally:client.close()

@pytest.mark.parametrize('reply',[None,[],{}, {'model':'gpt-4o-mini','choices':[]},
    {'model':'gpt-4o-mini','choices':[None]},
    {'model':'gpt-4o-mini','choices':[{'finish_reason':'stop','message':{'content':'{"indices":[true]}'}}]}])
def test_all_malformed_envelope_levels_are_typed_without_an_attribute_or_index_error(reply):
    from types import SimpleNamespace
    direct=SimpleNamespace(complete=lambda *a,**kw:reply)
    client=EvidenceSelector('https://synthetic.invalid',strict=True,client=direct)
    from mandol_competition.inference.errors import InferenceIdentityError
    with pytest.raises((InferenceSchemaError,InferenceIdentityError)):
        invoke(client)
