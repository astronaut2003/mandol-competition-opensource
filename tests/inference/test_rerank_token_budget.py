import json
import math
from types import SimpleNamespace
import httpx
import pytest

from mandol_competition.inference import rerank_query_views as guard
from mandol_competition.inference.rerank_tokenizer import encode_text
from mandol_competition.inference.dashscope_rerank import DashScopeRerankerClient,ENDPOINT
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.errors import InferenceInputRejected,InferenceUnavailable

def count(text):return len(encode_text(text))

def test_official_item_pair_and_request_budgets_reserve_200():
    assert guard.MAX_VIEW_TOKENS==30000-200
    assert guard.MAX_PAIR_TOKENS==32768-200
    assert guard.MAX_REQUEST_TOKENS==120000-200
    text='你'*29801
    plan=guard.plan_query(text)
    assert ''.join(plan.parts)==text and all(count(p)<=29800 for p in plan.parts)

def test_character_cap_also_protects_low_token_54k_input():
    text='Orion '*8997
    assert count(text)<29800 and len(text)>50000
    plan=guard.request_plan(text,['original evidence'])
    assert len(plan)==2 and ''.join(v.query for v in plan)==text
    assert all(len(v.query)<=49800 and v.documents==('original evidence',) for v in plan)

def test_query_is_shortened_for_entire_pool_request_cost(monkeypatch):
    monkeypatch.setattr(guard,'MAX_REQUEST_TOKENS',100)
    query='query '*50;docs=['doc A','doc B','doc C']
    plan=guard.request_plan(query,docs)
    assert len(plan)>1 and ''.join(v.query for v in plan)==query
    assert all(v.documents==tuple(docs) for v in plan)
    assert all(count(v.query)*3+sum(map(count,v.documents))<=100 for v in plan)

def test_pair_window_is_protected_in_addition_to_each_item():
    query='你'*29800;docs=['好'*29800]
    plan=guard.request_plan(query,docs)
    assert len(plan)>1 and ''.join(v.query for v in plan)==query
    assert all(count(v.query)+count(v.documents[0])<=32568 for v in plan)

def test_document_overflow_retains_every_candidate_and_text(monkeypatch):
    monkeypatch.setattr(guard,'MAX_VIEW_TOKENS',30)
    monkeypatch.setattr(guard,'MAX_REQUEST_TOKENS',100)
    monkeypatch.setattr(guard,'MAX_PAIR_TOKENS',50)
    docs=[label+' '+'alpha '*70+' END'+label for label in ('A','B','C')]
    plan=guard.request_plan('query',docs)
    assert len(plan)>1 and all(len(v.documents)==3 for v in plan)
    for index,original in enumerate(docs):
        assert ''.join(v.documents[index] for v in plan).startswith(original)
    assert all(count(v.query)*3+sum(map(count,v.documents))<=100 for v in plan)
    assert all(count(v.query)+max(map(count,v.documents))<=50 for v in plan)

def test_rrf_is_invariant_to_per_request_score_scaling_and_ties():
    query='x'*50001;docs=['A','B','C']
    def run(rows):
        it=iter(rows)
        return guard.rerank_query_views(SimpleNamespace(rerank=lambda q,d:next(it)),query,docs)
    a=run([[.99,.98,.01],[.02,.03,.01]])
    b=run([[.8,.6,.2],[.5,.9,.1]])
    assert a==b and a[0]==a[1]>a[2]
    assert all(math.isfinite(s) and 0<=s<=1 for s in a)

def native(handler):
    transport=HTTPTransport('https://rerank.invalid',transport=httpx.MockTransport(handler),max_attempts=1)
    return DashScopeRerankerClient(ENDPOINT,transport=transport)

def response(count):
    return httpx.Response(200,json={'output':{'results':[{'index':i,'relevance_score':.9/(i+1)} for i in range(count)]}})

def test_native_shared_client_guards_every_track_without_outer_wrapper():
    calls=[]
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        assert len(body['input']['query'])<=49800
        return response(len(body['input']['documents']))
    client=native(handler)
    try:
        assert client.score('Orion '*8997,['original evidence'])==[1.]
        assert len(calls)==2
    finally:client.close()

def test_only_recognized_length_400_replans_with_smaller_complete_views():
    calls=[]
    def handler(request):
        body=json.loads(request.content);calls.append(body)
        if len(body['input']['query'])>25000:
            return httpx.Response(400,json={'code':'InvalidParameter','message':'Range of input length should be [1, 25000]'})
        return response(len(body['input']['documents']))
    client=native(handler)
    try:
        assert client.score('你'*26000,['original evidence'])==[1.]
        assert len(calls)==3 and ''.join(b['input']['query'] for b in calls[1:])=='你'*26000
    finally:client.close()

@pytest.mark.parametrize('status,message',[(400,'unsupported parameter'),(503,'private document body')])
def test_other_errors_do_not_replay_paid_or_invalid_operations(status,message):
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(status,json={'code':'InvalidParameter','message':message})
    client=native(handler)
    try:
        with pytest.raises(InferenceUnavailable) as raised:client.score('query',['original evidence'])
        assert len(calls)==1 and message not in str(raised.value)
        if status==400:assert isinstance(raised.value,InferenceInputRejected)
    finally:client.close()
