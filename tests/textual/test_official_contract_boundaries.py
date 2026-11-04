"""Official wire invariants and the structural failures found in the audit."""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from pathlib import Path
import re
import threading
import time
import traceback

from fastapi.testclient import TestClient
import pytest

from mandol_competition.observability.diagnostics import current_context,diagnostic_call
from mandol_competition.textual.api.app import create_app
from mandol_competition.textual.api.config import Settings
from mandol_competition.runtime.contract import EXECUTION_TOP_K_LIMIT
from mandol_competition.runtime.http_boundary import DiagnosticContextMiddleware,SanitizedResponseFailure
from mandol_substrate.state import Conflict,StateUnavailable

SECRET='PRIVATE-SYNTHETIC-CONTENT-AND-CREDENTIAL'


class Memory:
    ready=False
    result=None
    failure=None
    receipt=1
    calls=None
    def start(self): self.ready=True
    def shutdown(self): self.ready=False
    def health(self): return True
    def add_messages(self,**kwargs): return self.receipt
    def search(self,**kwargs):
        self.calls.append(kwargs)
        if self.failure:
            def fail(): raise self.failure
            return diagnostic_call('reranker',fail)
        return self.result


@pytest.fixture
def api(tmp_path):
    memory=Memory();memory.result=[];memory.calls=[]
    settings=Settings(competition_run_root=str(tmp_path),evaluation_namespace='synthetic-contract-audit',
        memory_system_key=SECRET)
    app=create_app(settings,memory_service=memory)
    stream=io.StringIO();handler=logging.StreamHandler(stream)
    logger=logging.getLogger('mandol_competition.textual.diagnostics');logger.addHandler(handler)
    request_stream=io.StringIO();request_handler=logging.StreamHandler(request_stream)
    from mandol_competition.observability.runtime import SafeJSONFormatter
    request_handler.setFormatter(SafeJSONFormatter())
    requests=logging.getLogger('mandol_competition.requests');requests.addHandler(request_handler)
    requests.setLevel(logging.INFO)
    try:
        with TestClient(app,raise_server_exceptions=False) as client:
            client.headers.update({'X-Api-Key':SECRET,'X-Request-ID':'untrusted-caller-value'})
            until=time.monotonic()+5
            while client.get('/health').status_code!=200:
                assert time.monotonic()<until;time.sleep(.01)
            yield client,memory,stream,request_stream
    finally:
        logger.removeHandler(handler);requests.removeHandler(request_handler)


def search(client,**overrides):
    return client.post('/search',json=dict(user_id='synthetic-user',query=SECRET,top_k=100)|overrides)


@pytest.mark.parametrize('optional', [{},{'score':None},{'score':0},{'score':-1.5},
    {'created_at':'2026-11-01T00:00:00Z'},{'score':.7,'created_at':None}])
def test_optional_response_fields_are_optional(api,optional):
    client,memory,stream,_=api
    memory.result=[dict(id='synthetic-id',content='artificial evidence',**optional)]
    response=search(client)
    assert response.status_code==200
    row=response.json()['data'][0]
    assert row['id']=='synthetic-id' and row['content']=='artificial evidence'
    assert all(value is not None for value in row.values())
    assert not stream.getvalue()


@pytest.mark.parametrize('bad',[True,False,'0.5','NaN',float('inf'),float('nan'),{},[]])
def test_invalid_score_is_not_coerced_into_success(api,bad):
    client,memory,stream,_=api
    memory.result=[dict(id='id',content=SECRET,score=bad)]
    response=search(client)
    assert response.status_code==500
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['exception_type']=='ValidationError' and row['stage']=='response'
    assert 'stack' not in row and row['validation'][0]['loc']==['data',0,'score']
    assert SECRET not in response.text+stream.getvalue()


@pytest.mark.parametrize('hits',[None,{},(),['invalid'],[dict(id='a',content='artificial'),dict(id='b',content='artificial')]])
def test_invalid_container_or_excess_topk_fails_before_200(api,hits):
    client,memory,stream,_=api;memory.result=hits
    response=search(client,top_k=1)
    assert response.status_code==500
    assert json.loads(stream.getvalue().splitlines()[-1])['exception_type']=='OutputContractError'


@pytest.mark.parametrize('top_k',[1,100,101,10000,10**100])
@pytest.mark.parametrize('options',['absent',None,[],['First artificial option','Second artificial option']])
def test_output_limit_and_execution_budget_are_separate_and_choices_forwarded(api,top_k,options):
    client,memory,_,_=api
    memory.result=[dict(id='id',content='artificial')]
    response=search(client,top_k=top_k,**({} if options=='absent' else {'options':options}))
    assert response.status_code==200
    assert len(response.json()['data'])<=top_k
    call=memory.calls[-1]
    assert call['top_k']==min(top_k,EXECUTION_TOP_K_LIMIT)
    assert call['query']==SECRET and call['choices']==(None if options=='absent' else options)


@pytest.mark.parametrize('receipt',[None,False,True,0,2,-1,'1'])
def test_add_requires_a_matching_completion_receipt(api,receipt):
    client,memory,stream,_=api;memory.receipt=receipt
    response=client.post('/add',json=dict(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content=SECRET)]))
    assert response.status_code==500
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['stage']=='add_receipt' and row['exception_type']=='OutputContractError'
    assert SECRET not in response.text+stream.getvalue()


def test_add_echoes_exact_identifiers_after_valid_completion(api):
    client,_,_,_=api
    body=dict(user_id=' user:甲 ',request_id=' request:/r ',session_id=' session:β ',
        messages=[dict(role='user',content='Artificial memory')])
    response=client.post('/add',json=body)
    assert response.status_code==200 and response.json()==dict(success=True,**{key:body[key] for key in ('user_id','request_id','session_id')})


@pytest.mark.parametrize('timestamp',[-(2**63)-1,2**63])
def test_timestamp_storage_overflow_is_client_validation_before_any_add(api,timestamp):
    client,_,stream,_=api
    response=client.post('/add',json=dict(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content=SECRET,timestamp=timestamp)]))
    assert response.status_code==422
    errors=response.json()['detail']['errors']
    assert errors[0]['loc']==['body','messages',0,'timestamp']
    assert set(errors[0])=={'loc','type'} and SECRET not in response.text+stream.getvalue()


@pytest.mark.parametrize('failure,status,classification',[(ValueError,500,'server'),(RuntimeError,500,'server'),
    (StateUnavailable,503,'server'),(Conflict,409,'client')])
def test_single_trusted_id_and_correct_http_error_class(api,failure,status,classification):
    client,memory,stream,requests=api;memory.failure=failure(SECRET)
    response=search(client)
    assert response.status_code==status
    row=json.loads(stream.getvalue().splitlines()[-1])
    cid=response.headers['X-Request-ID']
    assert re.fullmatch('[a-f0-9]{32}',cid) and row['correlation_id']==cid and row['stage']=='reranker'
    request=[json.loads(line) for line in requests.getvalue().splitlines() if json.loads(line).get('operation')=='search'][-1]
    assert request['correlation_id']==cid and request['error_class']==classification
    assert SECRET not in response.text+stream.getvalue()+requests.getvalue()


def test_explicit_cause_preserves_original_type_and_stack_without_content(api):
    client,memory,stream,_=api
    def fail(**kwargs):
        try: raise ValueError(SECRET)
        except ValueError as cause: raise StateUnavailable(SECRET) from cause
    memory.search=fail
    assert search(client).status_code==503
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['exception_type']=='StateUnavailable'
    assert row['causes'][0]['exception_type']=='ValueError'
    assert row['causes'][0]['stack'][-1]['function']=='fail'
    assert SECRET not in stream.getvalue()


def test_partial_response_failure_is_signalled_and_sanitized():
    async def broken(scope,receive,send):
        await send(dict(type='http.response.start',status=200,headers=[]))
        await send(dict(type='http.response.body',body=b'{',more_body=True))
        raise RuntimeError(SECRET)
    messages=[]
    async def send(message): messages.append(message)
    async def receive(): return dict(type='http.request',body=b'',more_body=False)
    with pytest.raises(SanitizedResponseFailure) as failure:
        asyncio.run(DiagnosticContextMiddleware(broken)(dict(type='http',path='/search'),receive,send))
    formatted=''.join(traceback.format_exception(failure.value))
    assert SECRET not in formatted and 'response failed after headers' in formatted
    assert len(messages)==2 and messages[-1]['more_body'] is True
    assert current_context() is None


def test_simultaneous_diagnostics_do_not_leak_between_requests(api):
    client,memory,stream,_=api;memory.failure=ValueError(SECRET)
    with ThreadPoolExecutor(max_workers=16) as pool:
        responses=list(pool.map(lambda _:search(client),range(16)))
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert all(response.status_code==500 for response in responses)
    assert len(rows)==16 and len({row['correlation_id'] for row in rows})==16
    assert {response.headers['X-Request-ID'] for response in responses}=={row['correlation_id'] for row in rows}
    assert current_context() is None and SECRET not in stream.getvalue()


def test_domain_diagnostics_and_retrieval_have_no_http_adapter_import():
    root=Path(__file__).parents[2]/'src/mandol_competition'
    for file in (root/'observability/diagnostics.py',root/'textual/core/retrieval.py'):
        for node in ast.walk(ast.parse(file.read_text())):
            names=([node.module or ''] if isinstance(node,ast.ImportFrom) else
                [item.name for item in node.names] if isinstance(node,ast.Import) else [])
            assert all(not name.startswith(('fastapi','starlette')) and 'api.diagnostics' not in name for name in names)
