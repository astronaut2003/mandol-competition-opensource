"""Cross-track HTTP contract regression, using only artificial input/services."""
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from mandol_substrate.state import Conflict,StateUnavailable

SECRET='SYNTHETIC-CREDENTIAL-AND-CONTENT-DO-NOT-LOG'


class Memory:
    ready=False
    result=None
    failure=None
    receipt=1
    ack_override=None
    def __init__(self):
        self.calls=[]
        self.result=[]
        self.store=SimpleNamespace(state=None)
    def start(self): self.ready=True
    def shutdown(self): self.ready=False
    def health(self): return {'status':'ok'}
    def add_messages(self,**kwargs):
        if self.failure: raise self.failure
        return self.receipt
    def search(self,**kwargs):
        self.calls.append(kwargs)
        if self.failure: raise self.failure
        return self.result


@pytest.fixture(params=['textual','code','multimodal'])
def api(request,tmp_path):
    track=request.param;memory=Memory()
    if track=='textual':
        from mandol_competition.textual.api.app import create_app
        from mandol_competition.textual.api.config import Settings
        app=create_app(Settings(competition_run_root=str(tmp_path),evaluation_namespace='synthetic-audit',
            memory_system_key=SECRET,max_request_bytes=4096),memory_service=memory)
    elif track=='code':
        from mandol_competition.code.api.app import create_app
        from mandol_competition.code.api.config import Settings
        app=create_app(Settings(coding_db_path=str(tmp_path/'code.sqlite'),memory_system_key=SECRET,
            max_request_bytes=4096),memory_service=memory)
    else:
        from mandol_competition.multimodal.api import create_app
        from mandol_competition.multimodal.config import MultimodalConfig
        backend=SimpleNamespace(state=None,profile=SimpleNamespace(options={}),start=lambda:None,
            close=lambda:None,health=lambda:{'status':'ok'},purge_idle=lambda _:0)
        app=create_app(MultimodalConfig(data_dir=tmp_path,auth_token=SECRET,rerank_method=None,
            max_request_bytes=4096),backend=backend)
        def mmsearch(body):
            return {'data':memory.search(user_id=body.user_id,query=body.query,top_k=body.top_k,options=body.options)}
        def mmadd(body):
            if memory.failure: raise memory.failure
            return memory.ack_override if memory.ack_override is not None else dict(success=True,
                **{field:getattr(body,field) for field in ('user_id','session_id','request_id')})
        app.state.search_service.handle=mmsearch
        app.state.add_service.handle=mmadd
    stream=io.StringIO();handler=logging.StreamHandler(stream)
    logger=logging.getLogger('mandol_competition.'+track+'.diagnostics');logger.addHandler(handler)
    try:
        with TestClient(app,raise_server_exceptions=False) as client:
            client.headers['X-Api-Key']=SECRET
            until=time.monotonic()+5
            while client.get('/health').status_code!=200:
                assert time.monotonic()<until;time.sleep(.01)
            yield track,client,memory,stream
    finally: logger.removeHandler(handler)


def body():
    return dict(user_id='synthetic-user',session_id='synthetic-session',request_id='synthetic-request',
        messages=[dict(role='user',content=SECRET)])


def search(client,**update):
    return client.post('/search',json=dict(user_id='synthetic-user',query=SECRET,top_k=100)|update)


@pytest.mark.parametrize('fields',[{}, {'score':None},{'score':0},{'score':-3.5},
    {'created_at':'2026-11-01T00:00:00Z'},{'score':.9,'created_at':None}])
def test_optional_wire_fields_are_optional_in_all_tracks(api,fields):
    _,client,memory,stream=api
    memory.result=[dict(id='id',content='Artificial evidence',**fields)]
    response=search(client)
    assert response.status_code==200 and response.json()['data'][0]['id']=='id'
    assert all(value is not None for value in response.json()['data'][0].values())
    assert not stream.getvalue()


@pytest.mark.parametrize('score',[True,False,'0.5',float('nan'),float('inf')])
def test_non_numeric_or_nonfinite_internal_score_is_not_success(api,score):
    _,client,memory,stream=api
    memory.result=[dict(id='id',content=SECRET,score=score)]
    response=search(client)
    assert response.status_code==500 and SECRET not in response.text+stream.getvalue()
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['stage']=='response' and row['exception_type']=='ValidationError' and 'stack' not in row


def test_excess_results_do_not_get_silently_cropped_or_returned_200(api):
    _,client,memory,stream=api
    memory.result=[dict(id=str(i),content=SECRET) for i in range(2)]
    response=search(client,top_k=1)
    assert response.status_code==500
    assert json.loads(stream.getvalue().splitlines()[-1])['exception_type']=='OutputContractError'
    assert SECRET not in response.text+stream.getvalue()


@pytest.mark.parametrize('failure,status',[(ValueError,500),(RuntimeError,500),(StateUnavailable,503),(Conflict,409)])
def test_internal_and_domain_exceptions_never_echo_secret_messages(api,failure,status):
    track,client,memory,stream=api;memory.failure=failure(SECRET)
    response=search(client)
    assert response.status_code==status and SECRET not in response.text+stream.getvalue()
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['event']==track+'_failure' and row['track']==track
    assert row['correlation_id']==response.headers['X-Request-ID']


@pytest.mark.parametrize('route',['/add','/search'])
def test_provider_rate_limit_is_preserved_by_actual_track_routes(api,route):
    from mandol_competition.inference.errors import InferenceRateLimited
    _,client,memory,stream=api
    memory.failure=InferenceRateLimited(SECRET,retry_after=12.1,
        provider_error_code='Throttling.AllocationQuota',provider_request_id='provider-artificial-123')
    response=client.post(route,json=body() if route=='/add' else
                         dict(user_id='synthetic-user',query=SECRET,top_k=100))
    assert response.status_code==429 and response.headers['Retry-After']=='13'
    assert SECRET not in response.text+stream.getvalue()
    row=json.loads(stream.getvalue().splitlines()[-1])
    assert row['upstream_http_status']==429
    assert row['provider_error_code']=='Throttling.AllocationQuota'
    assert row['provider_request_id']=='provider-artificial-123'
    assert row['correlation_id']==response.headers['X-Request-ID']


@pytest.mark.parametrize('auth',['Token','Bearer','X-Api-Key'])
def test_all_auth_styles_independently_and_health_is_public(api,auth):
    _,client,_,_=api
    client.headers.pop('X-Api-Key')
    assert client.get('/health').status_code==200
    assert search(client).status_code==401
    headers={'X-Api-Key':SECRET} if auth=='X-Api-Key' else {'Authorization':auth+' '+SECRET}
    response=client.post('/search',headers=headers,json=dict(user_id='synthetic-user',query='Artificial query',top_k=100))
    assert response.status_code==200 and response.json()=={'data':[]}


@pytest.mark.parametrize('patch',[{'top_k':True},{'top_k':0},{'top_k':'100'},{'options':{}}, {'query':''}])
def test_client_validation_is_structured_and_private(api,patch):
    _,client,_,stream=api
    response=search(client,**patch)
    assert response.status_code==422
    errors=response.json()['detail']['errors']
    assert errors and all(set(error)=={'loc','type'} for error in errors)
    assert SECRET not in response.text+stream.getvalue()


@pytest.mark.parametrize('timestamp',[-(2**63)-1,2**63])
def test_timestamp_overflow_is_rejected_before_business_layer(api,timestamp):
    _,client,_,_=api
    value=body();value['messages'][0]['timestamp']=timestamp
    response=client.post('/add',json=value)
    assert response.status_code==422


def test_add_ids_are_echoed_exactly(api):
    _,client,_,_=api
    value=body()|dict(user_id=' user:甲 ',request_id=' r:/ ',session_id=' s:β ')
    response=client.post('/add',json=value)
    assert response.status_code==200 and response.json()==dict(success=True,
        **{field:value[field] for field in ('user_id','request_id','session_id')})


@pytest.mark.parametrize('bad',[None,False,True,0,2])
def test_unconfirmed_add_cannot_be_acknowledged(api,bad):
    track,client,memory,_=api
    if track=='multimodal':
        memory.ack_override=dict(success=1 if bad is True else bad,user_id='synthetic-user',session_id='synthetic-session',request_id='synthetic-request')
    else: memory.receipt=bad
    assert client.post('/add',json=body()).status_code==500


def test_wrong_add_echo_is_an_internal_error(api):
    track,client,memory,_=api
    if track!='multimodal':pytest.skip('text/code construct echoes from validated request')
    memory.ack_override=dict(success=True,user_id='wrong-user',request_id='synthetic-request',session_id='synthetic-session')
    assert client.post('/add',json=body()).status_code==500


def test_body_limit_is_explicit_413_and_never_truncates(api):
    _,client,_,stream=api
    value=body();value['messages'][0]['content']='Artificial text '*1000
    response=client.post('/add',json=value)
    assert response.status_code==413 and SECRET not in response.text+stream.getvalue()


def test_multimodal_parts_order_and_pixels_are_preserved(api):
    track,client,memory,_=api
    if track!='multimodal':pytest.skip('multimodal media only')
    stream=io.BytesIO();Image.new('RGB',(2,2),'blue').save(stream,format='PNG')
    uri='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()
    parts=[dict(type='text',text='Before image'),dict(type='image_url',image_url=dict(url=uri)),dict(type='text',text='After image')]
    memory.result=[dict(id='image',content=parts)]
    response=search(client)
    assert response.status_code==200 and response.json()['data'][0]['content']==parts


@pytest.mark.parametrize('uri',['https://example.invalid/PRIVATE','data:image/png;base64,AA==',
    'data:image/png;base64,not-base64','data:text/plain;base64,YQ=='])
def test_bad_internal_response_media_never_becomes_http_200(api,uri):
    track,client,memory,stream=api
    if track!='multimodal':pytest.skip('multimodal media only')
    memory.result=[dict(id='id',content=[dict(type='image_url',image_url=dict(url=uri))])]
    response=search(client)
    assert response.status_code==500 and uri not in response.text+stream.getvalue()


def test_parallel_ids_are_separate_and_safe(api):
    _,client,memory,stream=api;memory.failure=ValueError(SECRET)
    with ThreadPoolExecutor(max_workers=16) as pool:
        responses=list(pool.map(lambda _:search(client),range(16)))
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows)==16 and len({row['correlation_id'] for row in rows})==16
    assert all(response.status_code==500 for response in responses)
    assert SECRET not in stream.getvalue()
