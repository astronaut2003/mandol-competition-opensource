"""Actual HTTP acceptance of the published Code memory-api-v1.1 text forms."""
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from mandol_competition.code.api.app import create_app
from mandol_competition.code.api.config import Settings


class Memory:
    ready = False
    def __init__(self):
        self.add_calls=[]
        self.search_calls=[]
        self.store=SimpleNamespace(state=None)
    def start(self): self.ready=True
    def shutdown(self): self.ready=False
    def add_messages(self,**value):
        self.add_calls.append(value)
        return len(value['messages'])
    def search(self,**value):
        self.search_calls.append(value)
        return [{'id':'stable','content':'Synthetic Code evidence'}]


@pytest.fixture
def api(tmp_path):
    memory=Memory()
    app=create_app(Settings(coding_db_path=str(tmp_path/'owned.sqlite'),memory_system_key='synthetic-only'),memory_service=memory)
    with TestClient(app,raise_server_exceptions=False) as client:
        until=time.monotonic()+3
        while client.get('/health').status_code!=200:
            assert time.monotonic()<until
            time.sleep(.01)
        client.headers['X-Api-Key']='synthetic-only'
        yield client,memory


@pytest.mark.parametrize('content,expected',[
    ('  original\n代码🙂  ','  original\n代码🙂  '),
    ('[]','[]'),
    ([{'type':'text','text':'  first\n原文 '},{'type':'text','text':'second🙂  '}],'  first\n原文 \nsecond🙂  '),
    ([{'type':'text','text':'one'}],'one'),
])
def test_add_and_query_accept_both_official_text_forms_without_losing_order(api,content,expected):
    client,memory=api
    value={'request_id':' 请求 ','session_id':' 会话 ','user_id':' 用户 ',
        'messages':[{'role':'user','content':content,'timestamp':1704067200000}]}
    response=client.post('/add',json=value)
    assert response.status_code==200
    assert response.json()=={'success':True,**{k:value[k] for k in ('request_id','session_id','user_id')}}
    assert memory.add_calls[-1]['messages']==[{'role':'user','content':expected,'timestamp':1704067200000}]
    response=client.post('/search',json={'user_id':value['user_id'],'query':content,'options':['A','B'],'top_k':100})
    assert response.status_code==200 and response.json()=={'data':[{'id':'stable','content':'Synthetic Code evidence'}]}
    assert memory.search_calls[-1]=={'user_id':value['user_id'],'query':expected,'top_k':100}


@pytest.mark.parametrize('bad',[
    [],[{'type':'text','text':''}],[{'type':'text','text':'  '}],
    [{'type':'image_url','image_url':{'url':'data:image/png;base64,AA=='}}],
    [{'text':'missing discriminator'}],[{'type':'text'}],
    [{'type':'text','text':False}],[{'type':'text','text':'ok'},None],
    {'type':'text','text':'not an array'},
])
def test_invalid_or_image_parts_are_client_errors_before_code_business_logic(api,bad):
    client,memory=api
    add=client.post('/add',json={'request_id':'r','session_id':'s','user_id':'u','messages':[{'role':'user','content':bad}]})
    search=client.post('/search',json={'user_id':'u','query':bad,'top_k':100})
    assert add.status_code==search.status_code==422
    assert not memory.add_calls and not memory.search_calls
    for response in (add,search):
        assert all(set(item)=={'loc','type'} for item in response.json()['detail']['errors'])
