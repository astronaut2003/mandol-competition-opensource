import time

from fastapi.testclient import TestClient

from mandol_competition.code.api.app import create_app
from mandol_competition.code.api.config import Settings


def test_known_preparation_limit_is_client_error_and_conflict_keeps_priority(tmp_path):
    app=create_app(Settings(coding_memory=True,coding_db_path=str(tmp_path/'state.sqlite3'),memory_system_key='fixture-budget-key'))
    with TestClient(app) as client:
        until=time.monotonic()+5
        while client.get('/health').status_code!=200:
            assert time.monotonic()<until
            time.sleep(.01)
        memory=app.state.memory_service
        bound=memory.store.coordinator.max_mutation_memories
        original=dict(user_id='budget-user',session_id='s',request_id='ordinary',messages=[dict(role='user',content='  exact JWT issuer raw source\n')])
        headers={'X-Api-Key':'fixture-budget-key'}
        assert client.post('/add',json=original,headers=headers).status_code==200
        large=dict(original,request_id='over-budget',messages=[dict(role='user',content='synthetic record')]*(bound+1))
        response=client.post('/add',json=large,headers=headers)
        assert response.status_code==413 and response.json()['detail']['reason']=='service request memory budget exceeded'
        assert len(memory.store.snapshot('budget-user')[1])==1
        assert memory.store.state.request(memory.store.scope('budget-user'),'over-budget') is None
        assert client.post('/add',json=dict(large,request_id='ordinary'),headers=headers).status_code==409
        assert client.post('/add',json=original,headers=headers).status_code==200
        result=client.post('/search',json=dict(user_id='budget-user',query='JWT issuer',top_k=100),headers=headers)
        assert result.status_code==200 and result.json()['data']
