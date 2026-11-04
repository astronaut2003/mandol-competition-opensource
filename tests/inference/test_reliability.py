from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mandol_competition.inference.budget import SharedBudget, Reliability, estimate_tokens
from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.replicas import ReplicaReranker
from mandol_competition.runtime.http_boundary import install_http_boundary


class Clock:
    def __init__(self):
        self.value = 1000.
        self.waits = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds


def windows():
    return {'rerank': dict(usage_limit=90, usage_period=6., request_limit=30, request_period=6.),
            'embedding': dict(usage_limit=120, usage_period=60., request_limit=10, request_period=1.)}


def budget(tmp_path, clock):
    return SharedBudget(tmp_path/'admission.sqlite', windows(),
                        clock=clock, monotonic=clock, sleep=clock.sleep)


def test_process_instances_share_real_six_second_window_and_models_are_independent(tmp_path):
    clock = Clock()
    first, second = budget(tmp_path, clock), budget(tmp_path, clock)
    first.acquire('rerank', 90, until=1100.)
    second.acquire('embedding', 120, until=1100.)
    assert not clock.waits
    second.acquire('rerank', 90, until=1100.)
    assert clock.waits == [6.]
    first.acquire('embedding', 120, until=1100.)
    assert clock.value == 1060.


def test_cooldown_is_shared_and_never_shortened(tmp_path):
    clock = Clock()
    first, second = budget(tmp_path, clock), budget(tmp_path, clock)
    first.defer('rerank', 120.)
    second.acquire('rerank', 1, until=1200.)
    assert clock.waits == [120.]


def test_budget_deadline_does_not_sleep_or_dispatch_early(tmp_path):
    clock = Clock(); gate = budget(tmp_path, clock)
    gate.acquire('rerank', 90, until=1100.)
    with pytest.raises(InferenceRateLimited) as raised:
        gate.acquire('rerank', 90, until=1001.)
    assert raised.value.retry_after == 6.
    assert not clock.waits


def test_oversized_estimate_is_exclusive_not_a_payload_rejection(tmp_path):
    clock = Clock(); gate = budget(tmp_path, clock)
    gate.acquire('rerank', 100000, until=1100.)
    gate.acquire('rerank', 1, until=1100.)
    assert clock.waits == [6.]


def test_budget_cannot_silently_fall_back_when_state_is_unavailable(tmp_path):
    gate = SharedBudget(tmp_path/'missing'/'budget.sqlite', windows())
    with pytest.raises(InferenceUnavailable, match='budget_store'):
        gate.acquire('rerank', 1, until=__import__('time').monotonic()+1)


def test_actual_separate_processes_cannot_each_spend_the_same_window(tmp_path):
    path = str(tmp_path/'shared.sqlite')
    script = '''import json,sys,time
from mandol_competition.inference.budget import SharedBudget
b=SharedBudget(sys.argv[1],{'m':{'usage_limit':1,'usage_period':.12,'request_limit':1,'request_period':.12}})
b.acquire('m',1,until=time.monotonic()+10)
print(time.time())
'''
    processes = [subprocess.Popen([sys.executable, '-c', script, path],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    stamps = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 0, stderr
        stamps.append(float(stdout))
    stamps.sort()
    assert all(b-a >= .09 for a, b in zip(stamps, stamps[1:]))


def test_budget_charges_query_per_document_without_changing_input():
    value = {'model':'m','input':{'query':'人工','documents':['one','two']}}
    original = json.dumps(value)
    expected = 2*(len('人工'.encode())+64)+len(b'one')+len(b'two')
    assert estimate_tokens('m', value) == expected
    assert json.dumps(value) == original


def policy_stub(monkeypatch, *, attempts=4):
    policy = SimpleNamespace(seconds=10., attempts=attempts, model=lambda endpoint, payload: None)
    monkeypatch.setattr('mandol_competition.inference.budget.from_environment', lambda:policy)


def test_more_retries_preserve_payload_and_idempotency_key(monkeypatch):
    policy_stub(monkeypatch)
    seen, waits = [], []
    payload = {'model':'frozen','input':{'query':'synthetic','documents':['original evidence']}}
    def reply(request):
        seen.append(request)
        return httpx.Response(429, headers={'Retry-After':'2'}, json={'code':'Throttling.AllocationQuota',
            'request_id':'request-123','message':'DO_NOT_LOG_PRIVATE_TEXT'}) if len(seen)<4 else httpx.Response(200,json={'ok':True})
    client = HTTPTransport('https://provider.test', max_attempts=2, timeout=1,
                           transport=httpx.MockTransport(reply), sleep=waits.append)
    try:
        assert client.request('POST','/rerank',payload,retry_unknown=False) == {'ok':True}
        assert len(seen) == 4 and waits == [2.,2.,2.]
        assert len({r.headers['Idempotency-Key'] for r in seen}) == 1
        assert all(json.loads(r.content) == payload for r in seen)
    finally:
        client.close()


def test_proxy_does_not_multiply_provider_retries_and_preserves_metadata(monkeypatch):
    policy_stub(monkeypatch)
    seen=[]
    def reply(request):
        seen.append(request)
        return httpx.Response(429,headers={'Retry-After':'17','X-Mandol-Upstream-Code':'Throttling.AllocationQuota',
                                           'X-Mandol-Upstream-Request-ID':'provider-123'})
    client = HTTPTransport('http://127.0.0.1:18620',transport=httpx.MockTransport(reply))
    try:
        with pytest.raises(InferenceRateLimited) as raised:
            client.request('POST','/v1/encode',{'model':'text-embedding-v4'})
        assert len(seen) == 1
        assert raised.value.retry_after == 17
        assert raised.value.provider_error_code == 'Throttling.AllocationQuota'
        assert raised.value.provider_request_id == 'provider-123'
        assert float(seen[0].headers['X-Mandol-Deadline-Seconds']) > 0
    finally:
        client.close()


def test_missing_retry_after_defaults_to_sixty_and_error_body_is_not_exposed():
    client=HTTPTransport('https://provider.test',max_attempts=1,
        transport=httpx.MockTransport(lambda r:httpx.Response(429,json={
            'code':'Throttling.AllocationQuota','request_id':'provider-123','message':'PRIVATE_QUERY_API_KEY'})))
    try:
        with pytest.raises(InferenceRateLimited) as raised:
            client.request('POST','/rerank',{})
        assert raised.value.retry_after == 60
        assert raised.value.provider_error_code == 'Throttling.AllocationQuota'
        assert 'PRIVATE' not in str(raised.value)
    finally:
        client.close()


@pytest.mark.parametrize('track',['textual','code','multimodal'])
@pytest.mark.parametrize('delay,expected',[(None,'60'),(2.1,'3'),(120,'120')])
def test_http_boundary_keeps_429_and_retry_after_for_all_tracks(track,delay,expected):
    app=FastAPI()
    @app.post('/search')
    def search():
        raise InferenceRateLimited(retry_after=delay,provider_error_code='Throttling.AllocationQuota',
                                   provider_request_id='provider-123')
    install_http_boundary(app,track,ready=lambda:True)
    with TestClient(app,raise_server_exceptions=False) as client:
        response=client.post('/search',json={'query':'PRIVATE_QUERY'})
    assert response.status_code==429
    assert response.headers['Retry-After']==expected
    assert response.json()=={'detail':{'reason':'upstream rate limit exceeded'}}
    assert response.headers['X-Request-ID']
    assert 'PRIVATE' not in response.text


def test_true_dependency_outage_remains_503():
    app=FastAPI()
    @app.post('/search')
    def search(): raise InferenceUnavailable('offline')
    install_http_boundary(app,'textual',ready=lambda:True)
    with TestClient(app,raise_server_exceptions=False) as client:
        assert client.post('/search').status_code==503


class LimitedReplica:
    expected_profile={'model':'frozen'}
    def __init__(self): self.limited=False
    def validate_profile(self): return True
    def health(self): return True
    def rerank(self,query,documents):
        if self.limited:
            raise InferenceRateLimited(retry_after=60,provider_error_code='Throttling.AllocationQuota',
                                       provider_request_id='provider-123')
        return [1.]*len(documents)
    def close(self): pass


def test_replica_cooldown_does_not_erase_quota_reason_into_503():
    replica=LimitedReplica();pool=ReplicaReranker([replica],monitor=False)
    pool.validate_profile();replica.limited=True
    try:
        for _ in range(2):
            with pytest.raises(InferenceRateLimited) as raised:
                pool.rerank('synthetic',['original evidence'])
            assert raised.value.status_code==429
            assert 0<raised.value.retry_after<=60
            assert raised.value.provider_error_code=='Throttling.AllocationQuota'
        assert pool.snapshot()['replicas'][0]['errors']==1
        assert not any(isinstance(v,BaseException) for v in vars(pool._replicas[0]).values())
    finally:
        pool.close()


def test_vlm_quota_rejection_does_not_publish_a_degraded_success():
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatExtractor,VisionRequestError
    class Client:
        def complete(self,**kwargs):
            raise VisionRequestError('safe',retryable=False,status_code=429,retry_after=12,retry_managed=True)
    extractor=OpenAICompatExtractor(model_name='gpt-4o-mini',config=MultimodalConfig(),client=Client())
    with pytest.raises(InferenceRateLimited) as raised:
        extractor.extract(image_data_uri='synthetic',context_text='synthetic',image_position=0,image_count=1)
    assert raised.value.retry_after==12


def test_transports_share_policy_windows_without_changing_dispatched_payload(tmp_path,monkeypatch):
    clock=Clock()
    policy=Reliability({'state_path':str(tmp_path/'unused.sqlite'),
        'provider_origins':['https://provider.test'],'models':windows()})
    policy.budget=budget(tmp_path,clock)
    monkeypatch.setattr('mandol_competition.inference.budget.from_environment',lambda:policy)
    stamps=[]
    value={'model':'rerank','input':{'query':'synthetic','documents':['evidence']}}
    def reply(request):
        assert json.loads(request.content)==value
        stamps.append(clock.value)
        return httpx.Response(200,json={'ok':True})
    clients=[HTTPTransport('https://provider.test',transport=httpx.MockTransport(reply)) for _ in range(2)]
    try:
        for client in clients:assert client.request('POST','/rerank',value)=={'ok':True}
        assert stamps==[1000.,1006.]
        assert policy.model('http://127.0.0.1:18620',value) is None
        assert policy.model('https://other.test',value) is None
    finally:
        for client in clients:client.close()


def test_failed_cooldown_persistence_does_not_replace_real_provider_429(monkeypatch):
    def unavailable(*args):raise InferenceUnavailable('provider_budget_store_unavailable')
    policy=SimpleNamespace(seconds=10.,attempts=4,model=lambda *args:'rerank',
        budget=SimpleNamespace(acquire=lambda *args,**kwargs:None,defer=unavailable))
    monkeypatch.setattr('mandol_competition.inference.budget.from_environment',lambda:policy)
    client=HTTPTransport('https://provider.test',transport=httpx.MockTransport(
        lambda _:httpx.Response(429,headers={'Retry-After':'12'})))
    try:
        with pytest.raises(InferenceRateLimited) as caught:client.request('POST','/rerank',{})
        assert caught.value.retry_after==12
    finally:client.close()
