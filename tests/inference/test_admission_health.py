"""Admission pressure is not evidence that an upstream provider is unhealthy."""
import importlib
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from mandol_competition.inference.budget import Reliability,SharedBudget
from mandol_competition.inference.dashscope import EncodedPair,MODEL as EMBEDDING,official_embedding_recipes
from mandol_competition.inference.dashscope_rerank import DashScopeRerankerClient,MODEL as RERANK
from mandol_competition.inference.errors import (
    InferenceAdmissionDeferred,InferenceDeadlineExceeded,InferenceRateLimited,InferenceUnavailable)
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.replicas import ReplicaReranker


def test_real_six_second_budget_does_not_fail_a_three_second_health_probe(tmp_path):
    seen=[]
    def reply(request):
        import json
        payload=json.loads(request.content);seen.append(payload['model'])
        count=len(payload['input']['documents'])
        return httpx.Response(200,json={'output':{'results':[
            {'index':i,'relevance_score':1/(i+1)} for i in range(count)]}})
    wire=HTTPTransport('https://cn-beijing.maas.qianwenaiapi.com',transport=httpx.MockTransport(reply))
    client=DashScopeRerankerClient(transport=wire)
    pool=ReplicaReranker([client],monitor=False,probe_timeout=3)
    try:
        pool.validate_profile()
        policy=Reliability({'state_path':str(tmp_path/'budget.sqlite'),
            'provider_origins':['https://cn-beijing.maas.qianwenaiapi.com'],
            'models':{RERANK:dict(usage_limit=720000,usage_period=6,request_limit=240,request_period=6)}})
        policy.budget.acquire(RERANK,720000,until=time.monotonic()+10)
        wire.reliability=policy
        client._last_success=time.monotonic()-61
        last_success=client._last_success;before=len(seen)
        pool._probe(0)
        assert pool.health() is True and len(seen)==before
        row=pool.snapshot()['replicas'][0]
        assert row['ready'] and row['errors']==0 and row['admission_deferrals']==0
        assert client._last_success==last_success
    finally:pool.close()


class Replica:
    expected_profile={'model':'frozen'}
    error=None
    def validate_profile(self):return self.health()
    def health(self):
        if self.error:raise self.error
        return True
    def rerank(self,query,docs):
        self.health();return [1.]*len(docs)
    def close(self):pass


@pytest.mark.parametrize('error',[InferenceAdmissionDeferred(retry_after=6),InferenceDeadlineExceeded()])
def test_business_admission_failure_does_not_evict_a_healthy_replica(error):
    replica=Replica();pool=ReplicaReranker([replica],monitor=False)
    try:
        pool.validate_profile();replica.error=error
        with pytest.raises(type(error)):pool.rerank('artificial',['artificial evidence'])
        assert pool.health() is True
        row=pool.snapshot()['replicas'][0]
        assert row['errors']==0 and row['admission_deferrals']==1
        assert pool.snapshot()['failovers']==0
    finally:pool.close()


def test_local_deferral_cannot_restore_a_genuinely_failed_replica():
    replica=Replica();pool=ReplicaReranker([replica],monitor=False)
    try:
        pool.validate_profile()
        replica.error=InferenceRateLimited(retry_after=60,provider_request_id='artificial-provider-id')
        with pytest.raises(InferenceRateLimited):pool.rerank('artificial',['artificial'])
        replica.error=InferenceAdmissionDeferred(retry_after=6)
        pool._probe(0)
        row=pool.snapshot()['replicas'][0]
        assert not row['ready'] and row['errors']==1 and row['admission_deferrals']==1
        assert pool._replicas[0].provider_request_id=='artificial-provider-id'
        with pytest.raises(InferenceUnavailable):pool.health()
    finally:pool.close()


def test_startup_deferral_cannot_admit_a_never_verified_replica():
    replica=Replica();replica.error=InferenceAdmissionDeferred(retry_after=6)
    pool=ReplicaReranker([replica],monitor=False)
    try:
        with pytest.raises(InferenceUnavailable):pool.validate_profile()
        row=pool.snapshot()['replicas'][0]
        assert not row['ready'] and row['errors']==0 and row['admission_deferrals']==1
    finally:pool.close()


class Provider:
    def __init__(self):self.error=None;self.calls=0
    def encode(self,texts,*,query=False):
        if self.error:raise self.error
        self.calls+=1
        return [EncodedPair([1.]+[0.]*1023,{123:.5}) for _ in texts]
    def close(self):pass


@pytest.mark.parametrize('error,status',[(InferenceAdmissionDeferred(retry_after=6),429),
                                        (InferenceDeadlineExceeded(),503)])
def test_embedding_bridge_preserves_health_when_a_request_cannot_dispatch(error,status):
    module=importlib.import_module('mandol_inference.dashscope_bridge')
    provider=Provider();dense,sparse=official_embedding_recipes()
    body={'model':EMBEDDING,'input':['artificial'],'input_type':'query',
          'dense_recipe_hash':dense.hash,'sparse_recipe_hash':sparse.hash}
    with TestClient(module.create_app(lambda:provider,recovery_delay=.01)) as client:
        provider.error=error
        response=client.post('/v1/encode',json=body)
        assert response.status_code==status
        if status==429:assert response.headers['Retry-After']=='6'
        assert client.get('/health').status_code==200
        state=client.get('/ops/state').json()['provider_recovery']
        assert state['failures']==0 and state['admission_deferrals']==1
        assert state['recovery_probes']==0 and state['next_probe_seconds'] is None
        assert provider.calls==2  # Original warmup only; deferred calls cannot spend tokens.
        provider.error=InferenceRateLimited(retry_after=120)
        assert client.post('/v1/encode',json=body).status_code==429
        assert client.get('/health').status_code==503
        provider.error=error
        assert client.post('/v1/encode',json=body).status_code==status
        assert client.get('/health').status_code==503  # Do not hide the real prior failure.
        state=client.get('/ops/state').json()['provider_recovery']
        assert state['failures']==1 and state['next_probe_seconds']>119
        provider.error=None
        assert client.post('/v1/encode',json=body).status_code==200
        assert client.get('/health').status_code==200


def test_recovery_local_deferral_does_not_increase_provider_failure_backoff():
    module=importlib.import_module('mandol_inference.dashscope_bridge')
    provider=Provider();provider.error=InferenceAdmissionDeferred(retry_after=6)
    worker=module.ProviderRecovery(provider,delay=.01)
    try:
        worker.failed(InferenceUnavailable('real_prior_failure'))
        end=time.monotonic()+2
        while worker.snapshot()['admission_deferrals']==0 and time.monotonic()<end:time.sleep(.005)
        state=worker.snapshot()
        assert not state['available'] and state['failures']==1
        assert state['admission_deferrals']==1 and state['next_probe_seconds']>5
    finally:worker.close()


def test_local_deferral_does_not_invalidate_an_inflight_real_recovery_success():
    module=importlib.import_module('mandol_inference.dashscope_bridge')
    entered,release=threading.Event(),threading.Event()
    class Blocking(Provider):
        def encode(self,texts,*,query=False):
            entered.set();assert release.wait(2)
            return super().encode(texts,query=query)
    worker=module.ProviderRecovery(Blocking(),delay=.01)
    try:
        worker.failed(InferenceUnavailable('prior_failure'));assert entered.wait(1)
        worker.deferred(InferenceAdmissionDeferred(retry_after=6))
        release.set();end=time.monotonic()+2
        while not worker.snapshot()['available'] and time.monotonic()<end:time.sleep(.005)
        state=worker.snapshot()
        assert state['available'] and state['recoveries']==1 and state['failures']==0
    finally:release.set();worker.close()


@pytest.mark.parametrize('failure',['quota','connect'])
def test_later_local_wait_never_hides_a_real_error_from_an_earlier_attempt(monkeypatch,failure):
    acquisitions=[];requests=[]
    def acquire(*args,**kwargs):
        acquisitions.append(1)
        if len(acquisitions)>1:raise InferenceAdmissionDeferred(retry_after=6)
    policy=SimpleNamespace(seconds=10,attempts=8,model=lambda *args:RERANK,
        budget=SimpleNamespace(acquire=acquire,defer=lambda *args:None))
    monkeypatch.setattr('mandol_competition.inference.budget.from_environment',lambda:policy)
    def reply(request):
        requests.append(1)
        if failure=='connect':raise httpx.ConnectError('artificial failure',request=request)
        return httpx.Response(429,headers={'Retry-After':'0'},json={'code':'Throttling.AllocationQuota'})
    wire=HTTPTransport('https://provider.test',transport=httpx.MockTransport(reply),sleep=lambda _:None)
    try:
        with pytest.raises(InferenceUnavailable) as raised:wire.request('POST','/rerank',{},retry_unknown=False)
        assert not isinstance(raised.value,InferenceAdmissionDeferred)
        if failure=='quota':
            assert isinstance(raised.value,InferenceRateLimited)
            assert raised.value.provider_error_code=='Throttling.AllocationQuota'
        else:assert raised.value.error_class=='ConnectError'
        assert len(requests)==1 and len(acquisitions)==2
    finally:wire.close()
