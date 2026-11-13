"""No paid idle traffic; recovery uses only actual requests after cooldown."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import httpx
import pytest

from mandol_competition.inference.dashscope_rerank import DashScopeRerankerClient
from mandol_competition.inference.errors import InferenceUnavailable, InferenceRateLimited
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.replicas import create_pool


class Provider:
    def __init__(self):
        self.calls=[];self.status=200;self.block=None;self.entered=threading.Event()
    def reply(self,request):
        payload=json.loads(request.content);self.calls.append(payload['input'])
        if self.block:
            self.entered.set();assert self.block.wait(3)
        if self.status!=200:return httpx.Response(self.status,headers={'Retry-After':'60'})
        count=len(payload['input']['documents'])
        return httpx.Response(200,json={'output':{'results':[
            {'index':i,'relevance_score':1/(i+1)} for i in range(count)]}})


def build(provider,**options):
    wire=HTTPTransport('https://unit.test',max_attempts=1,transport=httpx.MockTransport(provider.reply))
    client=DashScopeRerankerClient(transport=wire)
    return create_pool([client],options),client


def test_paid_factory_disables_background_and_health_remains_passive_after_expiry():
    provider=Provider();pool,client=build(provider)
    try:
        pool.validate_profile()
        assert len(provider.calls)==1  # Explicit startup admission only.
        assert pool._thread is None
        assert pool.snapshot()['background_probes'] is False
        assert pool.snapshot()['request_driven_recovery'] is True
        client._last_success-=3600
        for _ in range(100):assert client.health() and pool.health()
        assert len(provider.calls)==1
        assert pool.rerank('real query',['first original','second original']) == [1.,.5]
        assert provider.calls[-1]=={'query':'real query','documents':['first original','second original']}
        assert len(provider.calls)==2
    finally:pool.close()


@pytest.mark.parametrize('status',[429,503])
def test_cooldown_has_no_probe_and_recovery_dispatches_the_real_query_once(status):
    provider=Provider();pool,client=build(provider)
    try:
        pool.validate_profile();provider.status=status
        with pytest.raises(InferenceUnavailable):pool.rerank('failed real query',['original'])
        count=len(provider.calls)
        assert pool._replicas[0].retry_at>time.monotonic()
        with pytest.raises(InferenceUnavailable):pool.rerank('before cooldown',['original'])
        with pytest.raises(InferenceUnavailable):client.health()
        assert len(provider.calls)==count
        provider.status=200
        with pool._condition:pool._replicas[0].retry_at=time.monotonic()-1
        assert pool.rerank('recovery real query',['kept original']) == [1.]
        assert len(provider.calls)==count+1
        assert provider.calls[-1]=={'query':'recovery real query','documents':['kept original']}
        assert pool.health() and client.health()
        assert pool.snapshot()['ready_replicas']==1
        assert pool.snapshot()['replicas'][0]['error_class']=='none'
    finally:pool.close()


def test_half_open_recovery_admits_only_one_real_request():
    provider=Provider();pool,client=build(provider,max_inflight=4,max_queue=0)
    release=threading.Event()
    try:
        pool.validate_profile();provider.status=503
        with pytest.raises(InferenceUnavailable):pool.rerank('failed',['old'])
        with pool._condition:pool._replicas[0].retry_at=time.monotonic()-1
        provider.status=200;provider.block=release
        with ThreadPoolExecutor(1) as worker:
            first=worker.submit(pool.rerank,'first recovery',['original'])
            assert provider.entered.wait(2)
            count=len(provider.calls)
            with pytest.raises(InferenceUnavailable,match='backpressure'):
                pool.rerank('second recovery',['other'])
            assert len(provider.calls)==count
            release.set();assert first.result()==[1.]
        assert pool.snapshot()['ready_replicas']==1
    finally:release.set();pool.close()


def test_a_never_admitted_paid_replica_still_fails_closed():
    provider=Provider();provider.status=503;pool,client=build(provider)
    try:
        with pytest.raises(InferenceUnavailable):pool.validate_profile()
        count=len(provider.calls)
        provider.status=200
        with pool._condition:pool._replicas[0].retry_at=time.monotonic()-1
        with pytest.raises(InferenceUnavailable):pool.rerank('not admitted',['original'])
        assert len(provider.calls)==count and pool._thread is None
    finally:pool.close()


def test_nonpaid_local_replica_keeps_existing_readiness_monitor():
    from .test_replicas import Replica
    pool=create_pool([Replica()],{'probe_interval':.02,'probe_timeout':.1})
    try:
        pool.validate_profile()
        assert pool._thread is not None and pool._thread.is_alive()
        assert pool.snapshot()['background_probes'] is True
        assert pool.snapshot()['request_driven_recovery'] is False
    finally:pool.close()
