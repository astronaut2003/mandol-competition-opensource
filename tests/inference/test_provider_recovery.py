"""Recovery needs no new user dispatch; polls and shutdown remain bounded."""
import importlib
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient

from mandol_competition.inference.dashscope import EncodedPair, MODEL, official_embedding_recipes
from mandol_competition.inference.errors import InferenceUnavailable


def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'services/inference/src'))
    return importlib.import_module('mandol_inference.dashscope_bridge')


class Provider:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.closed = False

    def encode(self, texts, *, query=False):
        assert not self.closed
        self.calls.append(query)
        if self.fail:
            raise InferenceUnavailable('http_503', status_code=503)
        return [EncodedPair([1.]+[0.]*1023, {123: .5}) for _ in texts]

    def close(self):
        self.closed = True


def test_health_only_recovers_both_modes_and_healthy_polls_are_free(monkeypatch):
    module = bridge(monkeypatch)
    provider = Provider()
    dense, sparse = official_embedding_recipes()
    with TestClient(module.create_app(lambda: provider, recovery_delay=.05)) as client:
        for _ in range(20): assert client.get('/health').status_code==200
        assert provider.calls==[False,True]
        provider.fail=True
        body={'model':MODEL,'input':['evidence'],'input_type':'query',
              'dense_recipe_hash':dense.hash,'sparse_recipe_hash':sparse.hash}
        assert client.post('/v1/encode',json=body).status_code==503
        assert client.get('/health').status_code==503
        provider.fail=False
        until=time.monotonic()+2
        while client.get('/health').status_code!=200 and time.monotonic()<until: time.sleep(.01)
        assert client.get('/health').status_code==200
        assert provider.calls==[False,True,True,False,True]
        metrics=client.get('/ops/state').json()['provider_recovery']
        assert metrics['recovery_probes']==metrics['recoveries']==1
        for _ in range(20): assert client.get('/health').status_code==200
        assert len(provider.calls)==5
    assert provider.closed


def test_retry_after_does_not_trigger_early_recovery_and_stop_interrupts_it(monkeypatch):
    module=bridge(monkeypatch); provider=Provider()
    worker=module.ProviderRecovery(provider,delay=.01)
    worker.failed(InferenceUnavailable('http_429',retry_after=120.))
    time.sleep(.04)
    assert provider.calls==[] and worker.snapshot()['next_probe_seconds']>119
    before=time.monotonic(); worker.close()
    assert time.monotonic()-before<.2


def test_old_failed_probe_cannot_override_new_success(monkeypatch):
    module=bridge(monkeypatch); entered=threading.Event(); release=threading.Event()
    class Blocking(Provider):
        def encode(self, texts, *, query=False):
            entered.set(); assert release.wait(2)
            raise InferenceUnavailable('old_probe')
    worker=module.ProviderRecovery(Blocking(),delay=.01)
    try:
        worker.failed(InferenceUnavailable('initial'))
        assert entered.wait(1)
        worker.succeeded()
        release.set()
        time.sleep(.04)
        assert worker.snapshot()['available'] is True and worker.snapshot()['failures']==0
    finally:
        release.set(); worker.close()
