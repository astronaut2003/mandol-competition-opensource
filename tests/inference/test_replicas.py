from concurrent.futures import ThreadPoolExecutor
import threading
import time

import pytest

from mandol_substrate.state import ProfileMismatch
from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.replicas import ReplicaReranker, replica_endpoints


class Replica:
    expected_profile = {'model': 'frozen', 'score': 'p1-p0'}

    def __init__(self):
        self.calls = []
        self.down = self.wrong = False
        self.block = None

    def validate_profile(self):
        if self.wrong:
            raise ProfileMismatch('wrong model')
        self.health()

    def health(self):
        if self.down:
            raise InferenceUnavailable('offline')
        return True

    def rerank(self, query, documents):
        self.health()
        if self.block:
            self.block.wait(2)
        self.calls.append((query, list(documents)))
        return [float(i) / 10 for i in range(len(documents))]

    def attestation(self):
        return self.expected_profile

    def close(self):
        pass


def admitted(clients=None, **kwargs):
    clients = clients or [Replica(), Replica(), Replica()]
    pool = ReplicaReranker(clients, monitor=False, **kwargs)
    pool.validate_profile()
    for client in clients:
        client.calls.clear()
    return pool, clients


def test_each_search_stays_on_one_replica_and_idle_ties_rotate():
    pool, clients = admitted()
    try:
        for _ in range(9):
            assert pool.rerank('q', ['a', 'b', 'c']) == [0., .1, .2]
        assert [len(c.calls) for c in clients] == [3, 3, 3]
        assert all(docs == ['a', 'b', 'c'] for c in clients for _, docs in c.calls)
    finally:
        pool.close()


def test_partial_failure_discards_scores_and_rescores_complete_list():
    pool, clients = admitted()
    def partial(query, documents):
        clients[0].calls.append((query, [documents[0]]))
        raise InferenceUnavailable('second_pair_failed')
    clients[0].rerank = partial
    try:
        assert pool.rerank('q', ['a', 'b', 'c']) == [0., .1, .2]
        assert clients[0].calls == [('q', ['a'])]
        assert clients[1].calls == [('q', ['a', 'b', 'c'])]
        assert clients[2].calls == []
        assert pool.snapshot()['ready_replicas'] == 2
        assert pool.snapshot()['failovers'] == 1
    finally:
        pool.close()


@pytest.mark.parametrize('down', [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)])
def test_any_one_or_two_failures_leave_an_identical_replica(down):
    pool, clients = admitted()
    try:
        for i in down:
            clients[i].down = True
            pool._probe(i)
        assert pool.rerank('q', ['a', 'b']) == [0., .1]
        assert pool.snapshot()['ready_replicas'] == 3 - len(down)
        assert pool.snapshot()['capacity_degraded']
    finally:
        pool.close()


def test_all_down_fails_without_unbounded_retry_and_recovery_warms_before_rejoin():
    pool, clients = admitted()
    try:
        for i, c in enumerate(clients):
            c.down = True
            pool._probe(i)
        start = time.monotonic()
        with pytest.raises(InferenceUnavailable, match='all_reranker'):
            pool.score('q', ['a'])
        assert time.monotonic() - start < .1
        clients[2].down = False
        pool._probe(2)
        assert clients[2].calls == [('memory', ['memory health probe'])]
        assert pool.rerank('q', ['a']) == [0.]
    finally:
        pool.close()


def test_model_drift_quarantines_and_does_not_hide_identity_failure():
    pool, clients = admitted()
    def drift(query, documents):
        raise ProfileMismatch('wrong response model')
    clients[0].rerank = drift
    try:
        with pytest.raises(ProfileMismatch):
            pool.rerank('q', ['a'])
        assert pool.snapshot()['replicas'][0]['quarantined']
        assert clients[1].calls == clients[2].calls == []
        assert pool.rerank('q', ['a']) == [0.]
    finally:
        pool.close()


def test_least_inflight_and_bounded_backpressure():
    pool, clients = admitted(max_inflight=1, max_queue=0)
    released = threading.Event()
    for c in clients:
        c.block = released
    try:
        with ThreadPoolExecutor(3) as executor:
            jobs = [executor.submit(pool.rerank, 'q', ['a']) for _ in range(3)]
            until = time.monotonic() + 1
            while time.monotonic() < until and sum(r['inflight'] for r in pool.snapshot()['replicas']) != 3:
                time.sleep(.005)
            assert [r['inflight'] for r in pool.snapshot()['replicas']] == [1, 1, 1]
            with pytest.raises(InferenceUnavailable, match='backpressure'):
                pool.rerank('q', ['a'])
            released.set()
            assert all(job.result() == [0.] for job in jobs)
        assert pool.snapshot()['rejected'] == 1
        assert pool.snapshot()['queue_depth'] == 0
    finally:
        released.set()
        pool.close()


def test_monitor_obeys_cooldown_then_recovers():
    clients = [Replica(), Replica()]
    pool = ReplicaReranker(clients, cooldown=.05, probe_interval=.02, probe_timeout=.2)
    pool.validate_profile()
    try:
        clients[0].down = True
        pool._probe(0)
        clients[0].down = False
        time.sleep(.01)
        assert pool.snapshot()['ready_replicas'] == 1
        until = time.monotonic() + 1
        while time.monotonic() < until and pool.snapshot()['ready_replicas'] != 2:
            time.sleep(.01)
        assert pool.snapshot()['ready_replicas'] == 2
    finally:
        pool.close()


def test_different_recipe_and_duplicate_endpoints_fail_closed(monkeypatch):
    clients = [Replica(), Replica()]
    clients[1].expected_profile = {'model': 'other'}
    with pytest.raises(ProfileMismatch):
        ReplicaReranker(clients)
    monkeypatch.setenv('MANDOL_TEXTUAL_RERANKER_REPLICAS', 'http://10.77.0.9:8003,http://10.77.0.9:8003')
    with pytest.raises(ValueError):
        replica_endpoints('textual', 'http://10.77.0.9:8003')
