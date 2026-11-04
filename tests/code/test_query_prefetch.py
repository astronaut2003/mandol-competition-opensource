from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.remote_embeddings import RecipeDenseEncoder, prepared_dense_query
from mandol_substrate.state import ProfileMismatch
from mandol_substrate.state import Scope
from mandol_substrate.gates import ScopeGates


class Encoder(RecipeDenseEncoder):
    def __init__(self, barrier=None, error=None):
        self.barrier, self.error, self.calls = barrier, error, 0

    def _encode(self, texts, *, query):
        self.calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if self.error is not None:
            raise self.error
        return np.asarray([[float(len(texts[0]))]], dtype=np.float32)


def test_prefetch_preserves_exact_float32_object_and_does_not_cache_other_requests():
    encoder = Encoder()
    with prepared_dense_query(encoder, 'same query'):
        first = encoder.encode_query('same query')
        assert encoder.encode_query('same query') is first
        assert first.dtype == np.float32 and encoder.calls == 1
        assert encoder.encode_query('different query')[0] == 15
    encoder.encode_query('same query')
    assert encoder.calls == 3


@pytest.mark.parametrize('error', [ProfileMismatch('identity'), RuntimeError('unavailable')])
def test_prefetch_replays_the_original_exception_into_the_original_retrieval_policy(error):
    encoder = Encoder(error=error)
    with prepared_dense_query(encoder, 'query'):
        with pytest.raises(type(error)) as caught:
            encoder.encode_query('query')
        assert caught.value is error and encoder.calls == 1


def test_two_named_v4_requests_overlap_http_and_keep_context_and_search_state_separate():
    encoder = Encoder(barrier=threading.Barrier(2))
    service = CodingMemoryService.__new__(CodingMemoryService)
    service.settings = SimpleNamespace(inference_contract='aml-official-eligible-v1')
    service.quality_candidate = None
    service._encoder, service._lock = encoder, threading.RLock()
    service._ready = True
    service._searches, service._lifecycle = ScopeGates(), ScopeGates()
    service._lifecycle_scope = Scope('code-service-lifecycle', 'instance')
    service.store = SimpleNamespace(snapshot=lambda user: ('complete-v1', []),
                                    scope=lambda user: Scope('fixture', user))
    service._cache = {('user', 'complete-v1'): (SimpleNamespace(raw_channel=object()), {})}
    # A deadlock here would prove query I/O still holds the global search lock.
    service._search = lambda user, query, top_k: encoder.encode_query(query)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda query: service.search('user', query, 100), ['one', 'second']))
    assert [float(v[0]) for v in results] == [3., 6.]
    assert encoder.calls == 2
