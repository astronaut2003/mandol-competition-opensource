from contextvars import ContextVar
from threading import Barrier
import pytest
from mandol_competition.multimodal.inference import ParallelHTTPEncoders


def test_parallel_http_keeps_context_values_and_waits_for_both_channels():
    barrier = Barrier(2)
    context = ContextVar('request_test', default=None)
    context.set('same-request')
    calls, closed = [], []
    class Dense:
        def embed(self, texts, query=False):
            barrier.wait(timeout=2)
            calls.append(('dense', context.get(), query))
            return [[1., 0.] for _ in texts]
        def close(self): closed.append('dense')
    class Sparse:
        def encode_documents(self, texts):
            barrier.wait(timeout=2)
            calls.append(('sparse', context.get(), None))
            return [{2: 1.} for _ in texts]
        def encode_query(self, text): return self.encode_documents([text])[0]
        def close(self): closed.append('sparse')
    encoders = ParallelHTTPEncoders(Dense(), Sparse())
    try:
        assert encoders.encode_query('q') == ([1., 0.], {2: 1.})
        assert encoders.encode_documents(['a', 'b']) == [([1., 0.], {2: 1.})] * 2
        assert all(c[1] == 'same-request' for c in calls)
    finally:
        encoders.close()
    assert set(closed) == {'dense', 'sparse'}


def test_parallel_failure_does_not_publish_partial_pair():
    barrier = Barrier(2)
    finished = []
    class Dense:
        def embed(self, texts, query=False):
            barrier.wait(timeout=2)
            raise ValueError('invalid vector')
        def close(self): pass
    class Sparse:
        def encode_query(self, text):
            barrier.wait(timeout=2)
            finished.append(True)
            return {2: 1.}
        def close(self): pass
    encoders = ParallelHTTPEncoders(Dense(), Sparse())
    try:
        with pytest.raises(ValueError, match='invalid vector'):
            encoders.encode_query('q')
        assert finished == [True]
    finally:
        encoders.close()
