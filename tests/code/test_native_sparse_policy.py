from dataclasses import replace
import random

import numpy as np
import pytest

from mandol_competition.code.api.config import Settings
from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.remote_embeddings import RecipeNativeEncoder, prepared_dense_query
from mandol_competition.code.native_sparse import NativeSparseCodec, POLICY, fuse
from mandol_competition.code.policies import CodeProcessor, retrieval_profile
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.state import CanonicalMemory, Scope, StateUnavailable
from mandol_substrate.vector.codec import MinimalProjectionCodec


def test_runtime_fusion_keeps_entire_tail_and_single_channel_identity():
    rng = random.Random(42)
    for count in (1, 23, 150, 320):
        dense = [(str(i), rng.random()) for i in range(count)]
        sparse = [(str(i), rng.random() + .01) for i in reversed(range(count))]
        result = fuse(dense, sparse)
        assert len(result) == count
        assert len({uid for uid, _ in result}) == count
        assert fuse(dense, []) == dense
        assert set(u for u, _ in fuse(dense, sparse)) == set(u for u, _ in dense)


def test_runtime_fusion_reciprocal_rank_example():
    # Equal first/second positions tie; deterministic source identity breaks ties.
    result = fuse([('a', 9.), ('b', 8.)], [('b', 1.), ('a', .5)])
    assert [uid for uid, _ in result] == ['a', 'b']
    assert result[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert result[1][1] == pytest.approx(1 / 61 + 1 / 62)


def test_native_projection_has_independent_identity_without_changing_c0():
    profile = retrieval_profile(CodeProcessor(), 'aml-official-eligible-v1')
    policy = QdrantProjectionPolicy(schema='mandol-qdrant-projection-dot-v1', dense_metric='Dot')
    legacy, native = MinimalProjectionCodec(profile, policy.hash), NativeSparseCodec(profile, policy.hash)
    scope, memory = Scope('code', 'u'), CanonicalMemory('raw', '  exact source\n', commit_seq=1)
    old, new = legacy.metadata(scope, memory), native.metadata(scope, memory)
    assert old['physical_id'] != new['physical_id']
    assert old['projection_hash'] != new['projection_hash']
    assert legacy.profile_hash == native.profile_hash == profile.hash
    assert native.metadata(scope, memory) == new
    assert native.metadata(Scope('code', 'other'), memory)['physical_id'] != new['physical_id']
    assert native.metadata(scope, replace(memory, version=2))['physical_id'] != new['physical_id']
    with pytest.raises(StateUnavailable):
        native.encode(new, [1.] + [0.] * 1023, {})


def test_paired_query_prefetch_is_exactly_one_call_and_context_local():
    class Endpoint:
        def validate_profile(self): return True
    class Clients:
        dense, sparse = Endpoint(), Endpoint()
        calls = 0
        def encode_query(self, text):
            self.calls += 1
            return [1.] + [0.] * 1023, {17: .25}
    clients = Clients()
    encoder = RecipeNativeEncoder('http://unused', clients=clients)
    with prepared_dense_query(encoder, 'query'):
        pair = encoder.encode_query_pair('query')
        assert pair[1] == {17: .25}
        assert np.array_equal(encoder.encode_query('query'), pair[0])
        assert clients.calls == 1
    encoder.encode_query_pair('query')
    assert clients.calls == 2


@pytest.mark.parametrize('contract,mode', [('vllm-http-v1', 'raw'), ('aml-official-eligible-v1', 'hybrid')])
def test_native_policy_cannot_adopt_another_recipe(tmp_path, contract, mode):
    with pytest.raises(ValueError, match='named-v4 raw'):
        CodingMemoryService(Settings(coding_db_path=str(tmp_path / 'state.sqlite3'),
            inference_contract=contract, coding_retrieval_mode=mode, coding_raw_policy=POLICY))
