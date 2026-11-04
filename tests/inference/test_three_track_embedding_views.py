"""All current track adapters reach the same guarded paired v4 query path."""
from concurrent.futures import ThreadPoolExecutor
import math

import pytest

from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPJointEncoders
from mandol_competition.inference.embedding_query_views import MAX_VIEW_TOKENS, plan_query
from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable
from tests.textual.test_embedding_query_wire import native_encoder
from mandol_competition.inference.embedding_tokenizer import count_tokens


def adapter(track, paired, monkeypatch):
    if track == 'coding':
        from mandol_competition.code.api.remote_embeddings import RecipeNativeEncoder
        return RecipeNativeEncoder('http://unused', clients=paired)
    if track == 'multimodal':
        from mandol_competition.multimodal import inference
        from mandol_competition.multimodal.config import MultimodalConfig
        from mandol_competition.multimodal.models import load_profile
        monkeypatch.setattr(inference, 'HTTPEmbeddingClient', lambda endpoint, recipe, **_: HTTPEmbeddingClient(
            'unused', recipe, transport=paired.dense.http))
        monkeypatch.setattr(inference, 'HTTPSparseEncoderClient', lambda endpoint, recipe, **_: HTTPSparseEncoderClient(
            'unused', recipe, transport=paired.sparse.http))
        config = MultimodalConfig(model_device='external-http', inference_contract='aml-official-eligible-v1')
        result = inference.remote_encoders(config, load_profile(config))
        assert isinstance(result, HTTPJointEncoders)
        return result
    return paired


def encode_query(encoder, text):
    return getattr(encoder, 'encode_query_pair', encoder.encode_query)(text)


@pytest.mark.parametrize('track', ['textual', 'coding', 'multimodal'])
def test_overlong_token_query_is_guarded_at_each_track_entry_without_dropping_tail(track, monkeypatch):
    payloads = []
    native, paired = native_encoder(payloads)
    encoder = adapter(track, paired, monkeypatch)
    text = 'HEAD-KEY\n' + '你' * 33_001 + '\nTAIL-KEY'
    try:
        with pytest.raises(InferenceUnavailable) as raised:
            native.encode([text], query=True)
        assert raised.value.status_code == 400
        payloads.clear()
        dense, sparse = encode_query(encoder, text)
        sent = [p['input']['texts'][0] for p in payloads]
        assert ''.join(sent) == text and set(sparse) == {3, 7}
        assert all(count_tokens(t) <= MAX_VIEW_TOKENS for t in sent)
        assert math.hypot(*dense) == pytest.approx(1., abs=1e-6)
        assert encoder.embedding_query_policy()['query_only']
    finally:
        native.close()


@pytest.mark.parametrize('track', ['textual', 'coding', 'multimodal'])
def test_ordinary_query_exact_pair_and_document_input_are_unchanged(track, monkeypatch):
    payloads = []
    native, paired = native_encoder(payloads)
    encoder = adapter(track, paired, monkeypatch)
    try:
        expected = native.encode([' original query '], query=True)[0]
        payloads.clear()
        dense, sparse = encode_query(encoder, ' original query ')
        assert list(dense) == expected.dense and sparse == expected.sparse
        assert len(payloads) == 1 and payloads[0]['input']['texts'] == [' original query ']
        assert payloads[0]['parameters']['text_type'] == 'query'
        payloads.clear()
        document = 'original source\n' + 'a' * 8500
        encoder.encode_documents([document])
        assert len(payloads) == 1 and payloads[0]['input']['texts'] == [document]
        assert payloads[0]['parameters']['text_type'] == 'document'
    finally:
        native.close()


@pytest.mark.parametrize('track', ['textual', 'coding', 'multimodal'])
def test_each_track_preserves_mid_view_quota_without_returning_partial_pair(track, monkeypatch):
    payloads = []
    native, paired = native_encoder(payloads, failure_at=2)
    encoder = adapter(track, paired, monkeypatch)
    try:
        with pytest.raises(InferenceRateLimited) as raised:
            encode_query(encoder, '你' * 33_001)
        assert raised.value.retry_after == 93 and len(payloads) == 2
    finally:
        native.close()


@pytest.mark.parametrize('track', ['textual', 'coding', 'multimodal'])
def test_each_track_16_concurrent_long_queries_keep_independent_complete_pairs(track, monkeypatch):
    payloads = []
    native, paired = native_encoder(payloads)
    encoder = adapter(track, paired, monkeypatch)
    query = 'HEAD-KEY\n' + '字🙂' * 5500 + '\nTAIL-KEY'
    try:
        with ThreadPoolExecutor(max_workers=16) as workers:
            pairs = list(workers.map(lambda _: encode_query(encoder, query), range(16)))
        assert len(pairs) == 16 and all(set(sparse) == {3, 7} for _, sparse in pairs)
        assert all(math.hypot(*dense) == pytest.approx(1., abs=1e-6) for dense, _ in pairs)
        assert len(payloads) == 16 * len(plan_query(query).views)
        assert all(count_tokens(p['input']['texts'][0]) <= MAX_VIEW_TOKENS for p in payloads)
    finally:
        native.close()
