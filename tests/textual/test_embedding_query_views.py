import math
import time

import pytest

from mandol_competition.inference.errors import (
    InferenceDeadlineExceeded, InferenceInputRejected, InferenceRateLimited,
    InferenceSchemaError, InferenceUnavailable,
)
from mandol_competition.inference import embedding_query_views as views
from mandol_competition.inference.embedding_tokenizer import count_tokens
from mandol_substrate.deadlines import deadline_budget


class Encoder:
    def __init__(self, outcomes):
        self.outcomes, self.calls = iter(outcomes), []
    def encode_query(self, query):
        self.calls.append(query)
        result = next(self.outcomes)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.mark.parametrize('query', [' raw\n query ', '你' * 8000,
    '<|im_start|>' * 8000, 'Artificial memory alpha beta gamma. ' * 1000,
    '蓝色笔记本位于第三层架子。' * 1000,
    'a' * 33001, '🙂' * 2000, 'e\u0301' * 5000], ids=[f'fits-{i}' for i in range(8)])
def test_token_fitting_queries_keep_original_text_and_exact_provider_pair(query):
    pair = ([.6, .8], {1: .123, 99: .456})
    encoder = Encoder([pair])
    assert count_tokens(query) <= views.MAX_VIEW_TOKENS
    assert views.encode_query_views(encoder, query) is pair
    assert encoder.calls == [query] and views.plan_query(query).parts == (query,)


@pytest.mark.parametrize('query', ['x ' * 16001, '你' * 8001,
    'HEAD\n' + '中文🙂 punctuation.\n' * 5000 + '\nTAIL',
    '🙂' * 9000, 'e\u0301' * 10000,
    '<|im_start|>' * 8001, ' first\n' + '字🙂' * 5500 + '\nlast '], ids=[f'long-{i}' for i in range(7)])
def test_split_retains_original_unicode_and_every_actual_payload_fits(query):
    plan = views.plan_query(query)
    assert ''.join(plan.parts) == query
    assert 1 < len(plan.parts) <= views.MAX_QUERY_VIEWS
    assert all(0 < count_tokens(part) <= 8000 for part in plan.parts)
    assert tuple(map(count_tokens, plan.parts)) == plan.token_counts
    assert all(part.encode('utf-8').decode('utf-8') == part for part in plan.parts)


def controlled_plan(monkeypatch, parts=('a', 'b'), counts=(1, 1)):
    monkeypatch.setattr(views, 'plan_query', lambda _: views.QueryPlan(parts, counts))


def test_weights_use_tokens_instead_of_language_dependent_byte_lengths(monkeypatch):
    parts = ('你' * 8, 'alphabet')
    assert tuple(map(count_tokens, parts)) == (8, 1)
    controlled_plan(monkeypatch, parts, (8, 1))
    encoder = Encoder([([1., 0.], {1: 6.}), ([0., 1.], {1: 3., 2: 9.})])
    dense, sparse = views.encode_query_views(encoder, ''.join(parts))
    assert dense == pytest.approx([8 / math.sqrt(65), 1 / math.sqrt(65)])
    assert sparse == pytest.approx({1: 51 / 9, 2: 1.})
    assert math.hypot(*dense) == pytest.approx(1.)
    assert encoder.calls == list(parts)


def test_zero_dense_cancellation_is_explicit_failure(monkeypatch):
    controlled_plan(monkeypatch)
    encoder = Encoder([([1., 0.], {1: 1.}), ([-1., 0.], {2: 1.})])
    with pytest.raises(InferenceSchemaError, match='zero_merged_dense'):
        views.encode_query_views(encoder, 'ab')


@pytest.mark.parametrize('failure', [InferenceRateLimited(retry_after=93),
    InferenceDeadlineExceeded(), InferenceUnavailable('http_503', status_code=503),
    InferenceInputRejected(status_code=400)])
def test_later_failure_keeps_original_signal_no_partial_pair_or_retry(monkeypatch, failure):
    controlled_plan(monkeypatch)
    encoder = Encoder([([1., 0.], {1: 1.}), failure])
    with pytest.raises(type(failure)) as raised:
        views.encode_query_views(encoder, 'ab')
    assert raised.value is failure and encoder.calls == ['a', 'b']


@pytest.mark.parametrize('pair', [([float('nan'), 0.], {1: 1.}),
    ([True, 0.], {1: 1.}), ([1.], {1: 1.}), ([1., 0.], {}),
    ([1., 0.], {1: -1.}), ([1., 0.], {1: float('inf')}),
    ([1., 0.], {True: 1.}), ([1., 0.], {2**32: 1.}), None])
def test_bad_later_pair_never_becomes_success(monkeypatch, pair):
    controlled_plan(monkeypatch)
    encoder = Encoder([([1., 0.], {1: 1.}), pair])
    with pytest.raises(InferenceSchemaError):
        views.encode_query_views(encoder, 'ab')


def test_whitespace_retained_in_plan_without_blank_dispatch(monkeypatch):
    monkeypatch.setattr(views, 'MAX_VIEW_TOKENS', 8)
    query = 'important question\n' + ' ' * 3000
    plan = views.plan_query(query)
    encoder = Encoder([([1., 0.], {1: 1.})])
    assert ''.join(plan.parts) == query
    assert views.encode_query_views(encoder, query) == ([1., 0.], {1: 1.})
    assert len(encoder.calls) == 1 and encoder.calls[0].strip() == 'important question'


def test_whole_input_capacity_checked_before_tokenizing_or_dispatch(monkeypatch):
    monkeypatch.setattr(views, 'encode_text', lambda _: pytest.fail('Over-capacity tokenized'))
    encoder = Encoder([])
    with pytest.raises(InferenceInputRejected) as raised:
        views.encode_query_views(encoder, 'x' * (views.MAX_QUERY_UTF8_BYTES + 1))
    assert raised.value.status_code == 413 and not encoder.calls


def test_view_capacity_checked_before_any_dispatch(monkeypatch):
    monkeypatch.setattr(views, 'MAX_VIEW_TOKENS', 4)
    monkeypatch.setattr(views, 'MAX_QUERY_VIEWS', 2)
    encoder = Encoder([])
    with pytest.raises(InferenceInputRejected) as raised:
        views.encode_query_views(encoder, '你' * 12)
    assert raised.value.status_code == 413 and not encoder.calls


def test_expired_deadline_prevents_tokenization_and_dispatch(monkeypatch):
    monkeypatch.setattr(views, 'encode_text', lambda _: pytest.fail('Expired request tokenized'))
    encoder = Encoder([])
    with deadline_budget(0), pytest.raises(InferenceDeadlineExceeded):
        views.encode_query_views(encoder, 'question')
    assert not encoder.calls


@pytest.mark.parametrize('slow_at', [1, 2])
def test_expiry_between_views_or_after_last_cannot_return_success(monkeypatch, slow_at):
    controlled_plan(monkeypatch)
    class Slow(Encoder):
        def encode_query(self, query):
            result = super().encode_query(query)
            if len(self.calls) == slow_at:
                time.sleep(.025)
            return result
    encoder = Slow([([1., 0.], {1: 1.}), ([0., 1.], {2: 1.})])
    with deadline_budget(.015), pytest.raises(InferenceDeadlineExceeded):
        views.encode_query_views(encoder, 'ab')
    assert len(encoder.calls) == slow_at


def test_disclosure_keeps_provider_revision_unattested_and_no_storage_change():
    value = views.descriptor()
    assert value['max_view_tokens'] == 8000 and value['remaining_headroom_tokens'] == 191
    assert value['tokenizer']['provider_v4_revision_attested'] is False
    assert not value['tokenizer']['runtime_network']
    assert value['query_only'] and 'approximate-token-weighted' in value['long_query']
    assert value['storage_or_document_encoding_changed'] is False
    assert value['lexical_and_reranker_query'] == 'original-full-query'


def test_invalid_unicode_rejected_without_dispatch():
    encoder = Encoder([])
    with pytest.raises(InferenceInputRejected) as raised:
        views.encode_query_views(encoder, 'invalid \ud800')
    assert raised.value.status_code == 400 and not encoder.calls
