from types import SimpleNamespace

import pytest

from mandol_competition.inference.errors import InferenceSchemaError
from mandol_competition.multimodal.reranking import RankBlendedReranker


POLICY = {'version':'mm-post-rerank-rrf-v1','weight':2.,'k':60}


class Model:
    recipe = SimpleNamespace(recipe_id='real-recipe-stays-with-transport')
    def __init__(self, scores):
        self.scores, self.calls = scores, []
    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return self.scores
    def attestation(self):
        return {'level':'fixture'}


def test_blend_calls_model_once_and_preserves_candidate_membership_and_input():
    model = Model([.2, .9, .9])
    policy = RankBlendedReranker(model, POLICY)
    documents = ['first fusion hit', 'second fusion hit', 'third fusion hit']
    values = policy.score('query', documents)
    assert len(values) == len(documents)
    assert values[1] > values[2] > values[0]
    assert model.calls == [('query', documents)]
    assert policy.recipe is model.recipe
    assert policy.attestation()['application_post_rerank_policy'] == POLICY
    assert policy.score('query', []) == []
    assert len(model.calls) == 1


def test_flat_model_scores_keep_the_observed_fusion_order():
    values = RankBlendedReranker(Model([.5]*4), POLICY).score('query', ['a','b','c','d'])
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == 4


@pytest.mark.parametrize('scores', [[float('nan'),.2], [float('inf'),.2], [.1]])
def test_blending_cannot_hide_an_invalid_model_response(scores):
    with pytest.raises(InferenceSchemaError):
        RankBlendedReranker(Model(scores), POLICY).score('query', ['a','b'])


@pytest.mark.parametrize('change', [{'version':'unknown'}, {'k':0}, {'weight':0},
    {'weight':float('nan')}, {'weight':True}, {'unversioned_extra':1}])
def test_unknown_or_unbounded_policy_fails_closed(change):
    with pytest.raises(ValueError):
        RankBlendedReranker(Model([]), POLICY | change)
