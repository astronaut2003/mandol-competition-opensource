from types import SimpleNamespace
import pytest

from mandol_competition.inference.evidence_selector import Selection
from mandol_competition.textual.core.answer_utility import AnswerUtilityPolicy, select_existing, candidate_views


def hits():return [{'id':str(i),'content':'existing raw '+str(i),'score':.9-i/100} for i in range(8)]


def test_fixed_adaptive_depth_respects_actual_request_maximum():
    fixed=AnswerUtilityPolicy(output_k=6)
    assert len(select_existing(fixed,hits(),query='fact',options=None,requested=3)[0])==3
    adaptive=AnswerUtilityPolicy(depth='adaptive-v1')
    assert adaptive.limit('List all the steps',100)==16
    assert adaptive.limit('Where does she live?',100)==6
    assert adaptive.limit('Compare both journeys',1)==1
    for invalid in ({'output_k':True},{'selector':'answer'},{'candidate_k':101},{'depth':'qa-id'}):
        with pytest.raises(ValueError):AnswerUtilityPolicy(**invalid)


def test_selector_returns_existing_evidence_with_bounded_rank_scores():
    calls=[]
    class Selector:
        def select(self,**kwargs):calls.append(kwargs);return Selection((4,1),'selected',.1)
    original=hits()
    selected,fallback=select_existing(AnswerUtilityPolicy(selector='mini'),original,
        query='actual question',options=['option A','option B'],requested=3,selector=Selector())
    assert [h['id'] for h in selected]==['4','1'] and not fallback
    assert [h['content'] for h in selected]==[original[4]['content'],original[1]['content']]
    assert [h['score'] for h in selected]==[1,.5]
    assert calls[0]['options']==['option A','option B']
    assert set(calls[0])=={'query','options','candidates','limit'}


def test_unknown_selector_outcome_returns_baseline_order_without_creating_answer():
    class Selector:
        def select(self,**kwargs):return Selection(tuple(range(8)),'fallback',90)
    original=hits()
    selected,fallback=select_existing(AnswerUtilityPolicy(selector='mini'),original,
        query='q',options=None,requested=4,selector=Selector())
    assert fallback and selected==original[:4]


def test_lifecycle_view_does_not_claim_fact_current_from_canonical_version():
    original=hits();original.append(dict(original[0],id='duplicate'))
    canonical={h['id']:SimpleNamespace(business_timestamp_ms=None) for h in original}
    indices,views=candidate_views(original,canonical=canonical,lifecycle=True)
    assert len(indices)==8
    assert all('fact current status unknown' in v['text'] for v in views)
    assert all('Unix milliseconds' not in v['text'] for v in views)
    # Distinct explicitly dated observations survive duplicate suppression.
    original[1]['content']='[Source mention time: 2020]\nSame preference'
    original[2]['content']='[Source mention time: 2026]\nSame preference'
    assert 1 in candidate_views(original,lifecycle=True)[0] and 2 in candidate_views(original,lifecycle=True)[0]
