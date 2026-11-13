"""Artificial task families and exhaustive-review invariants; no real models."""
from types import SimpleNamespace
import pytest

from mandol_competition.inference.evidence_selector import Selection
from mandol_competition.inference.errors import InferenceRateLimited, InferenceSchemaError
from mandol_competition.multimodal import evidence_review as review
from mandol_competition.multimodal.protocol import SearchRequest
from .test_grounded_evidence import memory

NO_MEDIA = SimpleNamespace(get=lambda _: None)


class Guard:
    def __init__(self, predicate=lambda text: False, *, only_verify=False, error=None):
        self.predicate, self.only_verify, self.error = predicate, only_verify, error
        self.calls = []

    def call(self, phase, inputs):
        self.calls.append((phase, inputs))
        if self.error:
            raise self.error
        indices = tuple(i for i, c in enumerate(inputs['candidates']) if self.predicate(c['text']))
        if self.only_verify and phase == 'primary':
            indices = ()
        return Selection(indices[:inputs['limit']], 'selected', 0.)

    def select(self, **inputs):
        return self.call('primary', inputs)

    def confirm_absence(self, **inputs):
        return self.call('verification', inputs)


def query(text='What is the printed folder label?', **kwargs):
    return SearchRequest(user_id='alice', query=text, top_k=100, **kwargs)


def test_long_100_source_absence_is_reviewed_fully_instead_of_655_byte_heads():
    pool = [(memory(str(i), 'A description of an unrelated shore. ' * 34), .9-i/200) for i in range(100)]
    guard = Guard()
    result = review.review_sources(guard, query(), pool, NO_MEDIA)
    assert result.hits == [] and result.complete_absence_review
    assert 4 <= result.model_calls <= review.MAX_MODEL_CALLS
    assert {phase for phase, _ in guard.calls} == {'primary', 'verification'}
    assert all(len(c['text']) <= 2400 for _, inputs in guard.calls for c in inputs['candidates'])


@pytest.mark.parametrize('count', [17, 32, 100])
def test_every_original_image_has_to_be_seen_in_both_passes(count):
    pool = [(memory(str(i), 'Artificial photo.', images=[f'owned-{i}']), .9-i/200) for i in range(count)]
    media = SimpleNamespace(get=lambda key: (key.encode(), 'image/png'))
    guard = Guard()
    result = review.review_sources(guard, query(), pool, media)
    assert result.hits == [] and result.complete_absence_review
    assert result.model_calls <= review.MAX_MODEL_CALLS
    for phase in ('primary', 'verification'):
        images = [uri for p, inputs in guard.calls if p == phase for c in inputs['candidates'] for uri in c['images']]
        assert len(images) == count


@pytest.mark.parametrize('fact', [
    'The folder is teal.', 'The label reads ORBIT-47.', 'There are three folders.',
    'I did not buy this folder.', 'Correction: the blue folder replaced the green folder.',
    'The person from the earlier session gave me this folder.', '这只文件夹上写着轨道四十七号。'])
@pytest.mark.parametrize('verify', [False, True])
def test_tail_atomic_negative_corrective_and_bridge_evidence_preserve_the_complete_floor(fact, verify):
    body = 'Unrelated scene. ' * 700 + fact
    pool = [(memory('noise', 'Trees beside a lake.'), .9), (memory('fact', body, session='later'), .01)]
    guard = Guard(lambda text: fact in text, only_verify=verify)
    result = review.review_sources(guard, query(), pool, NO_MEDIA)
    assert result.hits == pool and result.preferred_sources == (pool[1][0],)
    assert all(a is b for a,b in zip(pool,result.hits))
    assert result.model_calls <= review.MAX_MODEL_CALLS


def test_shared_original_is_repeated_when_a_new_batch_needs_it():
    pool = [(memory(str(i), 'Artificial source. ' * 100, images=['shared']), .8) for i in range(30)]
    media = SimpleNamespace(get=lambda _: (b'artificial original', 'image/png'))
    guard = Guard()
    result = review.review_sources(guard, query(), pool, media)
    assert result.complete_absence_review
    for _, inputs in guard.calls:
        assert sum(len(c['images']) for c in inputs['candidates']) == 1
        assert any('Shared original image in this batch' in c['text'] for c in inputs['candidates'])


@pytest.mark.parametrize('incomplete', ['missing_media','query_tail','options_tail','fragment_limit','batch_limit','deadline'])
def test_uncertainty_never_becomes_an_empty_success(incomplete, monkeypatch):
    pool = [(memory('fact', '原始说明。' * 500, images=['missing'] if incomplete=='missing_media' else []), .8)]
    request = query('q' * (review.MAX_QUERY_BYTES+1) if incomplete=='query_tail' else 'Printed label?',
                    options=['x' * (review.MAX_OPTIONS_BYTES+1)] if incomplete=='options_tail' else [])
    if incomplete=='fragment_limit':monkeypatch.setattr(review,'MAX_FRAGMENTS',1)
    if incomplete=='batch_limit':monkeypatch.setattr(review,'MAX_BATCHES',0)
    if incomplete=='deadline':monkeypatch.setattr(review,'remaining_budget',lambda _:0.)
    result = review.review_sources(Guard(), request, pool, NO_MEDIA)
    assert result.hits == pool and not result.complete_absence_review


def test_unseen_query_images_prevent_absence():
    request = query([{'type':'text','text':'A visual question'}, *[
        {'type':'image_url','image_url':{'url':'data:image/png;base64,QUJD'}} for _ in range(17)]])
    pool = [(memory('note','An unrelated note.'), .8)]
    result = review.review_sources(Guard(), request, pool, NO_MEDIA)
    assert result.hits == pool and not result.complete_absence_review


@pytest.mark.parametrize('values', [(True,), (-1,), (100,), ({},), (0,0)])
def test_malformed_selection_preserves_original_evidence_without_a_false_absence(values):
    guard = Guard()
    guard.select = lambda **_: Selection(values,'selected',0.)
    pool = [(memory('a','Existing fact.'),.7)]
    result = review.review_sources(guard,query(),pool,NO_MEDIA)
    assert result.hits == pool and not result.complete_absence_review
    assert result.review_incomplete_reason == 'invalid_selector_response'


def test_provider_429_is_not_empty_or_implicitly_retried():
    guard=Guard(error=InferenceRateLimited(retry_after=83))
    with pytest.raises(InferenceRateLimited) as caught:
        review.review_sources(guard,query(),[(memory('a','Existing fact.'),.7)],NO_MEDIA)
    assert caught.value.retry_after==83 and len(guard.calls)==1


def test_original_dialogue_is_losslessly_deduplicated_for_review():
    unit=memory('a','[Dialogue] Exact original.\nOCR: ORBIT-47',images=['picture'])
    unit.raw_data['original_content']='Exact original.'
    unit.raw_data['content_parts'].insert(0,{'type':'text','text':'Exact original.'})
    assert review.evidence_text(unit)=='[Dialogue] Exact original.\nOCR: ORBIT-47'


def test_empty_corpus_needs_no_models():
    assert review.review_sources(None,query(),[],None).model_calls==0


def test_selected_batch_local_index_maps_back_to_the_rank_100_source():
    pool=[(memory(str(i),'An unrelated note. '*65+(' UNIQUE-TAIL-FACT' if i==99 else '')),1-i/200)
          for i in range(100)]
    guard=Guard(lambda text:'UNIQUE-TAIL-FACT' in text)
    result=review.review_sources(guard,query(),pool,NO_MEDIA)
    assert result.hits==pool and result.preferred_sources==(pool[99][0],)
    assert len(guard.calls)>1
    assert all('"source_index"' not in c['text'] for _,inputs in guard.calls for c in inputs['candidates'])
