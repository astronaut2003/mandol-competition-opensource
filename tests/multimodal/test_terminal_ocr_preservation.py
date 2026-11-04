"""Observed exhausted Add shapes, synthetic sources, no provider or DB access."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from mandol_competition.multimodal.vlm import staged_ocr as staged, ocr_review
from mandol_competition.multimodal.vlm.stage_state import StageState, RecoveryStateError, RecoveryBudgetExhausted
from mandol_competition.multimodal.vlm.output_guard import (
    INCOMPLETE_OCR_POLICY, REGION_REVIEW_POLICY, validation_receipt_valid, verified_ocr_coverage)
from mandol_competition.multimodal.vlm.cache import ExtractionCache, _media_id, context_digest
from mandol_competition.multimodal.vlm.base import parse_extraction, ExtractionError
from mandol_competition.multimodal.ocr_provenance import incomplete_ocr_note
from .test_ocr_independent_review import saved_scene, contradictory_plan, accepted, no_text, review
from .test_staged_ocr_bounded_scene import FixtureClient, inputs, scene, crop, response
from .test_staged_ocr_pipeline import pipeline, extract, recovering_handler, uri


def exhausted(client, data, kind):
    state = saved_scene(client, data)
    if kind == 'content':
        for _ in range(4): state.content_failure()
    elif kind == 'repair':
        state.update(ocr_repair=dict(policy=ocr_review.policy(), reason='blank_conflict', failures=0, review=None))
        state.repair_failure(); state.repair_failure()
    elif kind == 'tokens':
        state.reserve('historical', 15360)
        state.reject(RuntimeError('artificial unknown paid result'))
    else:
        for _ in range(19): accepted(state, 'historical', 1)
    return state.state


@pytest.mark.parametrize('kind', ['content', 'repair', 'tokens', 'calls'])
def test_exhausted_add_preserves_scene_with_explicit_incomplete_ocr_and_no_new_dispatch(kind):
    client = FixtureClient([]); data = inputs(); old = exhausted(client, data, kind); snapshots = []
    payload, usage, receipt, _ = staged.complete_staged(client, **data, resume=old,
        checkpoint=snapshots.append, preserve_exhausted=True)
    assert {k: payload[k] for k in scene()} == scene() and payload['ocr_text'] == ''
    assert usage is None and receipt['policy'] == INCOMPLETE_OCR_POLICY
    assert receipt['finish_reason'] == receipt['ocr_coverage']['status'] == 'incomplete'
    assert validation_receipt_valid(receipt) and not client.calls
    for key in ('spent', 'observed_tokens', 'paid_calls', 'content_failures', 'calls', 'binding', 'ocr_repair'):
        assert snapshots[-1][key] == old[key]
    again = staged.complete_staged(client, **data, resume=snapshots[-1], preserve_exhausted=True)
    assert again[:3] == (payload, usage, receipt) and not client.calls


def test_partial_validated_literal_text_is_not_lost_with_other_unfinished_crops():
    data = inputs(); data['image_uri'] = uri(size=(200, 600)); client = FixtureClient([])
    old = contradictory_plan(client, data)
    state = StageState(old['binding'], resume=old)
    tasks = [staged.CropTask.from_payload(t) for t in state.state['tasks']]
    checked = staged.ValidatedCrop.from_payload(tasks[0], state.state['results'][tasks[0].crop_id]['validated'])
    from mandol_competition.multimodal.vlm.ocr_recovery import validate_crop_result
    checked = validate_crop_result(tasks[0], crop())
    results = state.state['results']; results[tasks[0].crop_id]['validated'] = checked.to_payload()
    state.update(results=results, ocr_repair=dict(policy=ocr_review.policy(), reason='blank_conflict', failures=0, review=None))
    state.repair_failure(); state.repair_failure()
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=state.state, preserve_exhausted=True)
    assert payload['ocr_text'] == 'Literal visible label'
    assert receipt['ocr_coverage']['status'] == 'incomplete' and not client.calls


def test_repeated_whole_image_and_crop_disagreement_ends_in_incomplete_not_verified_no_text():
    data = inputs(); data['image_uri'] = uri(size=(200, 600))
    client = FixtureClient([response(review('text_regions')), response(no_text()), response(no_text())]*2)
    old = contradictory_plan(client, data); snapshots=[]
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old,
        checkpoint=snapshots.append, preserve_exhausted=True)
    assert len(client.calls) == 6 and payload['ocr_text'] == ''
    assert receipt['ocr_coverage']['status'] == 'incomplete' and validation_receipt_valid(receipt)
    before = len(client.calls)
    staged.complete_staged(client, **data, resume=snapshots[-1], preserve_exhausted=True)
    assert len(client.calls) == before


def test_corrupt_binding_and_checkpoint_persistence_errors_never_become_preserved_success():
    data = inputs(); client = FixtureClient([]); old = exhausted(client, data, 'content')
    changed = dict(data, context='different owned source')
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client, **changed, resume=old, preserve_exhausted=True)
    def fail(_): raise OSError('artificial disk failure')
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client, **data, resume=old, checkpoint=fail, preserve_exhausted=True)
    assert old['final_payload'] is None and not client.calls


def test_budget_exhausted_without_valid_scene_cannot_fabricate_a_generic_visual_description():
    data = inputs(); client = FixtureClient([])
    state = StageState(staged.binding_for(client, data['prompt'], data['image_uri'],data['context'],0,1,None))
    for _ in range(4): state.content_failure()
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=state.state, preserve_exhausted=True)
    assert not client.calls


def test_incomplete_receipt_cannot_enter_success_cache_or_claim_completed_ocr(tmp_path):
    data = inputs(); client = FixtureClient([]); old = exhausted(client,data,'repair')
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old, preserve_exhausted=True)
    extraction = parse_extraction(payload,model='gpt-4o-mini',prompt_version='owned',schema_version='1')
    with pytest.raises(ExtractionError):
        ExtractionCache(tmp_path).put(_media_id(data['image_uri']),'owned',extraction,output_validation=receipt)
    for field, value in [('finish_reason','stop'), ('scene_retained',False), ('reason','unverified'),
                         ('completion_tokens',10), ('policy',REGION_REVIEW_POLICY)]:
        bad = deepcopy(receipt); bad[field]=value
        assert not validation_receipt_valid(bad)


def test_request_ledger_retains_incomplete_receipt_on_fresh_instance_without_caching(pipeline,tmp_path):
    from mandol_competition.inference.errors import InferenceUnavailable
    def handler(stage,n):
        if stage == 'crop': raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p = pipeline(handler)
    for _ in range(30):
        try: result=extract(p)
        except Exception as exc:
            from mandol_competition.multimodal.vlm.openai_compat import VisionExtractionUnavailable
            assert isinstance(exc,VisionExtractionUnavailable)
            continue
        break
    else: pytest.fail('exhausted Add did not retain its input-bound scene')
    entry=p.ledger().value['vision']['0:1']; before=len(p.sdk.calls)
    assert entry['ocr_coverage']['status']=='incomplete' and entry['output_validation']['policy']==INCOMPLETE_OCR_POLICY
    assert verified_ocr_coverage(entry['output_validation'],result,media_id=_media_id(p.inputs['image_data_uri']),
        context_sha256=context_digest(p.inputs['context_text']),image_position=0,image_count=1)['status']=='incomplete'
    assert extract(p)==result and len(p.sdk.calls)==before and not p.cache.writes
    note=incomplete_ocr_note(SimpleNamespace(metadata={'visual_ocr_readability':[dict(image_position=0,**entry['ocr_coverage'])]}))
    assert 'incomplete' in note and 'not evidence' in note
    from dataclasses import replace
    from mandol_competition.multimodal.protocol import AddRequest
    from mandol_competition.multimodal.service import AddService, selector_candidate_text
    p.backend.profile=SimpleNamespace(options={})
    config=replace(p.extractor.config,data_dir=tmp_path)
    body=AddRequest.model_validate(dict(user_id=p.request.user_id,request_id=p.request.request_id,
        session_id='owned-session',messages=[dict(role='user',content=[
            dict(type='text',text=p.inputs['context_text']),
            dict(type='image_url',image_url=dict(url=p.inputs['image_data_uri']))])]))
    units=AddService(config,extractor=p.extractor,backend=p.backend).prepare_units(body,preparation=p.ledger())
    assert units[0].metadata['vlm_degraded'] is True
    assert units[0].metadata['visual_ocr_readability'][0]['status']=='incomplete'
    assert units[0].raw_data['content_parts'][0]['text']==p.inputs['context_text']
    assert units[0].raw_data['multimodal_info']['images'][0]['description']==result.description
    assert 'incomplete' in selector_candidate_text(units[0]) and len(p.sdk.calls)==before
    from mandol_competition.multimodal.media_store import UserMediaStore
    import base64
    stored=UserMediaStore(config,p.request.user_id).get(_media_id(p.inputs['image_data_uri']))
    assert stored[0]==base64.b64decode(p.inputs['image_data_uri'].split(',')[1])
