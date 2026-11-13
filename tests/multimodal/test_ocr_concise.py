"""Strict alternate-crop retry semantics, artificial pictures/responses only."""
from copy import deepcopy
import pytest

from mandol_competition.multimodal.vlm import staged_ocr as staged,ocr_concise
from mandol_competition.multimodal.vlm.ocr_recovery import full_image_confirmation_task
from mandol_competition.multimodal.vlm.stage_state import StageState,RecoveryBudgetExhausted
from mandol_competition.multimodal.vlm.output_guard import validation_receipt_valid
from mandol_competition.inference.errors import InferenceRateLimited,InferenceUnavailable
from .test_staged_ocr_bounded_scene import FixtureClient,inputs,scene,localization,crop,response


def contradiction():
    return dict(inspected_entire_crop=True,assessment='no_text',lines=[],
        unreadable_regions=[dict(bbox=[0,0,1000,1000],reason='uncertain')])


def replies(last):
    return [response(dict(scene(),ocr_lines=['printed row']*64)),response(localization()),response(contradiction()),last]


def test_conflicting_complete_crop_gets_one_short_read_without_changing_scene_or_validators():
    client=FixtureClient(replies(response(crop())));saved=[]
    payload,_,receipt,_=staged.complete_staged(client,**inputs(),checkpoint=saved.append)
    assert payload['ocr_text']=='Literal visible label' and {k:payload[k] for k in scene()}==scene()
    assert validation_receipt_valid(receipt) and len(client.calls)==4
    assert client.calls[-1]['messages'][0]['content']==ocr_concise.SYSTEM
    assert client.calls[-1]['messages'][1]['content'][0]['image_url']==client.calls[-2]['messages'][1]['content'][1]['image_url']
    assert saved[-1]['content_failures']==1 and saved[-1]['spent']==160
    assert len(ocr_concise.paid_calls(saved[-1]['calls']))==1


def test_known_429_resumes_pending_short_read_without_repeating_paid_original_crop():
    client=FixtureClient(replies(InferenceRateLimited(retry_after=1)));saved=[];data=inputs()
    with pytest.raises(InferenceRateLimited):staged.complete_staged(client,**data,checkpoint=saved.append)
    old=saved[-1];assert old['spent']==120 and old['paid_calls']==3
    later=FixtureClient([response(crop())]);resumed=[]
    payload,_,_,_=staged.complete_staged(later,**data,resume=old,checkpoint=resumed.append)
    assert payload['ocr_text']=='Literal visible label' and len(later.calls)==1
    assert later.calls[0]['messages'][0]['content']==ocr_concise.SYSTEM
    assert resumed[-1]['content_failures']==old['content_failures']==1
    assert resumed[-1]['calls'][:len(old['calls'])]==old['calls']


def test_unknown_short_read_is_charged_and_does_not_refresh_its_paid_grant():
    client=FixtureClient(replies(InferenceUnavailable('ReadError',unknown_outcome=True)));saved=[];data=inputs()
    with pytest.raises(InferenceUnavailable):staged.complete_staged(client,**data,checkpoint=saved.append)
    old=saved[-1];assert old['spent']==120+1024 and old['paid_calls']==4
    task=staged.CropTask.from_payload(old['tasks'][0])
    assert not ocr_concise.authorized(old,ocr_concise.stage(task))


def test_no_short_read_is_authorized_without_a_validated_owned_crop_plan():
    client=FixtureClient([]);data=inputs()
    state=StageState(staged.binding_for(client,data['prompt'],data['image_uri'],data['context'],0,1,None))
    task=full_image_confirmation_task((120,120))
    with pytest.raises(RecoveryBudgetExhausted):state.reserve(ocr_concise.stage(task),1024)
    assert state.state['paid_calls']==0


def test_paid_short_call_limit_is_shared_across_crop_ids_and_versions():
    task=full_image_confirmation_task((120,120))
    state=dict(calls=[dict(stage=ocr_concise.VERSION+':old-a',known_rejection=False),
                      dict(stage=ocr_concise.VERSION+':old-b',known_rejection=False)],tasks=[task.to_payload()])
    assert not ocr_concise.authorized(state,ocr_concise.stage(task))


@pytest.mark.parametrize('wire',[{},dict(contradiction(),inspected_entire_crop=False),
    dict(contradiction(),lines=[dict(bbox=[0,0,900,100],text='visible')]),
    dict(inspected_entire_crop=True,assessment='no_text',lines=[],unreadable_regions=[])])
def test_short_route_does_not_weaken_other_crop_shapes_or_complete_blank_results(wire):
    assert not ocr_concise.contradictory(wire)


def switched_replies(last):
    second=crop();second['lines'][0]['text']='Second literal row'
    return [response(dict(scene(),ocr_lines=['printed row']*64)),response(localization()),
        response(contradiction()),response(crop()),last if last is not None else response(second)]


def test_verified_short_read_changes_later_crop_recipe_without_an_extra_retry():
    from .test_staged_ocr_pipeline import uri
    from mandol_competition.multimodal.vlm import ocr_strategy
    data=inputs();data['image_uri']=uri(size=(120,600))
    client=FixtureClient(switched_replies(None));saved=[]
    payload,_,receipt,_=staged.complete_staged(client,**data,checkpoint=saved.append)
    assert payload['ocr_text']=='Literal visible label\nSecond literal row'
    assert validation_receipt_valid(receipt) and len(client.calls)==5
    assert all(c['messages'][0]['content']==ocr_concise.SYSTEM for c in client.calls[-2:])
    assert len(ocr_concise.paid_calls(saved[-1]['calls']))==1
    assert ocr_strategy.is_stage(saved[-1]['calls'][-1]['stage'])
    assert saved[-1]['content_failures']==1


def test_strategy_survives_retry_without_paying_again_for_successful_crops():
    from .test_staged_ocr_pipeline import uri
    data=inputs();data['image_uri']=uri(size=(120,600));saved=[]
    client=FixtureClient(switched_replies(InferenceRateLimited(retry_after=1)))
    with pytest.raises(InferenceRateLimited):staged.complete_staged(client,**data,checkpoint=saved.append)
    before=saved[-1];assert before['ocr_strategy'] is not None and before['paid_calls']==4
    later=FixtureClient([response(crop())]);after=[]
    staged.complete_staged(later,**data,resume=before,checkpoint=after.append)
    assert len(later.calls)==1 and later.calls[0]['messages'][0]['content']==ocr_concise.SYSTEM
    assert after[-1]['paid_calls']==5 and after[-1]['ocr_strategy']==before['ocr_strategy']


def test_unverified_strategy_cannot_authorize_a_short_primary_crop():
    from mandol_competition.multimodal.vlm import ocr_strategy
    client=FixtureClient([]);data=inputs()
    state=StageState(staged.binding_for(client,data['prompt'],data['image_uri'],data['context'],0,1,None))
    with pytest.raises(staged.RecoveryStateError):
        state.reserve(ocr_strategy.stage(full_image_confirmation_task((120,120))),1024)
    assert state.state['paid_calls']==0
