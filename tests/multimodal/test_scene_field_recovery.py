"""Scene availability, preservation and budgets with artificial responses only."""
from copy import deepcopy
import json

import pytest

from mandol_competition.multimodal.vlm import staged_ocr as staged, scene_recovery as fields
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError, validation_receipt_valid
from mandol_competition.multimodal.vlm.scene_guard import (
    SceneOutputNeedsReview, scene_stream_observer, validate_scene_strings)
from mandol_competition.multimodal.vlm.stage_state import StageState, RecoveryBudgetExhausted, RecoveryStateError
from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable
from .test_staged_ocr_bounded_scene import (
    FixtureClient, inputs, scene, localization, crop, response, field_responses, names)


def old_failure(client, data):
    ledger=StageState(staged.binding_for(client,data['prompt'],data['image_uri'],data['context'],0,1,None))
    for name in ('scene',staged.bounded_scene_stage()):
        ledger.reserve(name,1024)
        ledger.accept(dict(finish_reason='length',completion_tokens=1024,output_token_budget=1024),
                      dict(prompt_tokens=10,completion_tokens=1024,total_tokens=1034))
    old=ledger.state
    old.pop('scene_recovery')  # An actual old checkpoint has no new field.
    return old


def run(client, data=None, **kwargs):
    snapshots=[]
    result=staged.complete_staged(client,**(data or inputs()),checkpoint=snapshots.append,**kwargs)
    return result,snapshots[-1]


def test_old_two_length_failures_recover_every_original_scene_field_without_reset():
    client=FixtureClient([*field_responses(),response(localization()),response(crop())])
    data=inputs();old=old_failure(client,data)
    (payload,usage,receipt,_),state=run(client,data,resume=old,preserve_exhausted=True)
    assert {key:payload[key] for key in scene()}==scene()
    assert payload['ocr_text']=='Literal visible label' and validation_receipt_valid(receipt)
    assert state['calls'][:2]==old['calls'] and state['binding']==old['binding']
    assert state['paid_calls']==9 and state['spent']==2048+7*40
    assert state['scene']['metadata']['scene_recovery_pages']==5
    assert state['scene']['metadata']['scene_recovery_policy']==fields.policy()
    assert usage['completion_tokens']==7*40
    assert names(client)[:5]==['scene_fields_v1_'+part for part in fields.GROUPS]
    assert all(c['options']['max_completion_tokens']<=1024 for c in client.calls)


def test_fresh_two_incomplete_scene_calls_complete_via_field_recovery():
    bad=response('{"description":"unfinished',tokens=1024,finish='length')
    client=FixtureClient([bad,bad,*field_responses(),response(localization()),response(crop())])
    (payload,_,receipt,_),state=run(client,preserve_exhausted=True)
    assert payload['description']==scene()['description'] and validation_receipt_valid(receipt)
    assert len(client.calls)==state['paid_calls']==9
    assert state['spent']==2048+7*40 and state['content_failures']==0


def test_scene_repetition_is_reread_while_ocr_and_genuine_object_duplicates_are_retained():
    repeated=dict(scene(),description='Artificial red cup. '*80,ocr_lines=[])
    client=FixtureClient([response(repeated,tokens=500),response(scene()),response(localization()),response(crop())])
    (payload,_,receipt,_),state=run(client)
    assert payload['description']==scene()['description'] and validation_receipt_valid(receipt)
    assert state['calls'][0]['metadata']['error_class']=='content_output_rejected'
    assert state['calls'][0]['charged']==1024
    observer=scene_stream_observer(primary=True)
    observer(json.dumps(dict(scene(),ocr_lines=['Same printed price']*80)))
    duplicate=dict(scene(),objects=[dict(name='card',count=1,attributes=['red'])]*2)
    client=FixtureClient([*field_responses(duplicate),response(localization()),response(crop())])
    data=inputs()
    (payload,_,_,_),_=run(client,data,resume=old_failure(client,data))
    assert payload['objects']==duplicate['objects']


@pytest.mark.parametrize('text',['Repeated scene words. '*80,'文字重复。'*100,'\n'*64,'7'*400])
def test_long_scene_string_cycles_signal_review_without_editing_values(text):
    with pytest.raises(SceneOutputNeedsReview):validate_scene_strings({'description':text})


@pytest.mark.parametrize('prefix',['', '{"description":"\\', '{"description":"\\u12',
                                  '{"description":"A normal brief scene'])
def test_partial_escapes_and_normal_prefixes_are_not_misclassified(prefix):
    scene_stream_observer()(prefix)


def test_partial_field_checkpoint_resumes_after_unknown_transport_without_repeating_completed_overview():
    client=FixtureClient([field_responses()[0],InferenceUnavailable('ReadError',unknown_outcome=True)])
    data=inputs();old=old_failure(client,data);snapshots=[]
    with pytest.raises(InferenceUnavailable):
        staged.complete_staged(client,**data,resume=old,checkpoint=snapshots.append)
    saved=snapshots[-1]
    assert len(saved['scene_recovery']['pages'])==1 and saved['scene'] is None
    assert saved['spent']==2048+40+1024
    later=FixtureClient([*field_responses()[1:],response(localization()),response(crop())])
    (payload,_,receipt,_),state=run(later,data,resume=saved)
    assert payload['description']==scene()['description'] and validation_receipt_valid(receipt)
    assert 'scene_fields_v1_overview' not in names(later)
    assert state['calls'][:len(saved['calls'])]==saved['calls']


def test_known_429_keeps_the_same_field_available_without_spending_its_paid_attempt():
    client=FixtureClient([InferenceRateLimited(retry_after=17)])
    data=inputs();old=old_failure(client,data);snapshots=[]
    with pytest.raises(InferenceRateLimited):
        staged.complete_staged(client,**data,resume=old,checkpoint=snapshots.append)
    saved=snapshots[-1]
    assert saved['spent']==2048 and saved['paid_calls']==2 and saved['known_rejections']==1
    later=FixtureClient([*field_responses(),response(localization()),response(crop())])
    (payload,_,_,_),state=run(later,data,resume=saved)
    assert payload['objects']==scene()['objects'] and state['known_rejections']==1


def test_truncated_object_page_retries_smaller_pages_and_preserves_all_items():
    expected=dict(scene(),objects=[dict(name='object-'+str(i),count=i+1,attributes=['attribute-'+str(i)]) for i in range(3)])
    replies=field_responses(expected)
    client=FixtureClient([*replies[:2],response('{"items":[',tokens=1024,finish='length'),
        *[response(dict(items=[item],has_more=i<2)) for i,item in enumerate(expected['objects'])],
        *replies[3:],response(localization()),response(crop())])
    data=inputs()
    (payload,_,receipt,_),state=run(client,data,resume=old_failure(client,data))
    assert payload['objects']==expected['objects'] and validation_receipt_valid(receipt)
    calls=[c for c in client.calls if c['options']['response_format']['json_schema']['name']=='scene_fields_v1_objects']
    assert [c['options']['response_format']['json_schema']['schema']['properties']['items']['maxItems'] for c in calls]==[fields.PAGE_ITEMS,1,1,1]
    assert state['spent']<=16384 and state['paid_calls']<=20


def test_identical_unfinished_page_is_reread_not_appended_or_silently_deduplicated():
    objects=[dict(name='object-'+str(i),count=1,attributes=[]) for i in range(5)]
    base=field_responses()
    first=response(dict(items=objects[:4],has_more=True))
    client=FixtureClient([*base[:2],first,first,response(dict(items=objects[4:],has_more=False)),
                         *base[3:],response(localization()),response(crop())])
    data=inputs()
    (payload,_,_,_),state=run(client,data,resume=old_failure(client,data))
    assert payload['objects']==objects
    assert len([p for p in state['scene_recovery']['pages'] if p['group']=='objects'])==2


def test_explicit_more_pages_preserve_every_item_and_supply_original_scene_labels():
    objects=[dict(name='unique-object-'+str(i),count=1,attributes=[]) for i in range(fields.PAGE_ITEMS+1)]
    replies=field_responses()
    client=FixtureClient([*replies[:2],response(dict(items=objects[:fields.PAGE_ITEMS],has_more=True)),
        response(dict(items=objects[fields.PAGE_ITEMS:],has_more=False)),*replies[3:],response(localization()),response(crop())])
    data=inputs()
    (payload,_,_,_),state=run(client,data,resume=old_failure(client,data))
    assert payload['objects']==objects
    object_calls=[c for c in client.calls if c['options']['response_format']['json_schema']['name']=='scene_fields_v1_objects']
    assert len(object_calls)==2
    body=json.loads(object_calls[1]['messages'][1]['content'][0]['text'].removeprefix(fields.PROMPT))
    assert body['already_collected']==objects[:fields.PAGE_ITEMS]
    assert body['collected_scene_fields']['overview']=={k:scene()[k] for k in ('description','tags','location_hint')}
    assert 'continuation page 1' in object_calls[1]['messages'][1]['content'][-1]['text']


def test_full_completed_field_does_not_invent_more_work_or_discard_any_item():
    expected=dict(scene(),objects=[dict(name='object-'+str(i),count=1,attributes=[]) for i in range(fields.PAGE_ITEMS)])
    client=FixtureClient([*field_responses(expected),response(localization()),response(crop())]);data=inputs()
    (payload,_,_,_),_=run(client,data,resume=old_failure(client,data))
    assert payload['objects']==expected['objects']
    assert names(client).count('scene_fields_v1_objects')==1


@pytest.mark.parametrize('finish,tokens',[('length',1024),('stop',1024)])
def test_complete_typed_bounded_scene_at_limit_is_preserved_before_independent_ocr(finish,tokens):
    client=FixtureClient([response('{"description":"unfinished',tokens=1024,finish='length'),
        response(json.dumps(scene())+' '*4000,tokens=tokens,finish=finish),
        response(localization()),response(crop())])
    (payload,_,receipt,_),state=run(client)
    assert {k:payload[k] for k in scene()}==scene() and validation_receipt_valid(receipt)
    assert state['scene']['metadata']['finish_reason']==finish and state['scene_recovery'] is None


def test_two_failed_attempts_per_field_remain_bounded_across_retries():
    bad=response('{"description":"unfinished',tokens=1024,finish='length')
    client=FixtureClient([bad,bad]);data=inputs();old=old_failure(client,data);snapshots=[]
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client,**data,resume=old,checkpoint=snapshots.append,preserve_exhausted=True)
    saved=snapshots[-1]
    for _ in range(5):
        with pytest.raises(RecoveryBudgetExhausted):
            staged.complete_staged(client,**data,resume=saved,preserve_exhausted=True)
    assert len(client.calls)==2 and saved['spent']==4096 and saved['scene'] is None


def test_page_checkpoint_persistence_failure_blocks_next_provider_dispatch():
    client=FixtureClient(field_responses());data=inputs();old=old_failure(client,data)
    def checkpoint(value):
        if value['scene_recovery'] and value['scene_recovery']['pages']:
            raise OSError('Artificial private checkpoint failure')
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client,**data,resume=old,checkpoint=checkpoint)
    assert len(client.calls)==1


def test_total_budget_and_binding_are_not_renewed_for_the_new_recipe():
    client=FixtureClient([]);data=inputs();old=old_failure(client,data)
    ledger=StageState(old['binding'],resume=old)
    for _ in range(14):
        ledger.reserve('historical-paid',1024)
        ledger.accept(dict(finish_reason='length',completion_tokens=1024,output_token_budget=1024))
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client,**data,resume=ledger.state)
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client,**{**data,'context':'Different source'},resume=old)
    assert not client.calls


def test_incomplete_page_cannot_be_forged_into_a_scene_after_resume():
    client=FixtureClient([field_responses()[0],InferenceUnavailable('ReadError',unknown_outcome=True)])
    data=inputs();old=old_failure(client,data);snapshots=[]
    with pytest.raises(InferenceUnavailable):
        staged.complete_staged(client,**data,resume=old,checkpoint=snapshots.append)
    corrupted=deepcopy(snapshots[-1]);corrupted['scene_recovery']['pages'][0]['wire']['description']='Changed value'
    # The immutable paid receipt alone cannot attest semantic text; persisted
    # page fingerprints are required in addition to that receipt.
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client,**data,resume=corrupted)


def test_completed_pages_cannot_be_removed_or_replaced_by_state_update():
    client=FixtureClient([field_responses()[0],InferenceUnavailable('ReadError',unknown_outcome=True)])
    data=inputs();snapshots=[]
    with pytest.raises(InferenceUnavailable):
        staged.complete_staged(client,**data,resume=old_failure(client,data),checkpoint=snapshots.append)
    saved=snapshots[-1]
    ledger=StageState(saved['binding'],resume=saved)
    with pytest.raises(RecoveryStateError):
        ledger.update(scene_recovery=None)
    with pytest.raises(RecoveryStateError):
        ledger.update(scene_recovery=dict(policy=fields.policy(),pages=[]))


def test_many_distinct_pages_keep_all_verified_items_and_stop_at_shared_recipe_limit():
    pages=[*field_responses()[:2]]
    for index in range(fields.MAX_CALLS-2):
        pages.append(response(dict(items=[dict(name='unique-object-'+str(index),count=1,attributes=[])],has_more=True)))
    client=FixtureClient(pages);data=inputs();snapshots=[]
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client,**data,resume=old_failure(client,data),checkpoint=snapshots.append)
    saved=snapshots[-1]
    assert len(client.calls)==fields.MAX_CALLS and saved['paid_calls']==fields.MAX_CALLS+2
    assert len(saved['scene_recovery']['pages'])==fields.MAX_CALLS
    assert saved['scene'] is None and saved['final_payload'] is None
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client,**data,resume=saved,preserve_exhausted=True)
    assert len(client.calls)==fields.MAX_CALLS
