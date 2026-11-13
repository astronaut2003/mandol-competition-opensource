"""Artificial paid-call fixtures only; no backend, database, model or network."""
import base64
from collections import deque
from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from mandol_competition.multimodal.vlm import staged_ocr as staged
from mandol_competition.multimodal.vlm.ocr_lines import FORMAT_POLICY
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError, validation_receipt_valid
from mandol_competition.multimodal.vlm.stage_state import (
    StageState, RecoveryBudgetExhausted, RecoveryStateError,
)


def image_uri():
    output = BytesIO()
    with Image.new('RGB', (120, 120), 'white') as image:
        image.save(output, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def scene():
    return dict(description='An indoor desk with two red cups and a green book.',
        entities=[dict(name='Cup label', type='product', attributes=['red label'])],
        objects=[dict(name='cup', count=2, attributes=['red']),
                 dict(name='book', count=1, attributes=['green'])],
        events=[dict(subject='person', action='holds', object='cup')],
        relations=[dict(source='cup', predicate='on', target='desk')],
        tags=['desk', 'cup', 'book'], location_hint='indoors')


def localization():
    return dict(inspected_entire_image=True, assessment='text_regions', regions=[dict(
        region_id='label', bbox=[100,100,900,900], readability='readable', reason='Artificial label')])


def crop():
    return dict(inspected_entire_crop=True, assessment='readable_text',
        lines=[dict(bbox=[100,100,900,200], text='Literal visible label')], unreadable_regions=[])


def field_responses(value=None):
    value = value or scene()
    return [response({key: value[key] for key in ('description', 'tags', 'location_hint')})] + [
        response(dict(items=value[key], has_more=False)) for key in ('entities', 'objects', 'events', 'relations')]


def response(body, *, tokens=40, finish='stop'):
    return SimpleNamespace(content=body if isinstance(body, str) else json.dumps(body),
        tokens=tokens, finish=finish)


class FixtureClient:
    model = request_model = 'gpt-4o-mini'
    wire_format = FORMAT_POLICY
    output_tokens = max_output_tokens = 1024

    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def _dispatch(self, messages, options):
        self.calls.append(dict(messages=deepcopy(messages), options=deepcopy(options)))
        assert self.responses, 'Unexpected extra paid dispatch'
        value = self.responses.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    def _dispatch_stream(self, messages, options, observer):
        value = self._dispatch(messages, options)
        observer(value.content)
        return value

    def _response_metadata(self, value, limit):
        usage = dict(prompt_tokens=10, completion_tokens=value.tokens, total_tokens=10+value.tokens)
        return usage, SimpleNamespace(content=value.content, refusal=None), dict(
            finish_reason=value.finish, completion_tokens=value.tokens, output_token_budget=limit)


def inputs():
    return dict(prompt='Artificial primary instructions', image_uri=image_uri(),
        context='Artificial original source context', position=0, count=1)


def history(client, data, *, scenes=2, other_calls=0, other_tokens=1):
    ledger = StageState(staged.binding_for(client, data['prompt'], data['image_uri'],
        data['context'], 0, 1, None))
    for _ in range(scenes):
        ledger.reserve('scene', 1024)
        ledger.accept(dict(finish_reason='length', completion_tokens=1024, output_token_budget=1024),
            dict(prompt_tokens=10, completion_tokens=1024, total_tokens=1034))
    for _ in range(other_calls):
        ledger.reserve('historical-paid-fixture', other_tokens)
        ledger.accept(dict(finish_reason='stop', completion_tokens=other_tokens, output_token_budget=other_tokens),
            dict(prompt_tokens=10, completion_tokens=other_tokens, total_tokens=10+other_tokens))
    return ledger.state


def names(client):
    return [v['options']['response_format']['json_schema']['name'] for v in client.calls]


def test_historical_two_length_scenes_use_one_versioned_scene_only_fallback_without_reset():
    client = FixtureClient([response(scene(), tokens=100), response(localization(), tokens=25), response(crop(), tokens=40)])
    data = inputs(); old = history(client, data); snapshots = []
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old, checkpoint=snapshots.append)
    assert names(client) == ['scene_bounded_v1','ocr_physical_text_localization_v1','ocr_physical_crop_lines_v1']
    assert all(v['options']['max_completion_tokens'] <= 1024 for v in client.calls)
    assert snapshots[-1]['calls'][:2] == old['calls']
    assert snapshots[-1]['spent'] == 2048+100+25+40 and snapshots[-1]['paid_calls'] == 5
    assert snapshots[-1]['binding'] == old['binding']
    assert snapshots[-1]['calls'][2]['stage'].startswith('scene_bounded_v1:')
    assert {k:payload[k] for k in scene()} == scene()
    assert payload['ocr_text'] == 'Literal visible label' and validation_receipt_valid(receipt)
    assert 'ocr_lines' not in client.calls[0]['options']['response_format']['json_schema']['schema']['properties']


def test_first_incomplete_scene_uses_different_schema_immediately_not_second_full_retry():
    client = FixtureClient([response('{"description":"unfinished', tokens=1024, finish='length'),
        response(scene()), response(localization()), response(crop())])
    snapshots=[]
    payload, _, _, _ = staged.complete_staged(client, **inputs(), checkpoint=snapshots.append)
    assert names(client) == ['visual_memory_ocr_lines_v2','scene_bounded_v1',
        'ocr_physical_text_localization_v1','ocr_physical_crop_lines_v1']
    assert snapshots[-1]['calls'][0]['metadata']['finish_reason'] == 'length'
    assert snapshots[-1]['spent'] == 1024+3*40 and snapshots[-1]['paid_calls'] == 4
    assert payload['description'] == scene()['description']


@pytest.mark.parametrize('bad', [
    response('{"description":"unfinished', tokens=1024, finish='length'),
    response('{"description":"still unfinished', tokens=300),
    response(dict(description='Only one field; never fill the others.')),
    response(dict(scene(), ocr_lines=[])),
])
def test_incomplete_fallback_and_failed_field_recovery_cannot_publish_or_dispatch_forever(bad):
    client=FixtureClient([bad, response({'description':'Missing typed fields'}),
                          response({'description':'Missing typed fields'})]); data=inputs(); snapshots=[]
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=history(client,data), checkpoint=snapshots.append)
    assert snapshots[-1]['scene'] is None and snapshots[-1]['final_payload'] is None
    before=len(client.calls); used=snapshots[-1]['spent']
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=snapshots[-1], checkpoint=snapshots.append)
    assert len(client.calls)==before==3 and snapshots[-1]['spent']==used
    assert snapshots[-1]['scene_recovery']['pages']==[]


def test_unknown_paid_bounded_dispatch_is_not_repeated_and_distinct_field_recipe_can_resume():
    class UnknownPaid(RuntimeError): unknown_outcome=True
    client=FixtureClient([UnknownPaid('artificial outcome unknown'),*field_responses(),
                         response(localization()),response(crop())]); data=inputs(); snapshots=[]
    with pytest.raises(UnknownPaid):
        staged.complete_staged(client, **data, resume=history(client,data), checkpoint=snapshots.append)
    assert snapshots[-1]['spent']==3072 and snapshots[-1]['paid_calls']==3
    payload,_,_,_=staged.complete_staged(client, **data, resume=snapshots[-1],checkpoint=snapshots.append)
    assert payload['description']==scene()['description']
    assert names(client).count('scene_bounded_v1')==1
    assert snapshots[-1]['spent']==3072+7*40 and snapshots[-1]['paid_calls']==10


@pytest.mark.parametrize('other_calls,other_tokens', [(14,1024),(18,1)])
def test_original_total_output_and_call_budgets_cannot_be_refreshed(other_calls,other_tokens):
    client=FixtureClient([]); data=inputs(); old=history(client,data,other_calls=other_calls,other_tokens=other_tokens)
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=old)
    assert client.calls==[]


def test_saved_complete_scene_is_reused_and_never_replaced_by_compact_scene():
    client=FixtureClient([response(localization()),response(crop())]);data=inputs()
    ledger=StageState(staged.binding_for(client,data['prompt'],data['image_uri'],data['context'],0,1,None))
    ledger.reserve('scene',1024)
    meta=dict(finish_reason='length',completion_tokens=1024,output_token_budget=1024)
    usage=dict(prompt_tokens=10,completion_tokens=1024,total_tokens=1034)
    ledger.accept(meta,usage)
    saved=dict(payload=scene(),payload_sha256=staged.digest(scene()),metadata=meta,usage=usage,review_reason='truncated_ocr')
    ledger.update(scene=saved);snapshots=[]
    payload,_,_,_=staged.complete_staged(client,**data,resume=ledger.state,checkpoint=snapshots.append)
    assert names(client)==['ocr_physical_text_localization_v1','ocr_physical_crop_lines_v1']
    assert snapshots[-1]['scene']==saved and {k:payload[k] for k in scene()}==scene()


def test_normal_primary_still_returns_original_full_fields_without_recovery():
    client=FixtureClient([response(dict(scene(),ocr_lines=['Ordinary visible label']))])
    payload,_,receipt,_=staged.complete_staged(client,**inputs())
    assert names(client)==['visual_memory_ocr_lines_v2']
    assert {k:payload[k] for k in scene()}==scene() and payload['ocr_text']=='Ordinary visible label'
    assert validation_receipt_valid(receipt)


def test_bounded_recipe_is_versioned_without_changing_historical_input_binding():
    client=FixtureClient([]);data=inputs()
    original=staged.binding_for(client,data['prompt'],data['image_uri'],data['context'],0,1,None)
    assert original['primary_schema_sha256']==staged.digest(staged.response_format())
    assert not any('bounded_scene' in key for key in original)
    assert staged.bounded_scene_stage().startswith('scene_bounded_v1:')
    assert set(staged.bounded_scene_response_format()['json_schema']['schema']['properties'])==set(scene())
    messages=staged.bounded_scene_messages(data['image_uri'],data['context'],0,1)
    assert messages[-1]['content'][-1]['image_url']['url']==data['image_uri']
    assert data['context'] in messages[-1]['content'][0]['text']


def test_no_bounded_scene_can_dispatch_after_checkpoint_persistence_failure():
    client=FixtureClient([response('{"description":"unfinished',tokens=1024,finish='length')])
    snapshots=[]
    def checkpoint(value):
        snapshots.append(deepcopy(value))
        if value['calls']: raise ValueError('Artificial durable receipt failure')
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client,**inputs(),checkpoint=checkpoint)
    assert names(client)==['visual_memory_ocr_lines_v2']
