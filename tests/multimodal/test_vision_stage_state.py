"""Pure checkpoint state/cost invariants; no database or provider calls."""
from copy import deepcopy

import pytest

from mandol_competition.inference.errors import InferenceRateLimited, InferenceUnavailable, InferenceDeadlineExceeded
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError
from mandol_competition.multimodal.vlm.stage_state import (
    StageState, RecoveryStateError, RecoveryBudgetExhausted, digest,
)


BINDING=dict(scope_sha256='a'*64,request_sha256='b'*64,body_sha256='c'*64,
    media_id='img_'+'d'*64,context_sha256='e'*64,prompt_sha256='f'*64,
    model='artificial-mini',provider_sha256='1'*64,schema='artificial-v1',image_position=0,image_count=1)


def metadata(tokens=3,limit=10,finish='stop'):
    return dict(finish_reason=finish,completion_tokens=tokens,output_token_budget=limit)


def scene():
    payload=dict(description='artificial source scene',entities=[],objects=[],events=[],relations=[],tags=[],location_hint=None)
    return dict(payload=payload,payload_sha256=digest(payload),metadata=metadata(),usage=None,review_reason='truncated_ocr')


def accepted(state,tokens=3,limit=10,stage='primary'):
    state.reserve(stage,limit)
    return state.accept(metadata(tokens,limit),dict(prompt_tokens=1,completion_tokens=tokens,total_tokens=1+tokens))


def test_reserve_persists_before_return_and_callback_and_state_are_deep_copies():
    snapshots=[]
    value=StageState(BINDING,checkpoint=snapshots.append)
    assert len(snapshots)==1 and snapshots[0]['reservation'] is None
    result=value.reserve('primary',10)
    assert snapshots[-1]['reservation']==result and snapshots[-1]['spent']==10
    result['limit']=999;snapshots[-1]['binding']['model']='changed'
    external=value.state;external['reservation']['limit']=999
    assert value.state['binding']==BINDING and value.state['reservation']['limit']==10


def test_accept_settles_only_current_reservation_and_retains_prior_charges():
    value=StageState(BINDING)
    receipt=accepted(value)
    assert receipt==dict(stage='primary',metadata=metadata(),usage=dict(prompt_tokens=1,completion_tokens=3,total_tokens=4))
    value.reserve('crop:1',10);value.accept(metadata(2),None)
    assert value.state['spent']==value.state['observed_tokens']==5 and value.state['paid_calls']==2
    assert [r['charged'] for r in value.state['calls']]==[3,2]


def test_unknown_completion_conservatively_consumes_limit():
    value=StageState(BINDING);value.reserve('primary',10)
    value.accept(metadata(None),None)
    assert value.state['spent']==10 and value.state['observed_tokens']==0
    assert value.state['paid_calls']==1


def test_accept_uses_consistent_usage_when_metadata_completion_missing():
    value=StageState(BINDING);value.reserve('primary',10)
    value.accept(metadata(None),dict(completion_tokens=4))
    assert value.state['spent']==value.state['observed_tokens']==4


@pytest.mark.parametrize('meta,usage',[(metadata(True),None),(metadata(11),None),
    (metadata(3,9),None),(metadata(3),dict(completion_tokens=4)),(metadata(3),dict(total_tokens=float('nan')))])
def test_invalid_settlement_leaves_reserved_charge(meta,usage):
    value=StageState(BINDING);value.reserve('primary',10)
    with pytest.raises(RecoveryStateError):value.accept(meta,usage)
    assert value.state['spent']==10 and value.state['reservation']==dict(stage='primary',limit=10)


def test_known_429_refunds_only_this_reservation_not_paid_primary():
    value=StageState(BINDING);accepted(value)
    value.reserve('localization',10);value.reject(InferenceRateLimited(retry_after=60))
    assert value.state['spent']==3 and value.state['paid_calls']==1 and value.state['observed_tokens']==3
    assert value.state['known_rejections']==1 and value.state['calls'][-1]['known_rejection'] is True


@pytest.mark.parametrize('exc',[InferenceUnavailable('ReadError',unknown_outcome=True),
    VisionOutputError('vision_output_ocr_repetition',retryable=False)])
def test_unknown_and_content_abort_keep_full_current_charge(exc):
    value=StageState(BINDING);accepted(value);value.reserve('crop:1',10);value.reject(exc)
    assert value.state['spent']==13 and value.state['paid_calls']==2
    assert value.state['calls'][-1]['known_rejection'] is False


def test_accepted_stream_429_and_paid_flag_cannot_refund():
    for kind in ('unknown','paid'):
        exc=InferenceRateLimited()
        if kind=='unknown':exc.unknown_outcome=True
        else:exc.vision_paid_output_failure=True
        value=StageState(BINDING);value.reserve('crop:1',10);value.reject(exc)
        assert value.state['spent']==10 and value.state['paid_calls']==1


def test_explicit_connect_before_dispatch_and_deadline_can_refund():
    for exc in (InferenceUnavailable('ConnectError',unknown_outcome=False),InferenceDeadlineExceeded()):
        value=StageState(BINDING);value.reserve('primary',10);value.reject(exc)
        assert value.state['spent']==value.state['paid_calls']==0


def test_crash_reservation_is_charged_once_on_resume_and_closed_before_next_reserve():
    value=StageState(BINDING);accepted(value);value.reserve('crop:1',10)
    original=value.state;snapshots=[]
    resumed=StageState(BINDING,resume=original,checkpoint=snapshots.append)
    assert len(snapshots)==1 and snapshots[0]['reservation'] is None
    assert resumed.state['spent']==13 and resumed.state['paid_calls']==2
    assert resumed.state['calls'][-1]['metadata']['error_class']=='dispatch_outcome_unknown_after_resume'
    again=StageState(BINDING,resume=resumed.state)
    assert again.state==resumed.state and original['reservation'] is not None


def test_scope_request_body_image_context_model_provider_and_prompt_mismatches_refused():
    original=StageState(BINDING).state
    for key in BINDING:
        other=deepcopy(BINDING);other[key]=other[key]+1 if isinstance(other[key],int) else 'different'
        with pytest.raises(RecoveryStateError):StageState(other,resume=original)


@pytest.mark.parametrize('field,value',[('spent',-1),('spent',True),('spent',float('nan')),
    ('paid_calls',21),('content_failures',5),('observed_tokens',1),('known_rejections',-1)])
def test_corrupt_or_inconsistent_counters_fail_closed(field,value):
    state=StageState(BINDING).state;state[field]=value
    with pytest.raises(RecoveryStateError):StageState(BINDING,resume=state)


def test_scene_schema_and_sha_validated_and_known_scene_cannot_be_replaced():
    value=StageState(BINDING);value.update(scene=scene())
    assert StageState(BINDING,resume=value.state).state['scene']==scene()
    for key,replacement in (('payload_sha256','0'*64),('review_reason',[])):
        bad=value.state;bad['scene'][key]=replacement
        with pytest.raises(RecoveryStateError):StageState(BINDING,resume=bad)
    bad=value.state;bad['scene']['payload']['objects']=[dict(name='x',count=0,attributes=[])]
    bad['scene']['payload_sha256']=digest(bad['scene']['payload'])
    with pytest.raises(RecoveryStateError):StageState(BINDING,resume=bad)
    with pytest.raises(RecoveryStateError):value.update(scene=None)


def test_only_non_budget_fields_can_update_and_all_values_stay_json_bounded():
    value=StageState(BINDING)
    value.update(localization={'artificial':True},tasks=[dict(id='one')],results={'one':{}},
        coverage={'complete':True},final_payload={'artificial':True},final_usage={'completion_tokens':1})
    for fields in (dict(spent=0),dict(calls=[]),dict(binding=BINDING),dict(unknown=True),dict(tasks={}),
                   dict(results=[]),
                   dict(coverage={'x':'x'*(2*1024*1024)})):
        with pytest.raises(RecoveryStateError):value.update(**fields)


def test_literal_image_uri_text_in_evidence_is_preserved_but_binding_image_payload_is_refused():
    value=StageState(BINDING)
    evidence=scene();evidence['payload']['description']='Artificial visible label: data:image/png;base64,example'
    evidence['payload_sha256']=digest(evidence['payload'])
    value.update(scene=evidence,final_payload={'ocr_text':'data:image/ is printed source text'})
    assert StageState(BINDING,resume=value.state).state['scene']==evidence
    for binding in (dict(BINDING,image_data_uri='data:image/png;base64,example'),
                    dict(BINDING,other='data:image/png;base64,example'),dict(BINDING,messages=[])):
        with pytest.raises(RecoveryStateError):StageState(binding)


def test_callback_failure_poison_blocks_all_later_dispatch():
    fail=False;persisted=[]
    def save(state):
        if fail:raise RuntimeError('artificial checkpoint unavailable')
        persisted.append(state)
    value=StageState(BINDING,checkpoint=save);fail=True
    with pytest.raises(RuntimeError):value.reserve('primary',10)
    with pytest.raises(RecoveryStateError,match='new dispatch forbidden'):value.reserve('primary',1)
    assert persisted[-1]['reservation'] is None


def test_budget_state_and_callback_errors_cannot_be_caught_as_ocr_value_errors():
    assert not isinstance(RecoveryBudgetExhausted('artificial'),(ValueError,TypeError))
    value=StageState(BINDING)
    def fail(_):raise ValueError('artificial persistence input rejection')
    value._checkpoint=fail
    with pytest.raises(RecoveryStateError,match='persistence failed') as exc:value.reserve('primary',10)
    assert not isinstance(exc.value,(ValueError,TypeError))
    assert isinstance(exc.value.__cause__,ValueError)


def test_legacy_grant_explicit_persistent_and_never_reissued_by_resume():
    snapshots=[];value=StageState(BINDING,legacy=True,checkpoint=snapshots.append)
    assert snapshots[0]['historical_cost_unknown'] is True and snapshots[0]['single_new_policy_grant'] is True
    value.reserve('primary',16384);value.reject(InferenceUnavailable('ReadError',unknown_outcome=True))
    for _ in range(4):
        value=StageState(BINDING,resume=value.state,legacy=True)
        assert value.state['historical_cost_unknown'] is True
        with pytest.raises(RecoveryBudgetExhausted):value.reserve('primary',1)


def test_content_failure_limit_and_paid_call_limit_are_cumulative():
    value=StageState(BINDING)
    for _ in range(4):value.content_failure()
    with pytest.raises(RecoveryBudgetExhausted):value.reserve('primary',1)
    with pytest.raises(RecoveryBudgetExhausted):value.content_failure()
    value=StageState(BINDING)
    for _ in range(20):accepted(value,tokens=0,limit=1)
    assert value.state['spent']==0 and value.state['paid_calls']==20
    with pytest.raises(RecoveryBudgetExhausted):value.reserve('crop:1',1)


def test_many_known_rejections_cannot_evict_paid_history_or_permanently_consume_calls():
    value=StageState(BINDING);accepted(value,3,10)
    for _ in range(80):
        value.reserve('localization',1);value.reject(InferenceRateLimited())
    assert value.state['spent']==3 and value.state['paid_calls']==1 and value.state['known_rejections']==80
    assert len(value.state['calls'])==20 and value.state['calls'][0]['stage']=='primary'
    value=StageState(BINDING,resume=value.state)
    for _ in range(19):accepted(value,tokens=0,limit=1,stage='crop:1')
    assert len(value.state['calls'])==20 and all(not r['known_rejection'] for r in value.state['calls'])


def test_no_new_dispatch_after_final_receipt():
    value=StageState(BINDING);accepted(value)
    value.update(complete_receipt={'artificial_validated_by_owner':True})
    with pytest.raises(RecoveryStateError,match='completed'):value.reserve('primary',10)


def test_completed_payload_automatic_fingerprint_rejects_resume_tampering_and_manual_hash_update():
    value=StageState(BINDING);value.update(final_payload={'ocr_text':'artificial literal text'})
    assert value.state['final_payload_sha256']==digest(value.state['final_payload'])
    bad=value.state;bad['final_payload']['ocr_text']='changed'
    with pytest.raises(RecoveryStateError,match='fingerprint'):StageState(BINDING,resume=bad)
    with pytest.raises(RecoveryStateError):value.update(final_payload_sha256='0'*64)
    with pytest.raises(RecoveryStateError):value.update(final_payload=None)
