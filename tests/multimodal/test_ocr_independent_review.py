"""Replay the observed failure shapes with artificial images and paid-call fixtures."""
from copy import deepcopy

import pytest

from mandol_competition.multimodal.vlm import staged_ocr as staged, ocr_review
from mandol_competition.multimodal.vlm.ocr_recovery import validate_crop_result
from mandol_competition.multimodal.vlm.output_guard import REGION_REVIEW_POLICY, VisionOutputError, validation_receipt_valid
from mandol_competition.multimodal.vlm.stage_state import StageState, RecoveryBudgetExhausted, RecoveryStateError
from .test_staged_ocr_bounded_scene import FixtureClient, inputs, scene, localization, crop, response, names
from .test_staged_ocr_pipeline import uri


def no_text():
    return dict(inspected_entire_crop=True, assessment='no_text', lines=[], unreadable_regions=[])


def review(assessment='no_text'):
    return dict(inspected_entire_image=True, assessment=assessment)


def accepted(state, stage, tokens, finish='stop'):
    state.reserve(stage, 1024)
    meta = dict(finish_reason=finish, completion_tokens=tokens, output_token_budget=1024)
    usage = dict(prompt_tokens=10, completion_tokens=tokens, total_tokens=tokens+10)
    state.accept(meta, usage)
    return meta, usage


def saved_scene(client, data):
    state = StageState(staged.binding_for(client, data['prompt'], data['image_uri'],
        data['context'], data['position'], data['count'], None))
    meta, usage = accepted(state, 'scene', 1024, 'length')
    value = scene()
    state.update(scene=dict(payload=value, payload_sha256=staged.digest(value),
        metadata=meta, usage=usage, review_reason='truncated_ocr'))
    return state


def locator_exhausted(client, data):
    state = saved_scene(client, data)
    for _ in range(2):
        accepted(state, 'ocr_locator', 258)
        state.content_failure()
    value = state.state
    value.pop('ocr_repair')  # Exact pre-fix schema must still resume.
    return value


def contradictory_plan(client, data):
    state = saved_scene(client, data)
    meta, usage = accepted(state, 'ocr_locator', 61)
    tasks = staged.recovery_crops((200, 600), localization(), 'truncated_ocr')
    accepted(state, 'ocr_crop', 40)
    results = {}
    for task in tasks:
        cm, cu = accepted(state, 'ocr_crop', 23)
        checked = validate_crop_result(task, no_text(), finish_reason='stop')
        results[task.crop_id] = dict(validated=checked.to_payload(), metadata=cm, usage=cu)
    for _ in range(4):
        state.content_failure()
    state.update(localization=dict(wire=localization(), metadata=meta, usage=usage),
        tasks=[t.to_payload() for t in tasks], results=results)
    value = state.state
    value.pop('ocr_repair')
    return value


def test_historical_two_invalid_locators_resume_without_repeating_them_or_resetting_costs():
    client = FixtureClient([response(review('text_regions')), response(crop())])
    data = inputs(); old = locator_exhausted(client, data); saved = []
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    assert names(client) == ['ocr_independent_review_v1', 'ocr_physical_crop_lines_v1']
    assert payload['ocr_text'] == 'Literal visible label'
    assert receipt['policy'] == REGION_REVIEW_POLICY and validation_receipt_valid(receipt)
    assert saved[-1]['scene'] == old['scene'] and saved[-1]['calls'][:3] == old['calls']
    assert saved[-1]['content_failures'] == 2 and saved[-1]['spent'] == 1540+80
    assert saved[-1]['paid_calls'] == 5 and saved[-1]['binding'] == old['binding']
    assert all(c['options']['max_completion_tokens'] == 1024 for c in client.calls)


@pytest.mark.parametrize('assessment', ['no_text', 'text_regions'])
def test_historical_four_failures_reinspect_and_replace_contradictory_blank_plan(assessment):
    data = inputs(); data['image_uri'] = uri(size=(200, 600))
    fresh = [response(no_text(), tokens=30)]*2 if assessment == 'no_text' else [
        response(crop(), tokens=30), response(crop(), tokens=30)]
    client = FixtureClient([response(review(assessment), tokens=20), *fresh])
    old = contradictory_plan(client, data); saved = []
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    assert names(client) == ['ocr_independent_review_v1'] + ['ocr_physical_crop_lines_v1']*2
    assert payload['ocr_text'] == ('' if assessment == 'no_text' else 'Literal visible label\nLiteral visible label')
    assert saved[-1]['content_failures'] == 4 and saved[-1]['spent'] == 1171+80
    assert saved[-1]['paid_calls'] == 8 and saved[-1]['calls'][:5] == old['calls']
    assert all(r['metadata']['completion_tokens'] == 30 for r in saved[-1]['results'].values())
    assert saved[-1]['scene'] == old['scene']
    assert receipt['stages'][1]['component'] == 'ocr_review' and validation_receipt_valid(receipt)


def test_new_invalid_locator_enters_different_review_schema_in_same_request():
    client = FixtureClient([response(dict(scene(), ocr_lines=['repeat']*64)),
        response({'invalid': 'artificial unusable region inventory'}), response(review()), response(no_text())])
    payload, _, receipt, _ = staged.complete_staged(client, **inputs())
    assert names(client) == ['visual_memory_ocr_lines_v2','ocr_physical_text_localization_v1',
        'ocr_independent_review_v1','ocr_physical_crop_lines_v1']
    assert payload['ocr_text'] == '' and validation_receipt_valid(receipt)


def test_single_small_crop_blank_conflict_uses_review_instead_of_refinement_or_permanent_failure():
    data = inputs(); data['image_uri'] = uri(size=(100, 40))
    client = FixtureClient([response(dict(scene(), ocr_lines=['repeat']*64)),
        response(localization()), response(no_text()), response(review()), response(no_text())])
    payload, _, receipt, _ = staged.complete_staged(client, **data)
    assert names(client) == ['visual_memory_ocr_lines_v2', 'ocr_physical_text_localization_v1',
        'ocr_physical_crop_lines_v1', 'ocr_independent_review_v1', 'ocr_physical_crop_lines_v1']
    assert payload['ocr_text'] == '' and receipt['policy'] == REGION_REVIEW_POLICY
    assert validation_receipt_valid(receipt)


def test_uncertain_review_never_becomes_empty_ocr_and_cannot_repeat_forever():
    client = FixtureClient([response(review('uncertain')), response(review('uncertain'))])
    data = inputs(); state = locator_exhausted(client, data); saved = []
    for _ in range(2):
        with pytest.raises(VisionOutputError):
            staged.complete_staged(client, **data, resume=state, checkpoint=saved.append)
        state = saved[-1]
        assert state['final_payload'] is None and state['complete_receipt'] is None
    before = deepcopy(state)
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=state, checkpoint=saved.append)
    assert len(client.calls) == 2 and saved[-1] == before


def test_two_independent_text_detections_with_blank_rereads_fail_boundedly_not_cached_forever():
    data = inputs(); data['image_uri'] = uri(size=(200, 600))
    responses = [response(review('text_regions')), response(no_text()), response(no_text())]*2
    client = FixtureClient(responses); state = contradictory_plan(client, data); saved = []
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=state, checkpoint=saved.append)
    assert len(client.calls) == 6 and saved[-1]['final_payload'] is None
    for _ in range(3):
        with pytest.raises(RecoveryBudgetExhausted):
            staged.complete_staged(client, **data, resume=saved[-1], checkpoint=saved.append)
    assert len(client.calls) == 6 and saved[-1]['content_failures'] == 4


def test_transient_paid_failure_resumes_review_plan_without_repeating_completed_review():
    class UnknownPaid(RuntimeError): unknown_outcome = True
    client = FixtureClient([response(review('text_regions')), UnknownPaid('artificial'), response(crop())])
    data = inputs(); state = locator_exhausted(client, data); saved = []
    with pytest.raises(UnknownPaid):
        staged.complete_staged(client, **data, resume=state, checkpoint=saved.append)
    after_failure = saved[-1]
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=after_failure, checkpoint=saved.append)
    assert names(client) == ['ocr_independent_review_v1','ocr_physical_crop_lines_v1','ocr_physical_crop_lines_v1']
    assert saved[-1]['spent'] == 1540+40+1024+40 and saved[-1]['paid_calls'] == 6
    assert payload['ocr_text'] and validation_receipt_valid(receipt)


def test_checkpoint_failure_starting_repair_prevents_any_new_paid_dispatch():
    client = FixtureClient([]); data = inputs(); old = locator_exhausted(client, data)
    def reject(_): raise OSError('artificial disk failure')
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client, **data, resume=old, checkpoint=reject)
    assert not client.calls


@pytest.mark.parametrize('kind', ['tokens','calls'])
def test_repair_does_not_refresh_original_total_token_or_paid_call_limits(kind):
    client = FixtureClient([]); data = inputs(); old = locator_exhausted(client, data)
    state = StageState(old['binding'], resume=old)
    for _ in range(14 if kind == 'tokens' else 17):
        accepted(state, 'historical-artificial-paid', 1024 if kind == 'tokens' else 1)
    old = state.state; saved = []
    with pytest.raises(RecoveryBudgetExhausted):
        staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    assert not client.calls and saved[-1]['spent'] == old['spent']
    assert saved[-1]['paid_calls'] == old['paid_calls'] and saved[-1]['content_failures'] == 2


def test_changed_input_binding_cannot_enter_new_repair_grant():
    client = FixtureClient([]); data = inputs(); old = locator_exhausted(client, data)
    data['context'] += 'changed'
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client, **data, resume=old)
    assert not client.calls


def test_review_429_refunds_only_current_reservation_and_preserves_retry_allowance():
    from mandol_competition.inference.errors import InferenceRateLimited
    client = FixtureClient([InferenceRateLimited(retry_after=60), response(review()), response(no_text())])
    data = inputs(); old = locator_exhausted(client, data); saved = []
    with pytest.raises(InferenceRateLimited):
        staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    assert saved[-1]['spent'] == old['spent'] and saved[-1]['paid_calls'] == old['paid_calls']
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=saved[-1], checkpoint=saved.append)
    assert payload['ocr_text'] == '' and validation_receipt_valid(receipt)
    assert saved[-1]['known_rejections'] == 1 and saved[-1]['paid_calls'] == 5


def test_paid_unknown_review_can_retry_once_but_keeps_its_full_charge():
    class UnknownPaid(RuntimeError): unknown_outcome = True
    client = FixtureClient([UnknownPaid('artificial'), response(review()), response(no_text())])
    data = inputs(); old = locator_exhausted(client, data); saved = []
    with pytest.raises(UnknownPaid):
        staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    assert saved[-1]['spent'] == old['spent']+1024
    _, _, receipt, _ = staged.complete_staged(client, **data, resume=saved[-1], checkpoint=saved.append)
    assert saved[-1]['spent'] == old['spent']+1024+80 and validation_receipt_valid(receipt)


def test_completed_repair_payload_replays_without_new_dispatch():
    client = FixtureClient([response(review()), response(no_text())])
    data = inputs(); old = locator_exhausted(client, data); saved = []
    result = staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    again = staged.complete_staged(client, **data, resume=saved[-1])
    assert result[:3] == again[:3] and len(client.calls) == 2


def test_no_text_review_cannot_suppress_text_observed_by_independent_crops():
    client = FixtureClient([response(review()), response(crop())])
    data = inputs(); old = locator_exhausted(client, data)
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=old)
    assert payload['ocr_text'] == 'Literal visible label'
    assert receipt['ocr_coverage']['status'] == 'readable'


def test_process_loss_during_review_consumes_reserved_charge_once_on_resume():
    client = FixtureClient([SystemExit('artificial process loss'), response(review()), response(no_text())])
    data = inputs(); old = locator_exhausted(client, data); saved = []
    with pytest.raises(SystemExit):
        staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    interrupted = saved[-1]
    assert interrupted['reservation']['stage'] == ocr_review.review_stage()
    _, _, receipt, _ = staged.complete_staged(client, **data, resume=interrupted, checkpoint=saved.append)
    assert saved[-1]['spent'] == 1540+1024+80 and saved[-1]['paid_calls'] == 6
    assert saved[-1]['calls'][3]['metadata']['error_class'] == 'dispatch_outcome_unknown_after_resume'
    assert validation_receipt_valid(receipt)


def test_reviewed_multiple_crops_resume_only_the_unfinished_crop_and_keep_all_source_text():
    class UnknownPaid(RuntimeError): unknown_outcome = True
    data = inputs(); data['image_uri'] = uri(size=(200, 600))
    client = FixtureClient([response(review('text_regions')), response(crop()),
        UnknownPaid('artificial'), response(crop())])
    old = contradictory_plan(client, data); saved = []
    with pytest.raises(UnknownPaid):
        staged.complete_staged(client, **data, resume=old, checkpoint=saved.append)
    first_saved = deepcopy(saved[-1]['results'])
    payload, _, receipt, _ = staged.complete_staged(client, **data, resume=saved[-1], checkpoint=saved.append)
    assert len(client.calls) == 4 and len(first_saved) == 1
    assert all(saved[-1]['results'][k] == v for k, v in first_saved.items())
    assert payload['ocr_text'] == 'Literal visible label\nLiteral visible label'
    assert validation_receipt_valid(receipt)


@pytest.mark.parametrize('change', [dict(reason=[]), dict(failures=-1), dict(failures=3), dict(policy='unknown')])
def test_malformed_repair_metadata_fails_closed_before_any_paid_request(change):
    client = FixtureClient([]); data = inputs(); old = locator_exhausted(client, data)
    old['ocr_repair'] = dict(policy=ocr_review.policy(), reason='locator_rejected', failures=0, review=None)
    old['ocr_repair'].update(change)
    with pytest.raises(RecoveryStateError):
        staged.complete_staged(client, **data, resume=old)
    assert not client.calls


def test_repair_stage_requires_a_durable_grant_even_with_unused_content_budget():
    state = StageState({'artificial_binding': True})
    with pytest.raises(RecoveryBudgetExhausted):
        state.reserve(ocr_review.review_stage(), 1024)
    assert state.state['paid_calls'] == state.state['spent'] == 0


def test_repair_grant_and_its_failure_counter_cannot_be_cleared_or_reset():
    client = FixtureClient([]); data = inputs(); old = locator_exhausted(client, data)
    state = StageState(old['binding'], resume=old)
    grant = dict(policy=ocr_review.policy(), reason='locator_rejected', failures=0, review=None)
    state.update(ocr_repair=grant); state.repair_failure()
    for value in (None, grant, dict(grant, policy='different-recipe')):
        with pytest.raises(RecoveryStateError):
            state.update(ocr_repair=value)
    assert state.state['ocr_repair']['failures'] == 1 and state.state['content_failures'] == 2
