"""Only completed typed components may survive an OCR-only recovery."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mandol_competition.inference.errors import InferenceDeadlineExceeded, InferenceRateLimited
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient, VisionRequestError
from mandol_competition.multimodal.vlm.ocr_lines import (FORMAT_POLICY, canonical_payload,
    completed_scene_prefix, sequential_numbers_need_review)
from mandol_competition.multimodal.vlm.output_guard import (COMPOSITE_POLICY,
    VisionOutputError, VisionOutputTruncated, validation_receipt_valid)
from .test_ocr_lines_wire import wire_payload


def reply(value, *, reason='stop', tokens=300, encoded=False):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,message=SimpleNamespace(
        content=value if encoded else json.dumps(value),refusal=None))],
        usage=SimpleNamespace(prompt_tokens=100,completion_tokens=tokens,total_tokens=100+tokens))


def ocr_reply(lines, **kwargs):
    return reply({'regions':[{'location':'visible text-bearing object','text':'\n'.join(lines)}]},**kwargs)


def client_for(*responses):
    seen = []
    iterator = iter(responses)
    def create(**kwargs):
        seen.append(kwargs)
        value = next(iterator)
        if isinstance(value,BaseException):raise value
        return value
    sdk=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client=OpenAICompatVisionClient(model='owned-mini',api_key='unused',base_url='https://owned.test/v1',
        timeout=2,client_factory=lambda:sdk,output_tokens=4096,max_output_tokens=8192,wire_format=FORMAT_POLICY)
    return client,seen


def unfinished_ocr():
    scene = {k:v for k,v in wire_payload().items() if k!='ocr_lines'}
    scene['description'] += ' A label literally contains "ocr_lines": [] as printed code.'
    return scene,json.dumps(scene)[:-1]+', "ocr_lines": ["Neay'+r'\n'*4000


def test_truncated_ocr_is_reread_without_fabricating_json_or_losing_completed_scene_facts():
    scene,unfinished=unfinished_ocr()
    client,seen=client_for(reply(unfinished,reason='length',tokens=4096,encoded=True),
        ocr_reply(['Neay','Unit 12','Unit 12'],tokens=40))
    try:
        result=client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:client.close()
    assert len(seen)==2 and [r['max_completion_tokens'] for r in seen]==[4096,8192]
    assert seen[1]['response_format']['json_schema']['name']=='visual_memory_literal_ocr_regions_v1'
    assert all(r['messages'][-1]['content'][-1]['image_url']['url']=='data:image/png;base64,QUJD' for r in seen)
    assert result.payload=={**scene,'ocr_text':'Neay\nUnit 12\nUnit 12'}
    assert result.usage=={'prompt_tokens':200,'completion_tokens':4136,'total_tokens':4336}
    assert result.output_validation['policy']==COMPOSITE_POLICY
    assert validation_receipt_valid(result.output_validation)


def test_incomplete_non_ocr_values_cannot_be_completed_by_the_recovery_parser():
    scene,unfinished=unfinished_ocr()
    for content in ('{"description":"unfinished', '{"ocr_lines":["Neay',
            '{"description":"owned", "objects":[{"name":"unfinished',
            '{"description":"owned", "objects":[1e, "ocr_lines":[]'):
        with pytest.raises((ValueError,TypeError)):
            completed_scene_prefix(content)
        client,seen=client_for(reply(content,reason='length',tokens=4096,encoded=True))
        try:
            with pytest.raises(VisionOutputTruncated):
                client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
            assert len(seen)==1
        finally:client.close()


def test_sequential_numbers_request_review_and_use_only_the_fresh_literal_ocr():
    source=wire_payload();source['ocr_lines']=[str(n) for n in range(1,81)]
    assert sequential_numbers_need_review(source['ocr_lines'])
    assert not sequential_numbers_need_review(['Unit 12','Unit 12','A','B'])
    actual_lines=['1 2 3 4 5 6 7','8 9 10 11 12 13 14','15 16 17 18 19 20 21',
        '22 23 24 25 26 27 28','29 30 31','Unit 12','Unit 12']
    client,seen=client_for(reply(source,tokens=700),ocr_reply(actual_lines,tokens=160))
    try:
        result=client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:client.close()
    assert len(seen)==2
    assert result.payload==canonical_payload(dict(source,ocr_lines=actual_lines))
    assert result.output_validation['ocr_review_reason']=='sequential_numbers'
    assert validation_receipt_valid(result.output_validation)


@pytest.mark.parametrize('lines',[[str(n) for n in range(1,81)],
    [' '.join(str(n) for n in range(1,81))],['Repeated label']*80])
def test_persistent_ocr_runaway_cannot_be_a_successful_extraction(lines):
    source=wire_payload();source['ocr_lines']=lines
    client,seen=client_for(reply(source,tokens=700),ocr_reply(lines,tokens=700))
    try:
        with pytest.raises(VisionOutputError) as caught:
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:client.close()
    assert len(seen)==2 and not caught.value.retryable
    assert caught.value.error_class=='vision_output_ocr_repetition'
    from mandol_competition.observability.diagnostics import _exception_metadata
    safe=_exception_metadata(caught.value)
    assert safe['inference_error_class']=='vision_output_ocr_repetition'
    assert 'Repeated label' not in json.dumps(safe)


def test_normal_calendar_rows_and_genuine_duplicates_have_no_arbitrary_text_quota():
    from mandol_competition.multimodal.vlm.ocr_lines import ocr_needs_review
    month=[' '.join(str(n) for n in range(start,min(start+7,32))) for start in range(1,32,7)]
    assert not ocr_needs_review(month*12)
    assert not ocr_needs_review(['Unit 12','Unit 12','abc']*10)


def test_region_ocr_preserves_order_indentation_duplicates_and_code_text():
    from mandol_competition.multimodal.vlm.ocr_lines import region_lines
    value={'regions':[{'location':'left label','text':'Unit 12\n\nUnit 12'},
        {'location':'right code block','text':'  keep indentation\ncode\\nline\n中文'}]}
    assert region_lines(value)==['Unit 12','Unit 12','  keep indentation',r'code\nline','中文']


def test_a_complete_region_with_runaway_blank_layout_is_not_silently_a_success():
    _,unfinished=unfinished_ocr()
    client,seen=client_for(reply(unfinished,reason='length',tokens=4096,encoded=True),
        ocr_reply(['Neay'+'\n'*100],tokens=200))
    try:
        with pytest.raises(VisionOutputError) as caught:
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:client.close()
    assert len(seen)==2 and caught.value.error_class=='vision_output_ocr_repetition'


@pytest.mark.parametrize('failure',[InferenceDeadlineExceeded(),InferenceRateLimited(retry_after=7)])
def test_rejected_ocr_stage_preserves_the_first_paid_stage_and_is_not_blindly_repeated(failure):
    _,unfinished=unfinished_ocr()
    client,seen=client_for(reply(unfinished,reason='length',tokens=4096,encoded=True),failure)
    try:
        with pytest.raises((InferenceDeadlineExceeded,VisionRequestError)) as caught:
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
        assert len(seen)==2
        if isinstance(caught.value,VisionRequestError):
            assert caught.value.paid_output_failure and not caught.value.retryable
            assert caught.value.status_code==429 and caught.value.retry_after==7
        else:
            assert caught.value.vision_paid_output_failure
    finally:client.close()


def test_a_second_truncated_or_invalid_ocr_response_is_never_a_successful_receipt():
    _,unfinished=unfinished_ocr()
    for second in (reply('{"ocr_lines":["bad',reason='length',tokens=8192,encoded=True),
            reply({'regions':[{'location':'visible text','text':['wrong type']}]},tokens=40)):
        client,seen=client_for(reply(unfinished,reason='length',tokens=4096,encoded=True),second)
        try:
            with pytest.raises(VisionOutputError) as caught:
                client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
            assert not caught.value.retryable and len(seen)==2
        finally:client.close()


def test_composite_receipt_cannot_claim_incomplete_or_unbounded_components():
    _,unfinished=unfinished_ocr()
    client,_=client_for(reply(unfinished,reason='length',tokens=4096,encoded=True),ocr_reply(['Neay'],tokens=15))
    try:
        receipt=client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD').output_validation
    finally:client.close()
    assert validation_receipt_valid(receipt)
    for mutate in (lambda r:r['stages'][0].update(complete=False),
            lambda r:r['stages'][1].update(finish_reason='length'),
            lambda r:r.update(completion_tokens=1),
            lambda r:r.update(output_token_budget=4096)):
        bad=deepcopy(receipt);mutate(bad)
        assert not validation_receipt_valid(bad)
