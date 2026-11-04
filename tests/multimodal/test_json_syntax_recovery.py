"""A repair may change punctuation, never complete missing visual evidence."""
import json

import pytest

from mandol_competition.multimodal.vlm.json_syntax import complete_object, JSONSyntaxError, POLICY
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError, VisionOutputTruncated
from .test_ocr_component_recovery import client_for, reply
from .test_ocr_lines_wire import wire_payload


@pytest.mark.parametrize('content', [
    '{"label":"单位 12", "values":[1,2,],}',
    '{"label":"单位 12" "values":[1 2]}',
    '```json\n{"label":"单位 12", "values":[1,2]}\n```',
    '```JSON\n{"label":"单位 12" "values":[1,2,],}\n```',
])
def test_complete_values_survive_syntax_recovery(content):
    value, receipt = complete_object(content)
    assert value == {'label':'单位 12','values':[1,2]}
    assert receipt['policy'] == POLICY


@pytest.mark.parametrize('content', [
    '{"label":"unfinished', '{"label":"complete"', '{"label":}',
    '{"values":[1,2}', '{"values":[1,,2]}', '{label:"text"}',
    '{"label":"first","label":"second",}', '{"value":NaN}',
    '{"value":Infinity}', '{"value":1e}', '{"value":truefalse}',
    '{"label":"line\nline"}', '{"value":1}{"value":2}',
    'Explanation: {"value":1}', '{"value":1} trailing prose',
])
def test_missing_or_ambiguous_evidence_is_never_invented(content):
    with pytest.raises(JSONSyntaxError):
        complete_object(content)


def test_strings_and_repeated_ocr_values_are_unchanged():
    source={'label':'a,} [ ] " \\  单位','lines':['Unit 12','Unit 12',r'code\nline']}
    encoded=json.dumps(source,ensure_ascii=False)
    recovered,receipt=complete_object(encoded[:-1]+',}')
    assert recovered == source and receipt['trailing_commas_removed']==1
    assert complete_object(encoded)==(source,None)


def test_stopped_complete_scene_is_repaired_without_another_model_request():
    source=wire_payload()
    malformed=json.dumps(source)[:-1]+',}'
    client,seen=client_for(reply(malformed,encoded=True))
    try:
        result=client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:
        client.close()
    assert len(seen)==1
    assert result.output_validation['json_syntax_repair']['policy']==POLICY
    assert result.payload['objects']==source['objects']
    assert result.payload['ocr_text']=='\n'.join(source['ocr_lines'])


def test_valid_looking_json_with_length_finish_cannot_be_syntax_repaired_to_success():
    # Even a balanced response cannot prove all non-OCR evidence completed.
    client,seen=client_for(reply('{"description":"short",}',reason='length',tokens=4096,encoded=True))
    try:
        with pytest.raises(VisionOutputTruncated):
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:
        client.close()
    assert len(seen)==1


def test_repaired_json_still_must_satisfy_the_exact_visual_schema():
    client,seen=client_for(reply('{"description":"short",}',encoded=True))
    try:
        with pytest.raises(VisionOutputError):
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:
        client.close()
    assert len(seen)==1
