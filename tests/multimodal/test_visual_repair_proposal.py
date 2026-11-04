from copy import deepcopy
from dataclasses import asdict,replace

import pytest

from mandol_competition.multimodal.unit_builder import build_memory_unit
from mandol_competition.multimodal.render import RenderImage
from mandol_competition.multimodal.representation import BASELINE
from mandol_competition.multimodal.vlm.base import Extraction
from mandol_competition.multimodal.vlm.cache import context_digest
from mandol_competition.multimodal.vlm.output_guard import OUTPUT_POLICY,extraction_fingerprint
from mandol_competition.multimodal.visual_repair import (
    VisualRepairRejected,propose_visual_repair,source_fingerprint)

MEDIA='img_'+'a'*64
OLD=Extraction(model='gpt-4o-mini',prompt_version='multimodal_sgm_v1',schema_version='1')
NEW=replace(OLD,description='An owned blue card.',ocr_text='SERIAL-001\n数量 42')


def fixture():
    unit=build_memory_unit(user_id='owned-user',request_id='owned-add',session_id='owned-session',
        message_index=0,role='user',timestamp_text='2026-01-01',dialogue='Original owned source text.',
        content_parts=[{'type':'text','text':'Original owned source text.'},{'type':'image','media_id':MEDIA}],
        images=[RenderImage(MEDIA,OLD)],extractor_model=OLD.model,
        prompt_version=OLD.prompt_version,schema_version=OLD.schema_version)
    unit.metadata['vlm_degraded']=True
    receipt=dict(policy=OUTPUT_POLICY,finish_reason='stop',output_token_budget=4096,completion_tokens=100,
        media_id=MEDIA,context_sha256=context_digest(unit.raw_data['original_content']),
        image_position=0,image_count=1,extraction_sha256=extraction_fingerprint(NEW))
    return unit,receipt


def proposal(unit,receipt,**kwargs):
    return propose_visual_repair(unit,0,NEW,receipt,expected_source_sha256=source_fingerprint(unit),
        representation_policy=BASELINE,**kwargs)


def test_proposal_is_unapplied_and_preserves_ids_sources_and_input_object():
    unit,receipt=fixture()
    original=deepcopy((unit.raw_data,unit.metadata))
    result=proposal(unit,receipt)
    assert result.applied is False and result.requires_atomic_reindex is True
    assert result.replacement.uid==unit.uid
    for key in ('original_content','content_parts'):
        assert result.replacement.raw_data[key]==unit.raw_data[key]
    assert result.replacement.metadata['request_id']==unit.metadata['request_id']
    assert not result.replacement.metadata.get('vlm_degraded')
    assert result.replacement.raw_data['multimodal_info']['images'][0]['ocr_text']==NEW.ocr_text
    assert 'SERIAL-001' in result.replacement.raw_data['text_content']
    assert (unit.raw_data,unit.metadata)==original


def test_unflagged_empty_requires_separate_review():
    unit,receipt=fixture()
    unit.metadata.pop('vlm_degraded')
    with pytest.raises(VisualRepairRejected):
        proposal(unit,receipt)
    assert proposal(unit,receipt,review_unflagged_empty=True).applied is False


@pytest.mark.parametrize('field,value',[
    ('finish_reason','length'),('completion_tokens',4096),('completion_tokens',None),
    ('media_id','img_'+'b'*64),('context_sha256','wrong'),('image_position',1),('image_count',2),
    ('extraction_sha256','wrong'),
])
def test_incomplete_or_wrong_input_output_cannot_be_staged(field,value):
    unit,receipt=fixture()
    receipt[field]=value
    with pytest.raises(VisualRepairRejected):
        proposal(unit,receipt)


def test_source_drift_healthy_replacement_and_identity_change_are_rejected():
    unit,receipt=fixture()
    old_hash=source_fingerprint(unit)
    unit.metadata['another_writer']='changed'
    with pytest.raises(VisualRepairRejected):
        propose_visual_repair(unit,0,NEW,receipt,expected_source_sha256=old_hash,representation_policy=BASELINE)
    with pytest.raises(VisualRepairRejected):
        propose_visual_repair(unit,0,replace(NEW,model='different-model'),receipt,
            expected_source_sha256=source_fingerprint(unit),representation_policy=BASELINE)
    unit.raw_data['multimodal_info']['images'][0].update(asdict(NEW))
    with pytest.raises(VisualRepairRejected):
        proposal(unit,receipt)


def test_contextual_representation_requires_exact_original_surrounding_context():
    unit,receipt=fixture()
    with pytest.raises(VisualRepairRejected):
        propose_visual_repair(unit,0,NEW,receipt,expected_source_sha256=source_fingerprint(unit),
                              representation_policy='mm-representation-r6-v1')
