"""Verified readability survives cache replay; synthetic media, no DB/models."""
import base64
from copy import deepcopy
from dataclasses import asdict
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from mandol_substrate.state import Scope
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.protocol import AddRequest
from mandol_competition.multimodal.service import AddService, PreparationLedger, _cached_extractor
from mandol_competition.multimodal.vlm.base import Extraction, ExtractionError
from mandol_competition.multimodal.vlm.cache import CachedResult, _media_id, context_digest
from mandol_competition.multimodal.vlm.ocr_lines import PROMPT_VERSION
from mandol_competition.multimodal.vlm.output_guard import (
    OCR_COVERAGE_POLICY, OUTPUT_POLICY, REGION_COMPOSITE_POLICY, extraction_fingerprint,
    validation_receipt_valid, verified_ocr_coverage,
)


def image_uri():
    out=BytesIO()
    with Image.new('RGB',(20,20),'white') as image:image.save(out,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()


def coverage(status='unreadable',unreadable=1):
    return dict(policy=OCR_COVERAGE_POLICY,status=status,unreadable_regions=unreadable,completed_crops=1)


def receipt(extraction,inputs,*,info=None):
    stages=[dict(component=component,complete=True,finish_reason='stop',output_token_budget=1024,
        completion_tokens=10,**({'crop_id':'crop-0'} if component=='ocr_crop' else {}))
        for component in ('scene','ocr_locator','ocr_crop')]
    value=dict(policy=REGION_COMPOSITE_POLICY,finish_reason='stop',output_token_budget=16384,
        completion_tokens=30,stages=stages,media_id=_media_id(inputs['image_data_uri']),
        context_sha256=context_digest(inputs['context_text']),image_position=inputs['image_position'],
        image_count=inputs['image_count'],extraction_sha256=extraction_fingerprint(extraction))
    if info is not None:value['ocr_coverage']=deepcopy(info)
    return value


class NoModel:
    model_name='gpt-4o-mini';supports_resumable_ocr=True
    def extract(self,**kwargs):raise AssertionError('must use the verified cache')
    def extract_resumable(self,**kwargs):raise AssertionError('must use the verified cache')


class MemoryLedger:
    def __init__(self,store):self.store=store
    def reserve(self,scope,rid,body):
        self.store.setdefault((scope.key,rid),dict(body=body,value={}))
        assert self.store[scope.key,rid]['body']==body
        return deepcopy(self.store[scope.key,rid]['value'])
    def checkpoint(self,scope,rid,body,value):
        assert self.store[scope.key,rid]['body']==body
        self.store[scope.key,rid]['value']=deepcopy(value)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    import mandol_competition.multimodal.service as service
    monkeypatch.setattr(service,'RequestLedger',MemoryLedger)
    config=MultimodalConfig(data_dir=tmp_path,prompt_version=PROMPT_VERSION,vision_failure_policy='retry-required-v1')
    inner=NoModel();vision=_cached_extractor(config,'user',inner)
    inputs=dict(image_data_uri=image_uri(),context_text='Artificial caption',image_position=0,image_count=1)
    extraction=Extraction(description='Artificial scene',model='gpt-4o-mini',prompt_version=PROMPT_VERSION,schema_version='1')
    store={};backend=SimpleNamespace(state=store,profile=SimpleNamespace(options={}),scope=lambda user:Scope('mm',user))
    def ledger(rid='one'):
        return PreparationLedger(backend,SimpleNamespace(user_id='user',request_id=rid),'a'*64,strict=True)
    def put(value):
        media=_media_id(inputs['image_data_uri']);key=vision._key(media,inputs['context_text'],0,1)
        vision._cache.put(media,key,extraction,output_validation=value)
        return media,key
    return SimpleNamespace(config=config,inner=inner,vision=vision,inputs=inputs,extraction=extraction,
        store=store,backend=backend,ledger=ledger,put=put)


def test_new_request_cache_replay_retains_verified_unreadability_without_new_stage_grant(prepared):
    p=prepared;proof=receipt(p.extraction,p.inputs,info=coverage());p.put(proof)
    for rid in ('one','two'):
        ledger=p.ledger(rid)
        assert ledger.extract(key='0:1',vision=p.vision,**p.inputs)==p.extraction
        entry=p.ledger(rid).value['vision']['0:1']
        assert entry['ocr_coverage']==coverage() and entry['output_validation']==proof
        assert 'stage_checkpoint' not in entry and entry['status']=='complete'
    cached=p.vision.get_cached_result(**p.inputs)
    assert isinstance(cached,CachedResult) and cached.output_validation==proof
    cached.output_validation['ocr_coverage']['status']='changed'
    assert p.vision.get_cached_result(**p.inputs).output_validation==proof


def request_for(p,rid='one'):
    return AddRequest.model_validate(dict(user_id='user',request_id=rid,session_id='s',messages=[dict(role='user',content=[
        dict(type='text',text=p.inputs['context_text']),dict(type='image_url',image_url=dict(url=p.inputs['image_data_uri']))])]))


def test_prepare_units_carries_cache_receipt_readability_into_canonical_metadata(prepared):
    p=prepared;p.put(receipt(p.extraction,p.inputs,info=coverage()))
    service=AddService(p.config,extractor=p.inner,backend=p.backend)
    units=service.prepare_units(request_for(p),preparation=p.ledger())
    assert len(units)==1 and units[0].metadata['visual_ocr_readability']==[dict(image_position=0,**coverage())]
    assert units[0].raw_data['multimodal_info']['images'][0]['ocr_text']==''
    assert units[0].raw_data['multimodal_info']['images'][0]['description']==p.extraction.description


def test_prior_complete_entry_and_untrusted_stage_coverage_cannot_invent_readability(prepared):
    p=prepared;legacy=receipt(p.extraction,p.inputs);p.put(legacy)
    ledger=p.ledger();ledger.value['vision']={'0:1':dict(status='complete',extraction=asdict(p.extraction),
        ocr_coverage=coverage(),stage_checkpoint=dict(coverage=coverage()))};ledger.save()
    units=AddService(p.config,extractor=p.inner,backend=p.backend).prepare_units(request_for(p),preparation=p.ledger())
    assert 'visual_ocr_readability' not in units[0].metadata
    assert p.vision.get_cached_result(**p.inputs).extraction==p.extraction  # Legacy receipt still usable.


@pytest.mark.parametrize('changes',[
    dict(policy='other'),dict(status='failed'),dict(status=[]),dict(unreadable_regions=True),
    dict(unreadable_regions=-1),dict(unreadable_regions=33),dict(unreadable_regions=0),
    dict(completed_crops=True),dict(completed_crops=0),dict(completed_crops=2),dict(extra='untrusted'),
])
def test_invalid_coverage_is_not_a_valid_success_receipt(prepared,changes):
    p=prepared;info={**coverage(),**changes};proof=receipt(p.extraction,p.inputs,info=info)
    assert not validation_receipt_valid(proof)
    with pytest.raises(ExtractionError):p.put(proof)


@pytest.mark.parametrize('field,value',[
    ('media_id','img_'+'0'*64),('context_sha256','0'*64),('image_position',1),('image_count',2),
    ('extraction_sha256','0'*64),('image_position',False),
])
def test_cache_readability_cannot_cross_image_context_position_or_fingerprint(prepared,field,value):
    p=prepared;proof=receipt(p.extraction,p.inputs,info=coverage());media,key=p.put(proof)
    raw=p.vision._cache._read_record(media,key);raw['output_validation'][field]=value
    p.vision._cache._write_record(media,key,raw)
    assert p.vision.get_cached_result(**p.inputs) is None


def test_other_user_cache_root_is_separate_and_old_three_field_result_is_compatible(prepared):
    p=prepared;p.put(receipt(p.extraction,p.inputs,info=coverage()))
    other=_cached_extractor(p.config,'other-user',p.inner)
    assert other.get_cached_result(**p.inputs) is None
    assert CachedResult(p.extraction,None,None).output_validation is None


def test_primary_receipt_cannot_claim_recovery_unreadability(prepared):
    p=prepared;proof=receipt(p.extraction,p.inputs,info=coverage());proof['policy']=OUTPUT_POLICY
    assert not validation_receipt_valid(proof)


def test_status_is_bound_to_actual_literal_ocr_and_legacy_absence_remains_unknown(prepared):
    p=prepared;identity=dict(media_id=_media_id(p.inputs['image_data_uri']),context_sha256=context_digest(p.inputs['context_text']),
        image_position=0,image_count=1)
    readable=receipt(p.extraction,p.inputs,info=coverage('readable',0))
    assert verified_ocr_coverage(readable,p.extraction,**identity) is None
    assert verified_ocr_coverage(receipt(p.extraction,p.inputs),p.extraction,**identity) is None
    with pytest.raises(ExtractionError):p.put(readable)
