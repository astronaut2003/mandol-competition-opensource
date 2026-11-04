"""Artificial model + in-memory ledger/cache integration; no DB or network."""
import base64
from copy import deepcopy
from dataclasses import asdict, replace
from io import BytesIO
import json
import math
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from mandol_substrate.state import Scope
from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.service import PreparationLedger
from mandol_competition.multimodal.vlm.cache import CachingExtractor
from mandol_competition.multimodal.vlm.ocr_lines import FORMAT_POLICY, PROMPT_VERSION
from mandol_competition.multimodal.vlm.openai_compat import (
    OpenAICompatExtractor, OpenAICompatVisionClient, VisionExtractionUnavailable,
)
from mandol_competition.multimodal.vlm.output_guard import validation_receipt_valid


def uri(color='white',size=(200,200)):
    out=BytesIO()
    with Image.new('RGB',size,color) as image:image.save(out,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()


def visual(lines=None):
    return dict(description='Artificial tabletop with two red cups',
        entities=[dict(name='Cup brand',type='product',attributes=['printed label'])],
        objects=[dict(name='cup',count=2,attributes=['red'])],
        events=[dict(subject='person',action='holds',object='cup')],
        relations=[dict(source='cup',predicate='on',target='table')],
        tags=['artificial','tabletop'],location_hint='inside',ocr_lines=['Label'] if lines is None else lines)


def localize(*,no_text=False):
    return dict(inspected_entire_image=True,assessment='no_text' if no_text else 'text_regions',
        regions=[] if no_text else [dict(region_id='label',bbox=[100,100,900,900],readability='readable',reason='artificial label')])


def crop(*,mode='readable',text='Literal label'):
    if mode=='no_text':return dict(inspected_entire_crop=True,assessment='no_text',lines=[],unreadable_regions=[])
    if mode in {'unreadable','incomplete'}:
        return dict(inspected_entire_crop=mode=='unreadable',assessment='unreadable_text',lines=[],
            unreadable_regions=[dict(bbox=[100,100,900,900],reason='blurred')])
    return dict(inspected_entire_crop=True,assessment='readable_text',
        lines=[dict(bbox=[100,100,900,200],text=text)],unreadable_regions=[])


def reply(payload, *,tokens=20,finish='stop'):
    return SimpleNamespace(model='gpt-4o-mini',choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(payload,ensure_ascii=False),refusal=None),finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10,completion_tokens=tokens,total_tokens=10+tokens))


class ArtificialSDK:
    def __init__(self,handler):
        self.handler=handler;self.calls=[]
        self.chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self,**kwargs):
        name=kwargs['response_format']['json_schema']['name']
        stage='scene' if name=='visual_memory_ocr_lines_v2' else 'locator' if name=='ocr_physical_text_localization_v1' else 'crop'
        self.calls.append(dict(stage=stage,options=deepcopy(kwargs)))
        return self.handler(stage,len(self.calls))


class MemoryCache:
    def __init__(self):self.records={};self.writes=[]
    def __len__(self):return len(self.records)
    def get(self,media_id,key):return deepcopy(self.records.get((media_id,key)))
    def put(self,media_id,key,extraction,**kwargs):
        assert validation_receipt_valid(kwargs.get('output_validation'))
        self.writes.append(dict(media_id=media_id,key=key,extraction=deepcopy(extraction),**deepcopy(kwargs)))
        self.records[media_id,key]=deepcopy(extraction)


class MemoryPreparationStore:
    def __init__(self):self.rows={};self.snapshots=[];self.fail_next_checkpoint=False


class MemoryRequestLedger:
    def __init__(self,store):self.store=store
    def reserve(self,scope,rid,body):
        key=scope.key,rid
        if key not in self.store.rows:self.store.rows[key]=dict(body=body,value={})
        assert self.store.rows[key]['body']==body
        return deepcopy(self.store.rows[key]['value'])
    def checkpoint(self,scope,rid,body,value):
        assert self.store.rows[scope.key,rid]['body']==body
        if self.store.fail_next_checkpoint:
            self.store.fail_next_checkpoint=False
            raise ValueError('artificial checkpoint persistence failure')
        assert 'data:image/' not in json.dumps(value)
        self.store.rows[scope.key,rid]['value']=deepcopy(value)
        self.store.snapshots.append(deepcopy(value))


@pytest.fixture
def pipeline(monkeypatch):
    import mandol_competition.multimodal.service as service
    monkeypatch.setattr(service,'RequestLedger',MemoryRequestLedger)
    def build(handler, *,schema_version='1'):
        sdk=ArtificialSDK(handler)
        client=OpenAICompatVisionClient(model='gpt-4o-mini',api_key='ARTIFICIAL-NOT-A-SECRET',
            base_url='https://artificial.invalid/v1',timeout=10,client_factory=lambda:sdk,
            output_tokens=128,max_output_tokens=512,wire_format=FORMAT_POLICY)
        config=MultimodalConfig(prompt_version=PROMPT_VERSION,schema_version=schema_version,
            vision_failure_policy='retry-required-v1',vlm_output_tokens=128,vlm_max_output_tokens=512)
        extractor=OpenAICompatExtractor(model_name='gpt-4o-mini',config=config,client=client)
        cache=MemoryCache()
        vision=CachingExtractor(extractor,cache,model='gpt-4o-mini',prompt_version=PROMPT_VERSION,
            schema_version=schema_version,provider_identity='artificial-provider',require_evidence=True)
        store=MemoryPreparationStore()
        backend=SimpleNamespace(state=store,scope=lambda user:Scope('artificial-scope',user))
        backend.publish=lambda *a,**kw:pytest.fail('preparation tests must not publish')
        request=SimpleNamespace(user_id='artificial-user',request_id='artificial-request')
        def ledger():return PreparationLedger(backend,request,'a'*64,strict=True)
        inputs=dict(image_data_uri=uri(),context_text='Artificial caption',image_position=0,image_count=1)
        return SimpleNamespace(sdk=sdk,client=client,extractor=extractor,vision=vision,cache=cache,
            store=store,backend=backend,request=request,ledger=ledger,inputs=inputs)
    return build


def extract(p,ledger=None,**changes):
    inputs={**p.inputs,**changes}
    return (ledger or p.ledger()).extract(key='0:1',vision=p.vision,**inputs)


def recovering_handler(stage,_):
    return reply(visual(['repeat']*64)) if stage=='scene' else reply(localize(),tokens=10) if stage=='locator' else reply(crop())


def test_normal_primary_preserves_all_original_fields_and_only_completed_cache(pipeline):
    p=pipeline(lambda stage,_:reply(visual()))
    result=extract(p)
    assert result.description==visual()['description'] and result.ocr_text=='Label'
    for key in ('entities','objects','events','relations','tags','location_hint'):
        assert asdict(result)[key]==visual()[key]
    assert [c['stage'] for c in p.sdk.calls]==['scene'] and len(p.cache.writes)==1
    assert p.ledger().value['vision']['0:1']['status']=='complete'
    assert extract(p)==result and len(p.sdk.calls)==1


def test_paid_ocr_failure_resumes_only_ocr_across_fresh_ledger_instances(pipeline):
    failed=False
    def handler(stage,n):
        nonlocal failed
        if stage=='crop' and not failed:
            failed=True;raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p=pipeline(handler)
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    saved=p.ledger().value['vision']['0:1']['stage_checkpoint']
    assert saved['scene'] is not None and saved['localization'] is not None and not p.cache.writes
    old_scene=deepcopy(saved['scene'])
    result=extract(p)
    assert result.ocr_text=='Literal label' and result.description==visual()['description']
    assert [c['stage'] for c in p.sdk.calls]==['scene','locator','crop','crop']
    assert p.ledger().value['vision']['0:1']['stage_checkpoint']['scene']==old_scene
    assert len(p.cache.writes)==1


def test_cumulative_budget_is_not_refreshed_by_multiple_sixty_second_windows(pipeline):
    def handler(stage,n):
        if stage=='crop':raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p=pipeline(handler);counts=[]
    for round_number in range(30):
        ledger=p.ledger();ledger._recovery_now=lambda:round_number*1000.
        try:
            result=extract(p,ledger)
        except VisionExtractionUnavailable:
            pass
        counts.append(len(p.sdk.calls))
    state=p.ledger().value['vision']['0:1']['stage_checkpoint']
    assert counts[-1]==counts[-3] and state['spent']<=16384 and state['paid_calls']<=20
    assert sum(c['stage']=='scene' for c in p.sdk.calls)==1 and not p.cache.writes
    assert result.description==visual()['description'] and result.ocr_text==''
    assert state['complete_receipt']['finish_reason']=='incomplete'
    assert p.ledger().value['vision']['0:1']['ocr_coverage']['status']=='incomplete'


def test_checkpoint_failure_before_first_dispatch_prevents_any_model_or_cache_call(pipeline):
    p=pipeline(lambda *_:pytest.fail('model must not dispatch'))
    p.store.fail_next_checkpoint=True
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    assert not p.sdk.calls and not p.cache.writes


def test_scene_checkpoint_persistence_failure_cannot_be_mistaken_for_crop_refinement(pipeline):
    p=pipeline(recovering_handler)
    original=MemoryRequestLedger.checkpoint
    def reject_scene(self,scope,rid,body,value):
        stage=value.get('vision',{}).get('0:1',{}).get('stage_checkpoint',{})
        if stage.get('scene') is not None:raise ValueError('artificial scene checkpoint write failure')
        return original(self,scope,rid,body,value)
    import mandol_competition.multimodal.service as service
    class RejectingLedger(MemoryRequestLedger):checkpoint=reject_scene
    service.RequestLedger=RejectingLedger
    # The second status-save can also fail; neither error may trigger a model.
    with pytest.raises((VisionExtractionUnavailable,ValueError)):extract(p)
    assert [c['stage'] for c in p.sdk.calls]==['scene'] and not p.cache.writes


def test_no_text_requires_an_independent_full_image_confirmation_before_success(pipeline):
    def handler(stage,n):
        if stage=='scene':return reply(visual(['repeat']*64))
        if stage=='locator':return reply(localize(no_text=True))
        return reply(crop(mode='no_text'))
    p=pipeline(handler);result=extract(p)
    assert [c['stage'] for c in p.sdk.calls]==['scene','locator','crop']
    assert result.ocr_text=='' and len(p.cache.writes)==1
    coverage=p.ledger().value['vision']['0:1']['stage_checkpoint']['coverage']
    assert coverage['status']=='no_text' and coverage['completed_crops']==1


@pytest.mark.parametrize('mode,success',[('unreadable',True),('incomplete',False)])
def test_unreadable_empty_ocr_requires_explicit_complete_readability_decision(pipeline,mode,success):
    p=pipeline(lambda stage,n:reply(crop(mode=mode)) if stage=='crop' else recovering_handler(stage,n))
    if success:
        assert extract(p).ocr_text=='' and len(p.cache.writes)==1
        assert p.ledger().value['vision']['0:1']['stage_checkpoint']['coverage']['status']=='unreadable'
    else:
        with pytest.raises(VisionExtractionUnavailable):extract(p)
        assert not p.cache.writes and p.ledger().value['vision']['0:1']['status']!='complete'


def test_real_distinct_physical_rows_with_identical_text_are_both_retained(pipeline):
    def handler(stage,n):
        if stage!='crop':return recovering_handler(stage,n)
        value=crop(text='SAME')
        value['lines'].append(dict(bbox=[100,500,900,600],text='SAME'))
        return reply(value)
    p=pipeline(handler)
    assert extract(p).ocr_text=='SAME\nSAME'
    assert len(p.cache.writes)==1


@pytest.mark.parametrize('field,value',[('context_text','changed caption'),('image_data_uri',uri('red')),
    ('image_position',1),('image_count',2)])
def test_staged_checkpoint_rejects_changed_image_context_or_position_before_dispatch(pipeline,field,value):
    def handler(stage,n):
        if stage=='crop':raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p=pipeline(handler)
    if field=='image_position':p.inputs['image_count']=2
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    before=len(p.sdk.calls)
    with pytest.raises(VisionExtractionUnavailable):extract(p,**{field:value})
    assert len(p.sdk.calls)==before and not p.cache.writes


def test_schema_version_binding_drift_rejects_resume_before_dispatch(pipeline):
    def handler(stage,n):
        if stage=='crop':raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p=pipeline(handler)
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    before=len(p.sdk.calls)
    p.extractor.config=replace(p.extractor.config,schema_version='changed-schema')
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    assert len(p.sdk.calls)==before and not p.cache.writes


def test_primary_wire_schema_binding_drift_rejects_resume_before_dispatch(pipeline,monkeypatch):
    def handler(stage,n):
        if stage=='crop':raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,n)
    p=pipeline(handler)
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    before=len(p.sdk.calls)
    import mandol_competition.multimodal.vlm.staged_ocr as staged
    original=staged.response_format
    def changed():
        value=deepcopy(original());value['json_schema']['schema']['description']='changed schema identity';return value
    monkeypatch.setattr(staged,'response_format',changed)
    with pytest.raises(VisionExtractionUnavailable):extract(p)
    assert len(p.sdk.calls)==before and not p.cache.writes


def test_final_payload_checkpoint_can_restore_success_without_repeating_primary(pipeline):
    p=pipeline(lambda *_:reply(visual()))
    result=extract(p)
    entry=p.ledger().value['vision']['0:1'];entry['status']='started';entry.pop('extraction')
    key=p.backend.scope(p.request.user_id).key,p.request.request_id
    p.store.rows[key]['value']['vision']['0:1']=entry
    p.cache.records.clear();p.cache.writes.clear()
    assert extract(p)==result and len(p.sdk.calls)==1 and len(p.cache.writes)==1


def test_five_capacity_length_refinements_finish_all_physical_children_without_content_failure(pipeline):
    lengths=0;accepted_ids=[]
    def handler(stage,n):
        nonlocal lengths
        if stage!='crop':return recovering_handler(stage,n)
        checkpoint=p.ledger().value['vision']['0:1']['stage_checkpoint']
        task=next(t for t in checkpoint['tasks'] if t['crop_id'] not in checkpoint['results'])
        source,owned=task['source_box'],task['ownership_box']
        if lengths<5 and owned[3]-owned[1]>=128:
            lengths+=1
            incomplete=reply(crop(),tokens=1024,finish='length')
            incomplete.choices[0].message.content='{"inspected_entire_crop":true,"assessment":"readable_text","lines":['
            return incomplete
        height=source[3]-source[1]
        top=(owned[1]-source[1])*1000/height;bottom=(owned[3]-source[1])*1000/height
        y1=max(2,math.ceil(top+(bottom-top)*.3));y2=min(998,math.floor(top+(bottom-top)*.7))
        value=crop(text='row-'+task['crop_id'])
        value['lines'][0]['bbox']=[100,y1,900,y2]
        accepted_ids.append(task['crop_id'])
        return reply(value)
    p=pipeline(handler);p.inputs['image_data_uri']=uri(size=(200,2048))
    result=extract(p)
    state=p.ledger().value['vision']['0:1']['stage_checkpoint']
    assert lengths==5 and state['content_failures']==0
    assert len(state['tasks'])==len(state['results'])==len(accepted_ids)
    assert set(result.ocr_text.splitlines())=={'row-'+name for name in accepted_ids}
    assert result.description==visual()['description'] and asdict(result)['objects']==visual()['objects']
    assert sum(c['stage']=='scene' for c in p.sdk.calls)==1 and len(p.cache.writes)==1
    assert sum(c['metadata'].get('finish_reason')=='length' for c in state['calls'])==5
    assert state['spent']>=5*1024 and state['spent']<=16384


def test_real_sse_cumulative_empty_lines_prefix_completes_no_text_pipeline(pipeline,monkeypatch):
    """Exercise HTTP/SSE -> cumulative observer -> crop validator, no SDK shortcut."""
    from mandol_competition.inference.http import HTTPTransport
    from mandol_competition.inference.vlm import HTTPVLMClient
    import mandol_competition.inference.budget as budget
    monkeypatch.setattr(budget,'from_environment',lambda:None)
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER','configured')
    p=pipeline(lambda *_:pytest.fail('legacy SDK must not run'));seen=[];closed=[]
    class Chunks(httpx.SyncByteStream):
        def __iter__(self):
            text=json.dumps(crop(mode='no_text'),separators=(',',':'))
            for character in text:
                chunk=dict(model='gpt-4o-mini',choices=[dict(index=0,delta={'content':character},finish_reason=None)])
                yield ('data: '+json.dumps(chunk)+'\n\n').encode()
            yield b'data: {"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b'data: {"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20,"total_tokens":30}}\n\n'
            yield b'data: [DONE]\n\n'
        def close(self):closed.append(True)
    def handle(request):
        data=json.loads(request.content);name=data['response_format']['json_schema']['name']
        stage='scene' if name=='visual_memory_ocr_lines_v2' else 'locator' if name=='ocr_physical_text_localization_v1' else 'crop'
        seen.append((stage,data.get('stream',False)))
        if data.get('stream'):return httpx.Response(200,stream=Chunks())
        value=visual(['repeat']*64) if stage=='scene' else localize(no_text=True)
        return httpx.Response(200,json=dict(model='gpt-4o-mini',choices=[dict(index=0,
            message=dict(content=json.dumps(value),refusal=None),finish_reason='stop')],
            usage=dict(prompt_tokens=10,completion_tokens=20,total_tokens=30)))
    wire=HTTPTransport('https://artificial.invalid/v1',transport=httpx.MockTransport(handle),max_attempts=1)
    p.client._http=HTTPVLMClient('https://artificial.invalid/v1','gpt-4o-mini',transport=wire)
    try:
        result=extract(p)
        assert result.ocr_text=='' and result.description==visual()['description']
        assert seen==[('scene',False),('locator',False),('crop',True)]
        assert closed==[True] and len(p.cache.writes)==1 and not p.sdk.calls
        assert wire.snapshot()['inflight']==wire.snapshot()['waiting']==0
        assert p.ledger().value['vision']['0:1']['stage_checkpoint']['coverage']['status']=='no_text'
    finally:wire.close()
