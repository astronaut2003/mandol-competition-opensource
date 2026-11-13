"""Opt-in original evidence with real owned persistence, synthetic model replies."""
from dataclasses import replace
import json
import math

import pytest
from mandol_substrate.state.models import StateUnavailable
from mandol_competition.multimodal.config import MultimodalConfig,ConfigurationError,get_config
from mandol_competition.multimodal.protocol import AddRequest,SearchRequest
from mandol_competition.multimodal.service import AddService,SearchService
from mandol_competition.multimodal.vlm.openai_compat import (
    OpenAICompatExtractor,OpenAICompatVisionClient,VisionExtractionUnavailable)
from mandol_competition.multimodal.vlm.ocr_lines import FORMAT_POLICY,PROMPT_VERSION
from mandol_competition.inference.errors import InferenceIdentityError
from .test_staged_ocr_pipeline import ArtificialSDK,reply,uri


def setup(backend,enabled,*,partial=False,identity_failure=False):
    config=replace(backend.config,vision_failure_policy='retry-required-v1',vision_source_fallback=enabled,
        prompt_version=PROMPT_VERSION,vlm_output_tokens=1024,vlm_max_output_tokens=1024,vlm_model='gpt-4o-mini')
    backend.config=config
    def handler(_stage,_number):
        if identity_failure:raise InferenceIdentityError('Artificial identity failure')
        name=sdk.calls[-1]['options']['response_format']['json_schema']['name']
        if partial and name=='scene_fields_v1_overview':
            return reply(dict(description='An artificial red cup on a table.',tags=['red cup'],location_hint=None))
        result=reply({},tokens=1024,finish='length')
        result.choices[0].message.content='{"description":"unfinished'
        return result
    sdk=ArtificialSDK(handler)
    client=OpenAICompatVisionClient(model='gpt-4o-mini',api_key='ARTIFICIAL-NOT-A-KEY',
        base_url='https://artificial.invalid/v1',timeout=10,client_factory=lambda:sdk,
        output_tokens=1024,max_output_tokens=1024,wire_format=FORMAT_POLICY)
    vision=OpenAICompatExtractor(model_name='gpt-4o-mini',config=config,client=client)
    return config,sdk,vision,AddService(config,extractor=vision,backend=backend)


def body(user='alice',rid='owned-source-only'):
    return AddRequest(user_id=user,request_id=rid,session_id='owned-session',messages=[dict(role='user',
        content=[dict(type='text',text='Original café 红杯 caption'),dict(type='image_url',image_url=dict(url=uri()))])])


def query(user='alice'):
    return SearchRequest(user_id=user,query='Original café 红杯 caption',top_k=100)


def test_default_disabled_exhaustion_still_errors_and_keeps_originals_without_publishing(backend):
    config,sdk,vision,add=setup(backend,False)
    with pytest.raises(VisionExtractionUnavailable):add.handle(body())
    assert len(sdk.calls)==4
    for _ in range(4):
        with pytest.raises(VisionExtractionUnavailable):add.handle(body())
    assert len(sdk.calls)==4 and backend.search('alice','caption',top_k=10)==[]
    originals=list((config.user_root('alice')/'raw-intake').glob('*.json'))
    assert len(originals)==1 and len(list((config.user_root('alice')/'media').rglob('*.bin')))==1
    assert not list((config.user_root('alice')/'vlm_cache').rglob('*.json'))


@pytest.mark.parametrize('partial',[False,True])
def test_enabled_terminal_source_is_searchable_with_original_pixels_and_honest_status(backend,partial):
    config,sdk,vision,add=setup(backend,True,partial=partial)
    request=body();response=add.handle(request);count=len(sdk.calls)
    assert response['success'] and count==(5 if partial else 4)
    assert add.handle(request)==response and len(sdk.calls)==count
    search=SearchService(config,extractor=vision,backend=backend)
    result=search.handle(query())
    assert len(result['data'])==1 and math.isfinite(result['data'][0]['score'])
    content=result['data'][0]['content']
    assert next(p['image_url']['url'] for p in content if p['type']=='image_url')==uri()
    assert any('Original café 红杯 caption' in p['text'] for p in content if p['type']=='text')
    note=next(p['text'] for p in content if p['type']=='text' and p['text'].startswith('[Visual extraction status]'))
    status=json.loads(note.partition('\n')[2])
    assert status['extraction_complete'] is False and status['original_images_retained'] is True
    unit=backend.search('alice','caption',top_k=10)[0][0]
    image=unit.raw_data['multimodal_info']['images'][0]
    assert image['description']==('An artificial red cup on a table.' if partial else '')
    assert image['objects']==[] and image['ocr_text']==''
    assert unit.metadata['visual_extraction_status']=='source_only'
    assert search.handle(query('bob'))=={'data':[]}
    assert not list((config.user_root('alice')/'vlm_cache').rglob('*.json'))


def test_partial_preparation_survives_publish_failure_and_retry_without_more_model_calls(backend,monkeypatch):
    config,sdk,vision,add=setup(backend,True)
    original=backend.publish;first=[True]
    def failed_once(*args,**kwargs):
        if first[0]:first[0]=False;raise StateUnavailable('Artificial projection outage')
        return original(*args,**kwargs)
    monkeypatch.setattr(backend,'publish',failed_once)
    with pytest.raises(StateUnavailable):add.handle(body())
    count=len(sdk.calls)
    assert backend.search('alice','caption',top_k=10)==[]
    assert add.handle(body())['success'] and len(sdk.calls)==count
    assert len(backend.search('alice','caption',top_k=10))==1


def test_disabled_switch_cannot_publish_a_prepared_source_only_unit(backend,monkeypatch):
    config,sdk,vision,add=setup(backend,True)
    original=backend.publish
    monkeypatch.setattr(backend,'publish',lambda *a,**kw:(_ for _ in ()).throw(StateUnavailable('Artificial outage')))
    with pytest.raises(StateUnavailable):add.handle(body())
    count=len(sdk.calls);monkeypatch.setattr(backend,'publish',original)
    strict=replace(config,vision_source_fallback=False)
    with pytest.raises(VisionExtractionUnavailable):
        AddService(strict,extractor=vision,backend=backend).handle(body())
    assert len(sdk.calls)==count and backend.search('alice','caption',top_k=10)==[]


def test_identity_failure_does_not_authorize_source_only_success(backend):
    config,sdk,vision,add=setup(backend,True,identity_failure=True)
    with pytest.raises(VisionExtractionUnavailable):add.handle(body())
    assert len(sdk.calls)==1 and backend.search('alice','caption',top_k=10)==[]
    for _ in range(6):
        with pytest.raises(VisionExtractionUnavailable):add.handle(body())
    assert backend.search('alice','caption',top_k=10)==[]


def test_archive_write_failure_stops_before_any_model_or_source_publication(backend,monkeypatch):
    config,sdk,vision,add=setup(backend,True)
    from mandol_competition.multimodal.raw_intake import RawIntakeStore,RawIntakeError
    monkeypatch.setattr(RawIntakeStore,'save',lambda *a,**kw:(_ for _ in ()).throw(RawIntakeError('Artificial fsync failure')))
    with pytest.raises(RawIntakeError):add.handle(body())
    assert not sdk.calls and backend.search('alice','caption',top_k=10)==[]


def test_source_fallback_does_not_apply_to_an_image_search_query(backend):
    config,sdk,vision,add=setup(backend,True)
    search=SearchService(config,extractor=vision,backend=backend)
    with pytest.raises(VisionExtractionUnavailable):
        search.prepare_query(SearchRequest(user_id='alice',query=[dict(type='image_url',image_url=dict(url=uri()))],top_k=10))


@pytest.mark.parametrize('saved_before_crash',[False,True])
def test_source_decision_checkpoint_failure_and_restart_never_lose_originals_or_redo_models(backend,monkeypatch,saved_before_crash):
    config,sdk,vision,add=setup(backend,True)
    from mandol_substrate.state.ledger import RequestLedger
    original=RequestLedger.checkpoint;failed=[False]
    def crash(self,scope,rid,body_hash,value):
        if value.get('source_only') and not failed[0]:
            failed[0]=True
            if saved_before_crash:original(self,scope,rid,body_hash,value)
            raise StateUnavailable('Artificial crash at source decision persistence')
        return original(self,scope,rid,body_hash,value)
    monkeypatch.setattr(RequestLedger,'checkpoint',crash)
    with pytest.raises(StateUnavailable):add.handle(body())
    count=len(sdk.calls)
    assert backend.search('alice','caption',top_k=10)==[]
    # A fresh preparation service reads the actual SQLite checkpoint.
    assert AddService(config,extractor=vision,backend=backend).handle(body())['success']
    assert len(sdk.calls)==count and len(backend.search('alice','caption',top_k=10))==1


def test_durable_source_decision_restarts_without_initializing_any_model_client(backend,monkeypatch):
    config,sdk,vision,add=setup(backend,True)
    from mandol_substrate.state.ledger import RequestLedger
    original=RequestLedger.checkpoint;failed=[False]
    def crash_after_source(self,scope,rid,body_hash,value):
        result=original(self,scope,rid,body_hash,value)
        if value.get('source_only') and not failed[0]:
            failed[0]=True;raise StateUnavailable('Artificial crash after durable source decision')
        return result
    monkeypatch.setattr(RequestLedger,'checkpoint',crash_after_source)
    with pytest.raises(StateUnavailable):add.handle(body())
    import mandol_competition.multimodal.vlm.openai_compat as wire
    monkeypatch.setattr(wire,'build_vision_client',lambda *_:pytest.fail('No new client or credentials may be needed'))
    fresh=OpenAICompatExtractor(model_name='gpt-4o-mini',config=config)
    assert AddService(config,extractor=fresh,backend=backend).handle(body())['success']
    assert fresh._client is None


def test_source_marker_cannot_validate_changed_visual_facts(backend):
    from copy import deepcopy
    from mandol_competition.multimodal.source_fallback import delivery_note
    config,sdk,vision,add=setup(backend,True)
    add.handle(body())
    unit=deepcopy(backend.search('alice','caption',top_k=10)[0][0])
    unit.raw_data['multimodal_info']['images'][0]['description']='Unverified invented fact'
    with pytest.raises(StateUnavailable):delivery_note(unit)


def test_verified_scene_with_incomplete_ocr_can_retry_publication_without_new_models(backend,monkeypatch):
    from .test_staged_ocr_pipeline import recovering_handler
    from mandol_competition.inference.errors import InferenceUnavailable
    config,sdk,vision,add=setup(backend,False)
    def handler(stage,number):
        if stage=='crop':raise InferenceUnavailable('ReadError',unknown_outcome=True)
        return recovering_handler(stage,number)
    sdk.handler=handler
    original=backend.publish;failed=[False]
    def publish(*args,**kwargs):
        if not failed[0]:
            failed[0]=True;raise StateUnavailable('Artificial outage after verified preparation')
        return original(*args,**kwargs)
    monkeypatch.setattr(backend,'publish',publish)
    for _ in range(30):
        try:add.handle(body())
        except VisionExtractionUnavailable:continue
        except StateUnavailable:break
        else:pytest.fail('The first publication must fail')
    else:pytest.fail('A verified scene never reached publication')
    count=len(sdk.calls)
    assert backend.search('alice','caption',top_k=10)==[]
    assert AddService(config,extractor=vision,backend=backend).handle(body())['success']
    assert len(sdk.calls)==count
    unit=backend.search('alice','caption',top_k=10)[0][0]
    assert unit.metadata['visual_ocr_readability'][0]['status']=='incomplete'
    assert 'visual_source_only' not in unit.metadata
    assert unit.raw_data['multimodal_info']['images'][0]['description']


@pytest.mark.parametrize('enabled',[False,True])
def test_default_http_503_and_explicit_opt_in_http_success(backend,enabled):
    from fastapi.testclient import TestClient
    from mandol_competition.multimodal.api import create_app
    config,sdk,vision,add=setup(backend,enabled)
    app=create_app(config,extractor=vision,backend=backend)
    try:
        with TestClient(app) as client:
            response=client.post('/add',json=body().model_dump())
            assert response.status_code==(200 if enabled else 503)
            found=client.post('/search',json=query().model_dump()).json()['data']
            assert len(found)==int(enabled)
    finally:
        # The app owns and closes this already-started fixture backend.
        backend.close()


def test_explicit_switch_parsing_and_strict_runtime_default_on(monkeypatch):
    assert MultimodalConfig().vision_source_fallback is False
    with pytest.raises(ConfigurationError):MultimodalConfig(vision_source_fallback=1)
    with pytest.raises(ConfigurationError):MultimodalConfig(vision_source_fallback=True)
    monkeypatch.setenv('MANDOL_MM_VISION_FAILURE_POLICY','retry-required-v1')
    monkeypatch.delenv('MANDOL_MM_VISION_SOURCE_FALLBACK',raising=False);get_config.cache_clear()
    assert get_config().vision_source_fallback is True
    for value,expected in [('0',False),('1',True)]:
        monkeypatch.setenv('MANDOL_MM_VISION_SOURCE_FALLBACK',value);get_config.cache_clear()
        assert get_config().vision_source_fallback is expected
    monkeypatch.setenv('MANDOL_MM_VISION_SOURCE_FALLBACK','yes');get_config.cache_clear()
    with pytest.raises(ConfigurationError):get_config()
    get_config.cache_clear()


def test_unknown_transport_exhaustion_cannot_be_published_as_source_only(backend):
    from mandol_competition.inference.errors import InferenceUnavailable
    config,sdk,vision,add=setup(backend,True)
    sdk.handler=lambda *_:(_ for _ in ()).throw(InferenceUnavailable('ReadError',unknown_outcome=True))
    for _ in range(25):
        with pytest.raises(VisionExtractionUnavailable):add.handle(body())
    assert backend.search('alice','caption',top_k=10)==[]


def test_quality_audit_is_durable_idempotent_and_excludes_unpublished_units(backend,monkeypatch):
    from mandol_competition.multimodal.add_quality import audit
    config,sdk,vision,add=setup(backend,True)
    original=backend.publish;failed=[False]
    def publish(*args,**kwargs):
        if not failed[0]:failed[0]=True;raise StateUnavailable('Artificial publication outage')
        return original(*args,**kwargs)
    monkeypatch.setattr(backend,'publish',publish)
    with pytest.raises(StateUnavailable):add.handle(body())
    database=config.data_dir/'state.sqlite3'
    assert audit(database)['published_images']==0
    assert add.handle(body())['success']
    for _ in range(5):assert AddService(config,extractor=vision,backend=backend).handle(body())['success']
    summary=audit(database)
    assert summary['published_images']==1 and summary['counts']['source_only']==1
    assert summary['source_only_rate']==summary['degradation_rate']==1
    assert summary['read_only'] and summary['retry_attempts_counted'] is False


def test_source_only_without_any_caption_preserves_image_and_counts_its_limitations(backend):
    from mandol_competition.multimodal.add_quality import audit
    config,sdk,vision,add=setup(backend,True)
    request=AddRequest(user_id='alice',request_id='owned-pure-image',session_id='owned-session',
        messages=[dict(role='user',content=[dict(type='image_url',image_url=dict(url=uri()))])])
    assert add.handle(request)['success']
    summary=audit(config.data_dir/'state.sqlite3')
    assert summary['published_images']==summary['counts']['source_only']==1
    assert summary['source_only_rate']==1  # No claim that image-only semantic recall is complete.
    assert len(list((config.user_root('alice')/'media').rglob('*.bin')))==1
