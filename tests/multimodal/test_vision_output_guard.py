"""Owned response fixtures: no provider calls or official evaluation content."""
from types import SimpleNamespace
from dataclasses import replace
import json
import httpx

import pytest

from mandol_competition.multimodal.vlm.base import ExtractionError
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatExtractor, VisionExtractionUnavailable
from mandol_competition.multimodal.config import MultimodalConfig, ConfigurationError, get_config
from mandol_competition.multimodal.vlm.output_guard import VisionOutputTruncated
from mandol_competition.multimodal.vlm.cache import CachingExtractor, ExtractionCache, _media_id
from mandol_competition.observability.diagnostics import _exception_metadata

IMAGE='data:image/png;base64,QUJD'
# An explicitly smaller budget exercises the optional bounded-expansion path.
CONFIG=MultimodalConfig(vision_failure_policy='retry-required-v1',
                        vlm_output_tokens=4096,vlm_max_output_tokens=8192)


def sdk_fixture(content, *, finish_reason='stop', completion_tokens=25, captured=None):
    def complete(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish_reason,
                                     message=SimpleNamespace(content=content, refusal=None))],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=completion_tokens,
                                  total_tokens=20+completion_tokens))
    return OpenAICompatVisionClient(model='gpt-4o-mini',api_key='fixture',
        base_url='https://fixture.invalid/v1',timeout=3,output_tokens=4096,max_output_tokens=8192,
        client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete))))


def test_length_cannot_be_accepted_even_with_parseable_json():
    client=sdk_fixture('{"description":"unfinished observation"}',
                       finish_reason='length',completion_tokens=16384)
    with pytest.raises(ExtractionError):
        client.complete(prompt='Return JSON.',image_data_uri='owned-fixture')


def test_truncated_outer_object_cannot_salvage_a_nested_description():
    content='{"events": [{"description": "partial"}], "ocr_text": "unfinished'
    with pytest.raises(ExtractionError):
        sdk_fixture(content).complete(prompt='Return JSON.',image_data_uri='owned-fixture')


def test_missing_finish_reason_is_not_a_verified_success():
    with pytest.raises(ExtractionError):
        sdk_fixture('{"description":"visible"}',finish_reason=None).complete(
            prompt='Return JSON.',image_data_uri='owned-fixture')


def test_custom_wire_budget_and_json_mode_are_explicit():
    captured={}
    sdk_fixture('{"description":"visible"}',captured=captured).complete(
        prompt='Return JSON.',image_data_uri='owned-fixture')
    assert captured['max_completion_tokens']==4096
    assert captured['response_format']=={'type':'json_object'}


@pytest.mark.parametrize('reason',[None,'content_filter','tool_calls','function_call','made-up',{},[]])
def test_unverified_endings_never_become_success(reason):
    with pytest.raises(ExtractionError):
        sdk_fixture('{"description":"visible"}',finish_reason=reason).complete(
            prompt='Return JSON.',image_data_uri=IMAGE)


def sequence_client(outcomes,calls):
    iterator=iter(outcomes)
    def complete(**kwargs):
        calls.append(kwargs)
        value=next(iterator)
        if isinstance(value,Exception):
            raise value
        reason,content,tokens=value
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,
            message=SimpleNamespace(content=content,refusal=None))],
            usage=SimpleNamespace(prompt_tokens=30,completion_tokens=tokens,total_tokens=30+tokens))
    return OpenAICompatVisionClient(model='gpt-4o-mini',api_key='fixture',
        base_url='https://fixture.invalid/v1',timeout=3,output_tokens=4096,max_output_tokens=8192,
        client_factory=lambda:SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=complete))))


def extract(client,config=CONFIG):
    return OpenAICompatExtractor(model_name='gpt-4o-mini',config=config,client=client).extract(
        image_data_uri=IMAGE,context_text='owned source caption',image_position=0,image_count=1)


def test_one_expansion_preserves_dense_multilingual_ocr_without_slicing(monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    ocr='\n'.join(f'{n:04d} 人工票据 café Ω قيمة 商品价格 123.45' for n in range(400))
    calls=[]
    client=sequence_client([('length','{"ocr_text":"unfinished',4096),
        ('stop',json.dumps({'description':'Owned receipt','ocr_text':ocr}),6000)],calls)
    result=extract(client)
    assert result.ocr_text==ocr
    assert [x['max_completion_tokens'] for x in calls]==[4096,8192]
    assert calls[0]['messages']==calls[1]['messages']  # Full image and OCR rules retained.


def test_repeated_truncation_stops_after_two_paid_attempts(monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    calls=[]
    client=sequence_client([('length','{"description":"partial"}',4096),
                           ('length','{"description":"partial"}',8192)],calls)
    with pytest.raises(VisionExtractionUnavailable) as caught:
        extract(client)
    assert len(calls)==2 and not caught.value.retry_safe and not caught.value.unknown_outcome
    assert caught.value.error_class=='vision_output_truncated'


@pytest.mark.parametrize('policy',['retry-required-v1','degrade'])
@pytest.mark.parametrize('status',[400,401])
def test_later_permanent_rejection_neither_refunds_paid_output_nor_degrades(policy,status,monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    error=RuntimeError('owned protocol rejection')
    error.status_code=status
    calls=[]
    client=sequence_client([('length','{}',4096),error],calls)
    with pytest.raises(VisionExtractionUnavailable) as caught:
        extract(client,replace(CONFIG,vision_failure_policy=policy))
    assert len(calls)==2 and not caught.value.retry_safe and not caught.value.unknown_outcome


def test_output_at_reported_budget_is_rejected_even_when_provider_says_stop():
    with pytest.raises(VisionOutputTruncated):
        sdk_fixture('{"description":"partial"}',completion_tokens=4096).complete(
            prompt='Return JSON.',image_data_uri=IMAGE)


@pytest.mark.parametrize('content',[
    '{"description":"first","description":"second"}',
    '{"description":"visible","objects":[{"count":NaN}]}',
    '{"description":"visible"} trailing partial output',
    '{"entities":[{"description":"nested"}],"ocr_text":"unfinished',
])
def test_repeated_nonfinite_or_incomplete_json_is_rejected(content):
    with pytest.raises(ExtractionError):
        sdk_fixture(content).complete(prompt='Return JSON.',image_data_uri=IMAGE)


@pytest.mark.parametrize('initial,maximum',[(0,8192),(4096,2048),(True,8192),(4096,16385),(4096,float('inf'))])
def test_invalid_output_budget_configuration_is_rejected(initial,maximum):
    with pytest.raises(ConfigurationError):
        replace(CONFIG,vlm_output_tokens=initial,vlm_max_output_tokens=maximum)


def test_output_configuration_can_be_explicitly_overridden(monkeypatch):
    monkeypatch.setenv('MANDOL_MM_VLM_OUTPUT_TOKENS','5000')
    monkeypatch.setenv('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS','9000')
    get_config.cache_clear()
    try:
        cfg=get_config()
        assert (cfg.vlm_output_tokens,cfg.vlm_max_output_tokens)==(5000,9000)
    finally:
        get_config.cache_clear()


def test_new_cache_records_include_input_bound_output_validation(tmp_path):
    calls=[]
    client=sequence_client([('stop','{"description":"Owned blue marker"}',30)],calls)
    inner=OpenAICompatExtractor(model_name='gpt-4o-mini',config=CONFIG,client=client)
    cache=ExtractionCache(tmp_path)
    wrapper=CachingExtractor(inner,cache,model='gpt-4o-mini',
        prompt_version=CONFIG.prompt_version,schema_version=CONFIG.schema_version,require_evidence=True)
    inputs=dict(image_data_uri=IMAGE,context_text='owned caption',image_position=0,image_count=1)
    first=wrapper.extract(**inputs)
    assert wrapper.extract(**inputs)==first and len(calls)==1
    key=wrapper._key(_media_id(IMAGE),'owned caption',0,1)
    record=cache._read_record(_media_id(IMAGE),key)
    assert record['output_validation']['finish_reason']=='stop'
    assert record['output_validation']['media_id']==_media_id(IMAGE)
    assert record['output_validation']['completion_tokens']==30
    assert len(record['output_validation']['extraction_sha256'])==64


def test_truncated_response_does_not_create_a_cache_record(tmp_path,monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    client=sequence_client([('length','{}',4096),('length','{}',8192)],[])
    inner=OpenAICompatExtractor(model_name='gpt-4o-mini',config=CONFIG,client=client)
    cache=ExtractionCache(tmp_path)
    wrapper=CachingExtractor(inner,cache,model='gpt-4o-mini',prompt_version=CONFIG.prompt_version,
                             schema_version=CONFIG.schema_version,require_evidence=True)
    with pytest.raises(VisionExtractionUnavailable):
        wrapper.extract(image_data_uri=IMAGE,context_text='',image_position=0,image_count=1)
    assert len(cache)==0


def test_output_diagnostics_are_content_free_and_bounded():
    error=VisionOutputTruncated(finish_reason='length',completion_tokens=4096,output_token_budget=4096)
    safe=_exception_metadata(error)
    assert safe['inference_error_class']=='vision_output_truncated'
    assert safe['vlm_finish_reason']=='length' and safe['vlm_completion_tokens']==4096
    assert 'content' not in safe and 'message' not in safe


def http_client(handler):
    client=OpenAICompatVisionClient(model='gpt-4o-mini',api_key='owned-fixture',
                                   base_url='https://fixture.invalid/v1',timeout=3,
                                   output_tokens=4096,max_output_tokens=8192)
    client._http.http.client.close()
    client._http.http.client=httpx.Client(base_url='https://fixture.invalid/v1',
                                        transport=httpx.MockTransport(handler))
    return client


def test_real_http_adapter_preserves_output_budget_during_expansion(monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    captured=[]
    def handler(request):
        payload=json.loads(request.content)
        captured.append(payload)
        return httpx.Response(200,json={'model':'gpt-4o-mini','choices':[{
            'finish_reason':'length' if len(captured)==1 else 'stop',
            'message':{'content':'{"description":"Owned label","ocr_text":"ABC 001"}'}}],
            'usage':{'completion_tokens':4096 if len(captured)==1 else 35}})
    client=http_client(handler)
    try:
        assert extract(client).ocr_text=='ABC 001'
    finally:
        client.close()
    assert [r['max_completion_tokens'] for r in captured]==[4096,8192]
    assert all(r['response_format']=={'type':'json_object'} for r in captured)


def test_gateway_parameter_rejection_never_falls_back_to_unbounded_generation():
    captured=[]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(400,json={'error':{'code':'invalid_parameter','message':'owned fixture'}})
    client=http_client(handler)
    try:
        with pytest.raises(VisionExtractionUnavailable):
            extract(client)
    finally:
        client.close()
    assert len(captured)==1 and captured[0]['max_completion_tokens']==4096


def test_parallel_output_receipts_stay_with_their_own_image():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier=Barrier(2)
    calls=[]
    client=sequence_client([('stop','{"description":"Owned observation"}',25)]*2,calls)
    inner=OpenAICompatExtractor(model_name='gpt-4o-mini',config=CONFIG,client=client)
    def run(uri):
        inner.extract(image_data_uri=uri,context_text='caption',image_position=0,image_count=1)
        barrier.wait(timeout=5)
        receipts=inner.take_output_receipts()
        assert inner.take_output_receipts()==[]
        return receipts[0]['media_id']
    uris=[IMAGE,'data:image/png;base64,REVG']
    with ThreadPoolExecutor(max_workers=2) as workers:
        assert list(workers.map(run,uris))==[_media_id(uri) for uri in uris]


def test_truncation_never_publishes_or_resets_the_durable_recovery_window(backend,monkeypatch):
    from .backend_support import request
    from .test_sqlite_qdrant import image_part
    from .test_vision_checkpoint_recovery import ledger,published
    from mandol_competition.multimodal.service import AddService,PreparationLedger
    from mandol_competition.inference.errors import InferenceRateLimited
    monkeypatch.setattr(PreparationLedger,'_recovery_now',lambda _:1000.)
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    calls=[]
    # First failure plus at most three recovery sequences, two paid calls each.
    client=sequence_client([('length','{}',n) for _ in range(4) for n in (4096,8192)],calls)
    config=replace(backend.config,vision_failure_policy='retry-required-v1',
                   vlm_output_tokens=4096,vlm_max_output_tokens=8192)
    vision=OpenAICompatExtractor(model_name='gpt-4o-mini',config=config,client=client)
    add=AddService(config,extractor=vision,backend=backend)
    body=request(content=[image_part()])
    for _ in range(4):
        with pytest.raises(VisionExtractionUnavailable):
            add.handle(body)
        assert not published(backend)
    saved=ledger(backend,body).value['vision']['0:0']
    assert saved['status']=='failed' and saved['recovery_attempts']==3
    with pytest.raises(InferenceRateLimited):
        add.handle(body)
    assert len(calls)==8 and not published(backend)


@pytest.mark.parametrize('later',['rate_limit','deadline','deadline_before_expansion'])
def test_later_rejection_does_not_refund_a_paid_truncated_attempt(backend,monkeypatch,later):
    from .backend_support import request
    from .test_sqlite_qdrant import image_part
    from .test_vision_checkpoint_recovery import ledger,published
    from mandol_competition.multimodal.service import AddService,PreparationLedger
    from mandol_competition.inference.errors import InferenceRateLimited,InferenceDeadlineExceeded
    from mandol_competition.inference.http import deadline_budget
    monkeypatch.setattr(PreparationLedger,'_recovery_now',lambda _:1000.)
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    calls=[]
    error=InferenceRateLimited(retry_after=60) if later=='rate_limit' else InferenceDeadlineExceeded()
    client=sequence_client([('length','{}',4096),error],calls)
    config=replace(backend.config,vision_failure_policy='retry-required-v1',vlm_max_attempts=2,
                   vlm_output_tokens=4096,vlm_max_output_tokens=8192)
    vision=OpenAICompatExtractor(model_name='gpt-4o-mini',config=config,client=client)
    add=AddService(config,extractor=vision,backend=backend)
    body=request(content=[image_part()])
    expected=InferenceRateLimited if later=='rate_limit' else InferenceDeadlineExceeded
    with deadline_budget(.5 if later=='deadline_before_expansion' else 10):
        with pytest.raises(expected) as caught:
            add.handle(body)
    assert caught.value.vision_paid_output_failure is True
    assert ledger(backend,body).value['vision']['0:0']['status']=='failed'
    assert not published(backend)
    assert len(calls)==(1 if later=='deadline_before_expansion' else 2)
