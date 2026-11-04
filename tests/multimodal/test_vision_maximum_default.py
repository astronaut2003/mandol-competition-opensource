"""MM's 1024-token per-call ceiling never authorizes truncated facts."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from mandol_competition.multimodal.config import MultimodalConfig, get_config
from mandol_competition.multimodal.vlm.openai_compat import (
    OpenAICompatVisionClient, OpenAICompatExtractor, VisionExtractionUnavailable)
from mandol_competition.multimodal.vlm.cache import CachingExtractor, ExtractionCache

IMAGE = 'data:image/png;base64,QUJD'


def default_client(outcomes, calls):
    sequence = iter(outcomes)
    def complete(**request):
        calls.append(request)
        reason, content, tokens = next(sequence)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason,
            message=SimpleNamespace(content=content, refusal=None))], usage=SimpleNamespace(
            prompt_tokens=30, completion_tokens=tokens, total_tokens=30 + tokens))
    return OpenAICompatVisionClient(model='gpt-4o-mini', api_key='synthetic-only',
        base_url='https://fixture.invalid/v1', timeout=3,
        client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete))))


def extract(client, config=None):
    cfg = config or MultimodalConfig(vision_failure_policy='retry-required-v1')
    return OpenAICompatExtractor(model_name='gpt-4o-mini', config=cfg, client=client).extract(
        image_data_uri=IMAGE, context_text='owned caption', image_position=0, image_count=1)


def test_dataclass_and_environment_defaults_use_the_mm_per_call_cap(monkeypatch):
    for key in ('MANDOL_MM_VLM_OUTPUT_TOKENS', 'MANDOL_MM_VLM_MAX_OUTPUT_TOKENS'):
        monkeypatch.delenv(key, raising=False)
    get_config.cache_clear()
    try:
        for cfg in (MultimodalConfig(), get_config()):
            assert cfg.vlm_output_tokens == cfg.vlm_max_output_tokens == 1024
    finally:
        get_config.cache_clear()


def test_capped_default_preserves_complete_short_multilingual_ocr_without_trimming():
    calls = []
    ocr = '\n'.join(f'{i:04d} Owned multilingual receipt 商品 цена café' for i in range(12))
    client = default_client([('stop', json.dumps({'description': 'Owned receipt', 'ocr_text': ocr}), 600)], calls)
    assert extract(client).ocr_text == ocr
    assert len(calls) == 1 and calls[0]['max_completion_tokens'] == 1024
    assert calls[0]['response_format'] == {'type': 'json_object'}


@pytest.mark.parametrize('reason', ['length', 'stop'])
def test_hitting_per_call_cap_never_repeats_or_caches_partial_output(tmp_path, reason):
    calls = []
    client = default_client([(reason, '{"description":"partial but parseable"}', 1024)], calls)
    cfg = MultimodalConfig(vision_failure_policy='retry-required-v1')
    inner = OpenAICompatExtractor(model_name=cfg.vlm_model, config=cfg, client=client)
    cache = ExtractionCache(tmp_path)
    wrapped = CachingExtractor(inner, cache, model=cfg.vlm_model,
        prompt_version=cfg.prompt_version, schema_version=cfg.schema_version, require_evidence=True)
    with pytest.raises(VisionExtractionUnavailable) as caught:
        wrapped.extract(image_data_uri=IMAGE, context_text='owned', image_position=0, image_count=1)
    assert caught.value.error_class == 'vision_output_truncated'
    assert not caught.value.retry_safe
    assert len(calls) == 1 and len(cache) == 0


def test_explicit_smaller_override_expands_only_512_to_1024_with_original_message(monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep', lambda _: None)
    calls = []
    client = default_client([('length', '{}', 512), ('stop', '{"description":"Owned full output"}', 700)], calls)
    cfg = MultimodalConfig(vision_failure_policy='retry-required-v1',
                          vlm_output_tokens=512, vlm_max_output_tokens=1024)
    assert extract(client, cfg).description == 'Owned full output'
    assert [r['max_completion_tokens'] for r in calls] == [512, 1024]
    assert calls[0]['messages'] == calls[1]['messages']


def test_long_document_at_cap_cannot_be_misrepresented_as_complete_or_cached(tmp_path):
    calls=[]
    ocr='\n'.join(f'{i:04d} Owned long receipt 商品 цена café' for i in range(450))
    client=default_client([('length',json.dumps({'description':'Owned long receipt','ocr_text':ocr}),1024)],calls)
    cfg=MultimodalConfig(vision_failure_policy='retry-required-v1')
    inner=OpenAICompatExtractor(model_name=cfg.vlm_model,config=cfg,client=client)
    cache=ExtractionCache(tmp_path)
    wrapped=CachingExtractor(inner,cache,model=cfg.vlm_model,prompt_version=cfg.prompt_version,
        schema_version=cfg.schema_version,require_evidence=True)
    with pytest.raises(VisionExtractionUnavailable) as caught:
        wrapped.extract(image_data_uri=IMAGE,context_text='owned',image_position=0,image_count=1)
    assert caught.value.error_class=='vision_output_truncated' and not caught.value.retry_safe
    assert len(calls)==1 and calls[0]['max_completion_tokens']==1024 and len(cache)==0


def test_second_truncation_at_cap_stops_after_one_expansion_without_cache(tmp_path,monkeypatch):
    monkeypatch.setattr('mandol_competition.multimodal.vlm.openai_compat.time.sleep',lambda _:None)
    calls=[];client=default_client([('length','{}',512),('length','{}',1024)],calls)
    cfg=MultimodalConfig(vision_failure_policy='retry-required-v1',vlm_output_tokens=512,vlm_max_output_tokens=1024)
    inner=OpenAICompatExtractor(model_name=cfg.vlm_model,config=cfg,client=client)
    cache=ExtractionCache(tmp_path)
    wrapped=CachingExtractor(inner,cache,model=cfg.vlm_model,prompt_version=cfg.prompt_version,
        schema_version=cfg.schema_version,require_evidence=True)
    with pytest.raises(VisionExtractionUnavailable) as caught:
        wrapped.extract(image_data_uri=IMAGE,context_text='owned',image_position=0,image_count=1)
    assert caught.value.error_class=='vision_output_truncated' and not caught.value.retry_safe
    assert [r['max_completion_tokens'] for r in calls]==[512,1024] and len(cache)==0
    assert calls[0]['messages']==calls[1]['messages']


def test_capped_default_durable_attempts_remain_bounded_and_unpublished(backend, monkeypatch):
    from .backend_support import request
    from .test_sqlite_qdrant import image_part
    from .test_vision_checkpoint_recovery import ledger, published
    from mandol_competition.multimodal.service import AddService, PreparationLedger
    from mandol_competition.inference.errors import InferenceRateLimited
    monkeypatch.setattr(PreparationLedger, '_recovery_now', lambda _: 1000.)
    calls = []
    client = default_client([('length', '{}', 1024)] * 4, calls)
    cfg = replace(backend.config, vision_failure_policy='retry-required-v1')
    assert cfg.vlm_output_tokens == cfg.vlm_max_output_tokens == 1024
    add = AddService(cfg, extractor=OpenAICompatExtractor(
        model_name='gpt-4o-mini', config=cfg, client=client), backend=backend)
    body = request(content=[image_part()])
    for _ in range(4):
        with pytest.raises(VisionExtractionUnavailable):
            add.handle(body)
        assert not published(backend)
    saved = ledger(backend, body).value['vision']['0:0']
    assert saved['status'] == 'failed' and saved['recovery_attempts'] == 3
    with pytest.raises(InferenceRateLimited):
        add.handle(body)
    assert len(calls) == 4
