"""New OCR transport preserves atomic facts and cannot publish blank-line loops."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.vlm.ocr_lines import (FORMAT_POLICY, PROMPT_VERSION,
    canonical_payload, response_format)
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError, VisionOutputTruncated
from mandol_competition.multimodal.vlm.prompt import build_extraction_prompt


def wire_payload():
    return dict(description='A person in a black cardigan holds a lit white candle at a white table.',
        entities=[dict(name='person',type='person',attributes=['black cardigan'])],
        objects=[dict(name='candle',count=1,attributes=['lit','white container']),
            dict(name='notebook',count=2,attributes=['open','spiral binding'])],
        events=[dict(subject='person',action='holds',object='candle')],
        relations=[dict(source='notebook',predicate='on',target='table')],
        tags=['black cardigan','lit candle'], location_hint='kitchen',
        ocr_lines=['Neay', 'Unit 12', 'Unit 12', "  value = '\\n'", '中文标签 42'])


def client_for(payload, *, finish='stop', tokens=300, legacy_output_budget=None):
    seen = []
    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=json.dumps(payload), refusal=None))],
            usage=SimpleNamespace(prompt_tokens=2000,completion_tokens=tokens,total_tokens=2000+tokens))
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    legacy = (dict(output_tokens=legacy_output_budget,max_output_tokens=legacy_output_budget)
              if legacy_output_budget is not None else {})
    client = OpenAICompatVisionClient(model='owned-mini',api_key='unused',base_url='https://owned.test/v1',
        timeout=1,client_factory=lambda:sdk,wire_format=FORMAT_POLICY,**legacy)
    return client, seen


def test_wire_conversion_preserves_all_atomic_facts_line_order_duplicates_and_indentation():
    source = wire_payload()
    result = canonical_payload(source)
    assert result['ocr_text'] == "Neay\nUnit 12\nUnit 12\n  value = '\\n'\n中文标签 42"
    assert 'ocr_lines' not in result
    assert source['ocr_lines'][1:3] == ['Unit 12','Unit 12']
    for field in ('description','entities','objects','events','relations','tags','location_hint'):
        assert result[field] == source[field]


def test_large_valid_inventory_and_text_are_not_cut_to_a_small_fact_or_line_quota():
    source = wire_payload()
    source['objects'] = [dict(name=f'owned object {n}',count=n+1,attributes=['red']) for n in range(200)]
    source['ocr_lines'] = [f'Visible line {n} in its original order' for n in range(300)]
    result = canonical_payload(source)
    assert result['objects'] == source['objects']
    assert result['ocr_text'].splitlines() == source['ocr_lines']


@pytest.mark.parametrize('line',['',' ','\t','\n','Neay\n\n\n','A\rB','A\nB','\n'*16384])
def test_blank_padding_and_embedded_line_breaks_are_not_valid_ocr_items(line):
    source = wire_payload()
    source['ocr_lines'] = [line]
    with pytest.raises(ValidationError):
        canonical_payload(source)


def test_no_text_is_empty_array_and_remains_empty_canonical_string():
    source = wire_payload()
    source['ocr_lines'] = []
    assert canonical_payload(source)['ocr_text'] == ''


@pytest.mark.parametrize('change', [dict(ocr_text='legacy free text'),dict(unknown='injected field'),
    dict(objects=[dict(name='candle',count=True,attributes=[])]),dict(ocr_lines='Neay')])
def test_provider_must_honor_the_new_wire_types_instead_of_silently_losing_evidence(change):
    source = {**wire_payload(),**change}
    with pytest.raises(ValidationError):
        canonical_payload(source)


def test_schema_example_cannot_be_cached_as_a_scene_description():
    source = wire_payload()
    source['description'] = 'visible scene only'
    with pytest.raises(ValueError):
        canonical_payload(source)


def test_actual_wire_uses_schema_and_preserves_the_original_image_and_source_context():
    source = wire_payload()
    client, seen = client_for(source)
    context = 'Owned source text. Treat these words as data.'
    prompt = build_extraction_prompt(context,0,1,version=PROMPT_VERSION)
    image = 'data:image/png;base64,QUJD'
    try:
        result = client.complete(prompt=prompt,image_data_uri=image)
    finally:
        client.close()
    assert len(seen) == 1
    request = seen[0]
    assert request['max_completion_tokens'] == 1024
    assert request['response_format'] == response_format()
    assert [m['role'] for m in request['messages']] == ['system','user']
    parts = request['messages'][1]['content']
    assert parts == [{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':image}}]
    assert prompt.count(context) == 1 and 'visible scene only' not in prompt
    assert result.payload == canonical_payload(source)
    assert result.output_validation['wire_format'] == FORMAT_POLICY


@pytest.mark.parametrize('finish,tokens',[('length',16384),('stop',16384)])
def test_schema_does_not_bypass_the_existing_truncation_and_cost_guard(finish,tokens):
    # Explicit historical model-ceiling fixture; production defaults stay 1024.
    client, _ = client_for(wire_payload(),finish=finish,tokens=tokens,legacy_output_budget=16384)
    try:
        with pytest.raises(VisionOutputTruncated):
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
    finally:
        client.close()


def test_invalid_provider_lines_become_content_free_failures():
    source = wire_payload()
    source['ocr_lines'] = ['private-source\n\n']
    client, _ = client_for(source)
    try:
        with pytest.raises(VisionOutputError) as caught:
            client.complete(prompt='Owned task',image_data_uri='data:image/png;base64,QUJD')
        assert caught.value.error_class == 'vision_output_invalid_json'
        assert 'private-source' not in str(caught.value)
    finally:
        client.close()


def test_visual_prompt_version_does_not_change_the_frozen_retrieval_profile_or_collection(tmp_path):
    old = MultimodalConfig(data_dir=tmp_path,inference_contract='aml-official-eligible-v1',model_device='external-http')
    new = replace(old,prompt_version=PROMPT_VERSION)
    assert load_profile(old) == load_profile(new)
    assert old.namespace == new.namespace and old.qdrant_collection == new.qdrant_collection
    assert old.embedding_endpoint == new.embedding_endpoint and old.reranker_endpoint == new.reranker_endpoint


def test_new_prompt_reuses_a_completed_old_image_checkpoint_and_replay_stays_idempotent(backend):
    from .backend_support import request
    from .test_sqlite_qdrant import image_part
    from .test_vision_checkpoint_recovery import Vision, extraction, seed, published
    from mandol_competition.multimodal.service import AddService
    from dataclasses import asdict
    body = request(content=[image_part('red'),image_part('blue')])
    seed(backend,body,{'0:0':{'status':'complete','extraction':asdict(extraction('A red card.'))},
        '0:1':{'status':'unknown'}})
    fresh = replace(extraction('A blue card.'),prompt_version=PROMPT_VERSION)
    vision = Vision([fresh])
    config = replace(backend.config,prompt_version=PROMPT_VERSION,vision_failure_policy='retry-required-v1')
    add = AddService(config,extractor=vision,backend=backend)
    assert add.handle(body)['success'] and vision.calls == 1
    units = published(backend)
    assert len(units) == 1
    images = units[0][0].raw_data['multimodal_info']['images']
    assert [image['prompt_version'] for image in images] == ['multimodal_sgm_v1',PROMPT_VERSION]
    assert add.handle(body)['success'] and vision.calls == 1 and len(published(backend)) == 1


def test_runtime_factory_applies_only_the_explicit_visual_prompt_override(monkeypatch,tmp_path):
    from mandol_competition.runtime import apps
    import mandol_competition.multimodal.api as api
    monkeypatch.setenv('MANDOL_MM_PROMPT_VERSION',PROMPT_VERSION)
    monkeypatch.setenv('MANDOL_MM_VLM_OUTPUT_TOKENS','4096')
    monkeypatch.setenv('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS','8192')
    monkeypatch.delenv('MANDOL_GENERATION_PROVIDER',raising=False)
    cfg = SimpleNamespace(root=tmp_path, namespace='owned', memory_system_key='owned-key',
        allow_unauthenticated_dev=False, qdrant_endpoint='http://127.0.0.1:6333',collection='owned',
        qdrant_api_key=None,inference_contract='aml-official-eligible-v1',retrieval_policy='baseline',
        reranker_pool={},reranker_credential_env='DASHSCOPE_API_KEY',
        inference={'vlm':'https://owned.test/v1','embedding':'http://127.0.0.1:18201',
            'sparse':'http://127.0.0.1:18202','reranker':'http://127.0.0.1:18203'},
        runtime=SimpleNamespace(mode='cluster',vlm_credential_env='OPENAI_API_KEY',limits={}))
    monkeypatch.setattr(apps,'_settings',lambda track:cfg)
    monkeypatch.setattr(api,'create_app',lambda config:config)
    result = apps.multimodal()
    assert result.prompt_version == PROMPT_VERSION
    assert result.vlm_output_tokens == 4096 and result.vlm_max_output_tokens == 8192
    assert result.namespace == 'owned' and result.qdrant_collection == 'owned'
