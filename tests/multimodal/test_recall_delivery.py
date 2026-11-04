"""Artificial image delivery and provenance checks, independent of model quality."""
import base64
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.evidence_review import ContextSource, EvidencePlan, review_sources
from mandol_competition.multimodal.grounded_delivery import deliver
from .test_partitioned_evidence import Guard, query
from .test_grounded_evidence import memory


def media_table(count):
    bodies={f'image-{i}':(f'owned-artificial-{i}'.encode(),'image/png') for i in range(count)}
    return bodies,SimpleNamespace(get=bodies.get)


def attached(data):
    return [base64.b64decode(part['image_url']['url'].split(',',1)[1])
        for row in data for part in row['content'] if part['type']=='image_url']


def test_positive_eleventh_original_is_delivered_with_the_unchanged_atomic_floor():
    bodies,media=media_table(11)
    pool=[(memory(str(i),'Useful ORBIT-47' if i==10 else 'Unrelated scenery.',images=[f'image-{i}']),1-i/20) for i in range(11)]
    before=deepcopy(pool)
    plan=review_sources(Guard(lambda text:'ORBIT-47' in text),query(),pool,media)
    data=deliver(MultimodalConfig(max_images=10),plan,media)['data']
    assert [r['id'] for r in data]==[u.uid for u,_ in pool]
    assert [r['score'] for r in data]==[s for _,s in pool]
    assert bodies['image-10'][0] in attached(data) and len(attached(data))==10
    assert 'ORBIT-47' in json.dumps(data) and 'image_count_budget' in json.dumps(data)
    assert pool==before


def test_two_selected_sources_share_budget_fairly_before_unselected_images():
    bodies,media=media_table(15)
    many=memory('many','Source A.',images=[f'image-{i}' for i in range(14)])
    other=memory('one','Source B.',images=['image-14'])
    plan=EvidencePlan([(many,.9),(other,.8)],(many,other))
    wire=deliver(MultimodalConfig(max_images=2),plan,media)['data']
    assert set(attached(wire))=={bodies['image-0'][0],bodies['image-14'][0]}


@pytest.mark.parametrize('limits,reason', [({'max_images':1},'image_count_budget'),
    ({'max_image_bytes':1},'single_image_byte_budget'),({'image_byte_budget':1},'response_image_byte_budget')])
def test_real_budgets_preserve_all_original_text_and_report_omissions(limits,reason):
    _,media=media_table(2)
    pool=[(memory(str(i),f'Original fact {i}.',images=[f'image-{i}']),.01-i/1000) for i in range(2)]
    wire=deliver(replace(MultimodalConfig(),**limits),EvidencePlan(pool,(pool[1][0],)),media)['data']
    text=json.dumps(wire)
    assert reason in text and all(f'Original fact {i}.' in text for i in range(2))
    assert [r['score'] for r in wire]==[s for _,s in pool]
    json.dumps(wire,allow_nan=False)


def test_shared_context_original_is_attached_once_and_references_a_real_outer_id():
    bodies,media=media_table(1)
    a=memory('a','Parent source.',images=['image-0'])
    b=memory('b','Correction in another session.',session='later',role='assistant',images=['image-0'])
    context=ContextSource('a',b,'shared-explicit-text-anchor')
    wire=deliver(MultimodalConfig(),EvidencePlan([(a,.6)],(b,),(context,)),media)['data']
    assert len(wire)==1 and wire[0]['id']=='a' and attached(wire)==[bodies['image-0'][0]]
    text=json.dumps(wire)
    assert '"source_memory_id": "b"' in text.replace('\\"','"')
    assert 'image_in_memory' in text and 'later' in text and 'assistant' in text
    assert 'Correction in another session.' in text


def test_deduplication_never_truncates_original_or_generated_visual_fields():
    original='Exact raw text\n'+'尾部原子事实。'*600
    unit=memory('long','[Dialogue] '+original+'\nOCR: ORBIT-47',images=['absent'])
    unit.raw_data['original_content']=original
    unit.raw_data['content_parts'].insert(0,{'type':'text','text':original})
    before=deepcopy(unit)
    wire=deliver(MultimodalConfig(),EvidencePlan([(unit,.001)],(unit,)),SimpleNamespace(get=lambda _:None))['data']
    texts=[p['text'] for p in wire[0]['content'] if p['type']=='text']
    assert original in texts and 'OCR: ORBIT-47' in texts
    assert sum(original in text for text in texts)==1 and unit==before


def test_context_support_can_veto_empty_without_consuming_or_changing_top_k():
    seed=memory('seed','I bought the one selected before.')
    context=memory('earlier','The earlier folder was ORBIT-47.',session='previous')
    guard=Guard(lambda text:'ORBIT-47' in text)
    plan=review_sources(guard,query('Which folder was it?'),[(seed,.2)],SimpleNamespace(get=lambda _:None),
                        [ContextSource('seed',context,'shared-explicit-text-anchor')])
    assert plan.hits==[(seed,.2)] and plan.contexts[0].unit.uid=='earlier'
    wire=deliver(MultimodalConfig(),plan,SimpleNamespace(get=lambda _:None))['data']
    assert len(wire)==1 and wire[0]['id']=='seed' and wire[0]['score']==.2
    assert 'ORBIT-47' in json.dumps(wire) and 'previous' in json.dumps(wire)


def test_v4_is_a_distinct_search_identity_with_the_same_frozen_storage_profile():
    from mandol_competition.multimodal.models import load_profile
    from mandol_competition.multimodal.search_profiles import BASELINE, descriptor, identity
    base=MultimodalConfig(inference_contract='aml-official-eligible-v1',model_device='external-http')
    config=replace(base,vlm_backend='openai_compat',evidence_policy='source-grounded-recall-v4')
    profile=load_profile(config)
    assert profile.hash==load_profile(base).hash
    method=descriptor(profile,BASELINE,config.evidence_policy)
    assert method['storage_migration'] is False and method['max_selector_calls']==16
    assert method['max_context_sources']==8 and all(len(v)==64 for v in method['response_policy_sha256'].values())
    assert identity(profile,BASELINE,config.evidence_policy)!=identity(profile,BASELINE,'source-grounded-delivery-v3')
