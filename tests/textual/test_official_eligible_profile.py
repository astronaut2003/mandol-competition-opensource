import hashlib
import json

import httpx
import pytest

from mandol_competition.textual.api.config import Settings, AML_V4_QUALITY_SHA256, QUALITY_SHA256
from mandol_competition.textual.core.profile import load_profile
from mandol_competition.textual.runtime import seed_frozen_capsule
from mandol_competition.inference.dashscope import official_embedding_recipes
from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPJointEncoders
from mandol_competition.inference.http import HTTPTransport
from mandol_substrate.state.models import ProfileMismatch


def settings(root):
    return Settings(competition_run_root=str(root),evaluation_namespace='textual',
        memory_system_key='unit-only-key',inference_contract='aml-official-eligible-v1',
        qdrant_collection='textual_aml_v4_v1',embedding_url='http://127.0.0.1:18120',
        splade_url='http://127.0.0.1:18120')


def test_official_named_family_is_explicit_and_keeps_research_capsule(tmp_path):
    original=seed_frozen_capsule(tmp_path/'research')
    before={p.name:p.read_bytes() for p in original.glob('*.json')}
    admitted=settings(tmp_path/'native').bind_run()
    profile=load_profile(admitted.competition_profile)
    assert profile.embedding_model==profile.splade_model=='text-embedding-v4'
    assert profile.embedding_revision=='provider-managed-unattested'
    assert profile.options['sparse_index_bound']==2**32
    assert profile.options['splade_max_length']==8192
    assert hashlib.sha256((tmp_path/'native/quality-profile.json').read_bytes()).hexdigest()==AML_V4_QUALITY_SHA256
    assert hashlib.sha256((original/'quality-profile.json').read_bytes()).hexdigest()==QUALITY_SHA256
    assert all((original/name).read_bytes()==data for name,data in before.items())
    with pytest.raises(ValueError,match='differs'):
        settings(original).bind_run()


def test_wrong_collection_or_independent_sparse_endpoint_fails_closed(tmp_path):
    cfg=settings(tmp_path)
    cfg.qdrant_collection='textual_v1'
    with pytest.raises(ValueError,match='collection'):cfg.validate()
    cfg=settings(tmp_path);cfg.splade_url='http://127.0.0.1:18102'
    with pytest.raises(ValueError,match='paired'):cfg.validate()


def test_native_capsule_cannot_lie_about_model_or_sparse_bound(tmp_path):
    root=seed_frozen_capsule(tmp_path,'aml_official_v4_v1')
    path=root/'profile.json';original=json.loads(path.read_text())
    for patch in ({'embedding_model':'Qwen/Qwen3-Embedding-0.6B'},
                  {'options':original['options']|{'sparse_index_bound':30522}}):
        path.write_text(json.dumps(original|patch))
        with pytest.raises(ProfileMismatch):load_profile(path)


def test_joint_native_query_document_mode_does_not_use_qwen_prefix_or_30522_bound():
    dense_recipe,sparse_recipe=official_embedding_recipes();calls=[]
    def handler(request):
        data=json.loads(request.content);calls.append(data)
        return httpx.Response(200,json={'model':'text-embedding-v4','data':[{
            'index':0,'pair':{'embedding':[1.]+[0.]*1023,'sparse':{'51765':.7}}}]})
    dense=HTTPEmbeddingClient('http://unit',dense_recipe,
        transport=HTTPTransport('http://unit',transport=httpx.MockTransport(handler)))
    sparse=HTTPSparseEncoderClient('http://unit',sparse_recipe,
        transport=HTTPTransport('http://unit',transport=httpx.MockTransport(handler)))
    encoders=HTTPJointEncoders(dense,sparse)
    assert encoders.encode_query('exact query')[1]=={51765:.7}
    assert encoders.encode_documents(['exact document'])[0][1]=={51765:.7}
    assert calls[0]['input']==['exact query'] and calls[0]['input_type']=='query'
    assert calls[1]['input']==['exact document'] and calls[1]['input_type']=='document'
    encoders.close()
