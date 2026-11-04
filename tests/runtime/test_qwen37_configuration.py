from dataclasses import replace
import json
from pathlib import Path

import pytest

from mandol_competition.config import load_config
from mandol_competition.inference.dashscope_rerank import BACKEND, ENDPOINT, MODEL
from mandol_competition.runtime.__main__ import child_environment, describe, profile_path
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.search_profiles import BASELINE, QWEN37, descriptor, identity
from mandol_competition.multimodal.inference import remote_reranker


def test_unified_new_v4_default_is_cloud_without_changing_code_or_storage(tmp_path, monkeypatch):
    monkeypatch.delenv('MANDOL_CONFIG', raising=False)
    config = tmp_path / 'cloud.toml'
    config.write_text('[textual]\ninference_contract="aml-official-eligible-v1"\n'
                      '[multimodal]\ninference_contract="aml-official-eligible-v1"\n')
    cfg = load_config(config)
    assert cfg.track('textual').reranker_backend == BACKEND
    assert cfg.track('multimodal').retrieval_policy == QWEN37
    assert cfg.track('textual').inference['reranker'] == ENDPOINT
    assert cfg.track('multimodal').inference['reranker'] == ENDPOINT
    assert cfg.track('code').reranker_backend == 'profile'
    assert set(cfg.track('code').inference) == {'embedding'}
    assert cfg.track('textual').evidence_profile == 'default'


def test_explicit_querit_rollback_and_historical_profiles_remain_loadable(tmp_path):
    config = tmp_path / 'old.toml'
    config.write_text('[textual]\ninference_contract="aml-official-eligible-v1"\nreranker_backend="profile"\n'
                      '[multimodal]\ninference_contract="aml-official-eligible-v1"\nretrieval_policy="mm-querit-same-pool-rank-only-v1"\n')
    cfg = load_config(config)
    assert cfg.track('textual').reranker_backend == cfg.track('multimodal').reranker_backend == 'profile'
    for mode in ('local', 'cluster'):
        assert load_config(profile_path(mode)).track('textual').reranker_backend == 'profile'
        assert load_config(profile_path(mode)).track('code').reranker_backend == 'profile'


def test_explicit_legacy_endpoint_remains_frozen_and_explicit_provider_endpoint_is_preserved(tmp_path):
    path = tmp_path / 'explicit.toml'
    path.write_text('[textual]\ninference_contract="aml-official-eligible-v1"\n'
                    '[textual.inference]\nreranker="http://querit.example:8003"\n')
    assert load_config(path).track('textual').reranker_backend == 'profile'
    alternate = ENDPOINT.replace('cn-beijing.maas.qianwenaiapi.com', 'workspace.cn-beijing.maas.aliyuncs.com')
    path.write_text('[textual]\ninference_contract="aml-official-eligible-v1"\n'
                    '[textual.inference]\nreranker="' + alternate + '"\n')
    cfg = load_config(path).track('textual')
    assert cfg.reranker_backend == BACKEND and cfg.inference['reranker'] == alternate


def test_cloud_backend_refuses_private_querit_endpoint_before_opening_storage(tmp_path):
    path = tmp_path / 'wrong.toml'
    path.write_text('[textual]\nreranker_backend="dashscope-qwen37-v1"\n'
                    '[textual.inference]\nreranker="http://querit.example:8003"\n')
    with pytest.raises(ValueError, match='HTTPS'):
        load_config(path)


def test_cloud_credential_forwarding_is_explicit_and_not_in_describe(tmp_path, monkeypatch):
    source = profile_path('local').read_text().replace('reranker_backend = "profile"',
        'reranker_backend = "dashscope-qwen37-v1"')
    # Historical examples contain private endpoints; use the native public
    # endpoint for this cloud composition and remove private replica lists.
    import re
    source = re.sub(r'^reranker = .*$', 'reranker = "' + ENDPOINT + '"', source, flags=re.M)
    path = tmp_path / 'cloud.toml'
    path.write_text(source)
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only-cloud-value')
    cfg = load_config(path)
    assert child_environment(cfg, path)['DASHSCOPE_API_KEY'] == 'test-only-cloud-value'
    assert 'test-only-cloud-value' not in json.dumps(describe(cfg))
    assert all(t.reranker_pool['endpoints'] == [ENDPOINT] for n, t in cfg.tracks.items() if n != 'code')


def test_qwen_overlay_retains_exact_m0_storage_hash_and_declares_real_model(monkeypatch):
    original = MultimodalConfig(inference_contract='aml-official-eligible-v1', model_device='external-http')
    changed = replace(original, search_policy=QWEN37, reranker_endpoint=ENDPOINT)
    profile = load_profile(original)
    assert profile == load_profile(changed)
    assert identity(profile, QWEN37) != identity(profile, BASELINE)
    assert descriptor(profile, QWEN37)['reranker']['model_id'] == MODEL
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-only')
    client = remote_reranker(changed, profile)
    try:
        assert client.recipe.model_id == MODEL
        assert client._replicas[0].client.endpoint == ENDPOINT
    finally:
        client.close()


def test_code_reranker_requires_its_named_v4_contract(tmp_path):
    path = tmp_path / 'code.toml'
    path.write_text('[code]\nreranker_backend="dashscope-qwen37-v1"\n')
    with pytest.raises(ValueError, match='official v4'):
        load_config(path)


def test_code_v4_defaults_to_qwen_and_explicit_rollback_preserves_native_projection(tmp_path):
    path = tmp_path / 'code.toml'
    text = '[code]\ninference_contract="aml-official-eligible-v1"\nretrieval_policy="code-native-sparse-rrf100-v1"\n'
    path.write_text(text)
    active = load_config(path).track('code')
    assert active.reranker_backend == BACKEND
    assert active.inference['reranker'] == ENDPOINT and active.reranker_pool == {}
    path.write_text(text + 'reranker_backend="profile"\n')
    rollback = load_config(path).track('code')
    assert set(rollback.inference) == {'embedding'}
    assert active.root == rollback.root and active.collection == rollback.collection
    assert active.retrieval_policy == rollback.retrieval_policy
    path.write_text('[code]\ninference_contract="aml-official-eligible-v1"\nretrieval_policy="c0-original-v1"\n')
    assert load_config(path).track('code').reranker_backend == 'profile'
