from pathlib import Path

import pytest

from mandol_competition.multimodal.config import MultimodalConfig,ConfigurationError
from mandol_competition.multimodal.models import load_profile


def test_native_contract_selects_packaged_frozen_profile_without_repo_side_dependency():
    config=MultimodalConfig(inference_contract='aml-official-eligible-v1',model_device='external-http')
    actual=load_profile(config)
    expected=load_profile(MultimodalConfig(model_device='external-http',
        retrieval_profile_path=Path('configs/profiles/aml-official-eligible/multimodal-v1.json')))
    assert actual.hash==expected.hash
    assert actual.options['inference_recipe']=='aml-official-eligible-v1'
    assert actual.embedding_model=='text-embedding-v4'


def test_native_flag_cannot_silently_bind_research_qwen_profile():
    with pytest.raises(ConfigurationError,match='research model profile'):
        load_profile(MultimodalConfig(inference_contract='aml-official-eligible-v1',model_device='external-http',
            retrieval_profile_path=Path('src/mandol_competition/multimodal/profiles/vllm-bf16-v1.json')))
