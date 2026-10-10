from dataclasses import replace
from types import SimpleNamespace

from mandol_competition.multimodal.config import MultimodalConfig, get_config
from mandol_competition.multimodal.service import _cached_extractor
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient


def test_explicit_wire_model_preserves_deployment_alias():
    captured = {}

    def complete(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"description":"red square"}'))], usage=None)

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    client = OpenAICompatVisionClient(model="gpt-4o-mini-closeai", request_model="gpt-4o-mini",
        api_key="fixture", base_url="https://example.invalid/v1", timeout=2,
        client_factory=lambda: sdk)
    result = client.complete(prompt="describe", image_data_uri="data:image/png;base64,QQ==")
    assert captured["model"] == "gpt-4o-mini"
    assert result.model == "gpt-4o-mini-closeai"


def test_provider_identity_invalidates_extraction_cache(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path, vlm_model="alias",
        vlm_request_model="wire-v1", vlm_base_url="https://example.invalid/v1")
    extractor = FakeVisionExtractor({}, model="alias", prompt_version="v1", schema_version="1")
    def key(cfg):
        return _cached_extractor(cfg, "user", extractor)._key("img_abc", "context", 0, 1)
    assert key(config) == key(replace(config, vlm_api_key_env="ROTATED_SECRET"))
    assert key(config) != key(replace(config, vlm_request_model="wire-v2"))
    assert key(config) != key(replace(config, vlm_base_url="https://other.invalid/v1"))
    assert _cached_extractor(config, "user", extractor).model_name == "alias"


def test_wire_model_environment_is_explicit(monkeypatch):
    get_config.cache_clear()
    monkeypatch.setenv("MANDOL_MM_VLM_MODEL", "gpt-4o-mini-closeai")
    monkeypatch.setenv("MANDOL_MM_VLM_REQUEST_MODEL", "gpt-4o-mini")
    try:
        assert get_config().vlm_model == "gpt-4o-mini-closeai"
        assert get_config().vlm_request_model == "gpt-4o-mini"
    finally:
        get_config.cache_clear()
