import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mandol_competition.multimodal.api import build_extractor, create_app
from mandol_competition.multimodal.config import ConfigurationError, MultimodalConfig
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor


class Graph:
    def __init__(self):
        self.units = {}

    def add_unit(self, unit, **kwargs):
        self.units[unit.uid] = unit

    def rebuild_all_indexes(self):
        pass

    def get_multi_retriever(self):
        return self

    def smart_search(self, query, **kwargs):
        return [(unit, 0.9) for unit in list(self.units.values())[: kwargs["top_k"]]]

    def close(self):
        pass


def image_uri():
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def add_body(uri=None):
    parts = [{"type": "text", "text": "red backpack"}]
    if uri is not None:
        parts.append({"type": "image_url", "image_url": {"url": uri}})
    return {"request_id": "r1", "user_id": "alice", "session_id": "s1",
            "messages": [{"role": "user", "content": parts}]}


def test_http_contract_auth_health_and_errors(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path, auth_token="secret", rerank_method=None)
    extractor = FakeVisionExtractor({}, model="mock", prompt_version="v1", schema_version="1")
    app = create_app(config, extractor=extractor, graph_factory=lambda user: Graph())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/add", json=add_body()).status_code == 401
        assert client.post("/admin/purge", params={"user_id": "alice"}).status_code == 401
        assert client.post("/search", json={"query": [{"type": "text", "text": "x"}],
                                            "user_id": "alice", "top_k": 100}).status_code == 401
        headers = {"X-Api-Key": "secret"}
        added = client.post("/add", headers=headers, json=add_body(image_uri()))
        assert added.status_code == 200
        assert added.json() == {"success": True, "request_id": "r1", "user_id": "alice", "session_id": "s1"}
        query = {"query": [{"type": "text", "text": "backpack"}], "user_id": "alice", "top_k": 100}
        result = client.post("/search", headers={"Authorization": "Bearer secret"}, json=query)
        assert result.status_code == 200
        assert list(result.json()) == ["data"]
        assert [part["type"] for part in result.json()["data"][0]["content"]] == ["text", "image_url"]
        assert result.json()["data"][0]["content"][1]["image_url"]["url"] == image_uri()
        assert client.post("/add", headers={"Authorization": "Token secret"},
                           json=add_body(image_uri())).json() == added.json()
        assert client.post("/add", headers=headers, json=add_body()).status_code == 409
        assert client.post("/add", headers=headers, json=add_body("https://example.org/x")).status_code == 422
        string_result = client.post("/search", headers=headers, json={**query, "query": "backpack"})
        assert string_result.status_code == 200 and string_result.json() == result.json()
        assert client.post("/search", headers=headers, json={**query, "query": ""}).status_code == 422
        assert client.post("/search", headers=headers, json={**query, "user_id": "bob"}).json() == {"data": []}
        purged = client.post("/admin/purge", headers=headers, params={"user_id": "alice"})
        assert purged.status_code == 200
        assert purged.json()["units_removed"] == 1
        assert client.post("/search", headers=headers, json=query).json() == {"data": []}


def test_production_requires_auth_endpoint_and_key(tmp_path, monkeypatch):
    config = MultimodalConfig(data_dir=tmp_path, env="production", vlm_backend="openai_compat")
    with pytest.raises(ConfigurationError, match="AUTH_TOKEN"):
        create_app(config)
    config = MultimodalConfig(data_dir=tmp_path, env="production", vlm_backend="openai_compat",
                              auth_token="secret")
    with pytest.raises(ConfigurationError, match="vlm_base_url"):
        create_app(config)
    config = MultimodalConfig(data_dir=tmp_path, env="production", vlm_backend="openai_compat",
                              auth_token="secret", vlm_base_url="https://example.invalid/v1",
                              vlm_api_key_env="TEST_VISION_KEY")
    monkeypatch.delenv("TEST_VISION_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="TEST_VISION_KEY"):
        create_app(config)
    monkeypatch.setenv("TEST_VISION_KEY", "test-key")
    assert build_extractor(config).model_name == config.vlm_model
    from mandol_competition.multimodal.models import load_profile
    assert load_profile(config).embedding_dimension == 1024  # packaged frozen recipe
    with pytest.raises(ConfigurationError, match="experimental"):
        create_app(config, graph_factory=lambda _: Graph())


def test_build_extractor_fake_only_in_development(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path)
    assert isinstance(build_extractor(config), FakeVisionExtractor)
