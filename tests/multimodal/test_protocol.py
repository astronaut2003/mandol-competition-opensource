"""Offline checks for the AML wire contract and production-safe configuration."""

from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest
from pydantic import TypeAdapter, ValidationError

from mandol_competition.multimodal.config import ConfigurationError, MultimodalConfig, get_config
from mandol_competition.multimodal.protocol import (
    AddRequest,
    AddResponse,
    ContentPart,
    ImagePart,
    SearchItem,
    SearchRequest,
    SearchResponse,
    TextPart,
    join_text_parts,
)

PART = TypeAdapter(ContentPart)
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def test_content_part_discriminates_both_variants():
    assert isinstance(PART.validate_python({"type": "text", "text": "hi"}), TextPart)
    assert isinstance(PART.validate_python(IMAGE), ImagePart)


@pytest.mark.parametrize("payload", [
    {"type": "video", "url": "x"},
    {"type": "text", "text": "hi", "image_url": IMAGE["image_url"]},
    {**IMAGE, "text": "sneaky"},
    {"type": "text", "text": ""},
    {"type": "image_url", "image_url": {}},
    {"type": "image_url", "image_url": {"url": "", "extra": "x"}},
])
def test_content_part_rejects_unknown_mixed_and_empty_values(payload):
    with pytest.raises(ValidationError):
        PART.validate_python(payload)


def test_add_accepts_string_and_preserves_order_and_timestamp():
    request = AddRequest(
        request_id="r1",
        user_id="u1",
        session_id="s1",
        messages=[
            {"role": "assistant", "content": "hello"},
            {"role": "user", "timestamp": 1704067200000, "content": [
                {"type": "text", "text": "look"}, IMAGE, {"type": "text", "text": "after"},
            ]},
        ],
    )
    assert request.messages[0].parts() == [TextPart(type="text", text="hello")]
    assert [part.type for part in request.messages[1].parts()] == ["text", "image_url", "text"]
    assert request.messages[1].timestamp == 1704067200000
    assert join_text_parts(request.messages[1].parts()) == "look\nafter"


@pytest.mark.parametrize("overrides", [
    {"request_id": ""}, {"messages": []}, {"user_id": ""}, {"session_id": ""},
    {"messages": [{"role": "user", "content": ""}]},
    {"messages": [{"role": "user", "content": []}]},
    {"messages": [{"role": "system", "content": "hello"}]},
    {"messages": [{"role": "user", "content": "hello", "metadata": {}}]},
    {"metadata": {}},
])
def test_add_rejects_invalid_fields(overrides):
    payload = {"request_id": "r1", "messages": [{"role": "user", "content": "hello"}],
               "user_id": "u1", "session_id": "s1"}
    with pytest.raises(ValidationError):
        AddRequest.model_validate({**payload, **overrides})


def test_add_response_has_only_contract_fields():
    response = AddResponse(success=True, request_id="r1", user_id="u1", session_id="s1")
    assert response.model_dump() == {
        "success": True, "request_id": "r1", "user_id": "u1", "session_id": "s1",
    }
    with pytest.raises(ValidationError):
        AddResponse(success=False, request_id="r1", user_id="u1", session_id="s1")
    with pytest.raises(ValidationError):
        AddResponse(success=True, request_id="r1", user_id="u1", session_id="s1", memory_ids=[])


def test_search_accepts_ordered_parts_and_options():
    request = SearchRequest(query=[IMAGE, {"type": "text", "text": "then?"}],
                            options=["A", "B"], user_id="u1", top_k=100)
    assert [part.type for part in request.parts()] == ["image_url", "text"]
    assert request.options == ["A", "B"]


def test_search_string_and_single_text_part_have_identical_representation():
    text = "  Which database mode?\nKeep original whitespace.  "
    plain = SearchRequest(query=text, options=["A", "B"], user_id="u1", top_k=100)
    parts = SearchRequest(query=[{"type": "text", "text": text}],
                          options=["A", "B"], user_id="u1", top_k=100)
    assert plain.parts() == parts.parts()
    assert join_text_parts(plain.parts()) == text
    assert plain.options == parts.options


@pytest.mark.parametrize("overrides", [
    {"query": ""}, {"query": []}, {"query": 12}, {"top_k": 0}, {"user_id": ""}, {"filters": {}},
])
def test_search_rejects_invalid_request(overrides):
    with pytest.raises(ValidationError):
        SearchRequest.model_validate({"query": [{"type": "text", "text": "q"}],
                                      "user_id": "u1", "top_k": 100, **overrides})


def test_search_response_has_exact_wrapper_and_rank_order():
    response = SearchResponse(data=[
        SearchItem(id="first", content=[{"type": "text", "text": "evidence"}, IMAGE], score=0.87,
                   created_at="2026-09-01T12:00:00Z"),
        SearchItem(id="second", content=[{"type": "text", "text": "more"}]),
    ])
    assert response.model_dump(exclude_none=True) == {"data": [
        {"id": "first", "content": [{"type": "text", "text": "evidence"}, IMAGE],
         "score": 0.87, "created_at": "2026-09-01T12:00:00Z"},
        {"id": "second", "content": [{"type": "text", "text": "more"}]},
    ]}
    assert SearchResponse(data=[]).model_dump() == {"data": []}
    with pytest.raises(ValidationError):
        SearchResponse.model_validate({"data": [], "items": []})


@pytest.mark.parametrize("payload", [{"id": "", "content": [{"type": "text", "text": "x"}]},
                                     {"id": "a", "content": []},
                                     {"id": "a", "content": [IMAGE], "metadata": {}}])
def test_search_item_rejects_malformed_or_extra_fields(payload):
    with pytest.raises(ValidationError):
        SearchItem.model_validate(payload)


def test_config_defaults_and_user_isolation(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path)
    assert (config.max_images, config.image_byte_budget, config.max_image_bytes) == (
        10, 28 * 1024 * 1024, 10 * 1024 * 1024,
    )
    assert (config.vlm_max_attempts, config.max_graphs, config.snapshot_every) == (3, 64, 100)
    assert config.retrieval_methods == ("bm25", "cosine_similarity", "splade")
    assert config.rerank_method == "baai"
    assert config.user_key_for("eval:run/1") == sha256(b"eval:run/1").hexdigest()
    assert config.user_key_for("a/b") != config.user_key_for("a:b")
    assert config.user_root("u1") == (tmp_path / "media" / sha256(b"multimodal").hexdigest()
                                      / "users" / config.user_key_for("u1"))
    with pytest.raises(FrozenInstanceError):
        config.max_images = 11


@pytest.mark.parametrize("kwargs", [
    {"env": "production"}, {"env": "production", "vlm_backend": "fake"},
    {"env": "production", "vlm_backend": "other"},
    {"max_image_bytes": 11 * 1024 * 1024},
    {"max_request_image_bytes": 31 * 1024 * 1024},
    {"image_byte_budget": 40 * 1024 * 1024},
    {"embedding_model_name": "local-model"},
    {"embedding_model_name": "local-model", "embedding_dim": -1},
    {"retrieval_methods": ()},
])
def test_config_rejects_fake_production_and_over_limit_budgets(kwargs):
    with pytest.raises(ConfigurationError):
        MultimodalConfig(**kwargs)


def test_real_production_config_does_not_connect_to_provider(monkeypatch):
    monkeypatch.setenv("MANDOL_MM_ENV", "production")
    monkeypatch.setenv("MANDOL_MM_VLM_BACKEND", "openai_compat")
    monkeypatch.setenv("MANDOL_MM_VLM_MODEL", "vision-model")
    monkeypatch.setenv("MANDOL_MM_RERANK_METHOD", "")
    get_config.cache_clear()
    try:
        config = get_config()
        assert config.env == "production"
        assert config.vlm_backend == "openai_compat"
        assert config.vlm_model == "vision-model"
        assert config.rerank_method is None
    finally:
        get_config.cache_clear()


def test_local_embedding_and_retrieval_settings_from_environment(monkeypatch):
    monkeypatch.setenv("MANDOL_MM_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("MANDOL_MM_EMBEDDING_DIM", "384")
    monkeypatch.setenv("MANDOL_MM_RETRIEVAL_METHODS", "bm25, cosine_similarity")
    get_config.cache_clear()
    try:
        config = get_config()
        assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.embedding_dim == 384
        assert config.retrieval_methods == ("bm25", "cosine_similarity")
    finally:
        get_config.cache_clear()
