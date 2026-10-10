import base64
import hashlib
import json

import pytest

from mandol_competition.multimodal.vlm.base import Extraction, ExtractionError, VisionExtractor, parse_extraction
from mandol_competition.multimodal.vlm.cache import CachingExtractor, ExtractionCache, cache_key, context_digest
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor
from mandol_competition.multimodal.vlm.prompt import build_extraction_prompt

META = dict(model="test", prompt_version="v1", schema_version="1")


def _uri(raw=b"ABC"):
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _media_id(raw=b"ABC"):
    return "img_" + hashlib.sha256(raw).hexdigest()


def _extract(extractor, *, image_data_uri=None, context_text="", image_position=0, image_count=1):
    return extractor.extract(
        image_data_uri=image_data_uri or _uri(), context_text=context_text,
        image_position=image_position, image_count=image_count,
    )


def test_parse_extraction_preserves_order_and_normalizes_missing_fields():
    result = parse_extraction({
        "description": "  red bag  ",
        "entities": [{"name": "Grace", "type": "person"}, {"name": "Zoe"}, {"type": "person"}],
        "objects": [{"name": "balloon", "count": 3}, {"name": "bag", "count": "bad"}],
        "events": [{"subject": "Grace", "action": "holds", "object": "bag"}],
        "relations": [{"source": "Grace", "predicate": "holds", "target": "bag"}],
        "tags": ["red", "blue", ""], "ocr_text": None,
    }, **META)
    assert result.description == "red bag"
    assert [entity.name for entity in result.entities] == ["Grace", "Zoe"]
    assert [obj.count for obj in result.objects] == [3, 1]
    assert result.ocr_text == "" and result.tags == ["red", "blue"]
    assert result.location_hint is None and not hasattr(result, "time_hint")
    assert result.model == "test" and result.prompt_version == "v1"


@pytest.mark.parametrize("payload", [[], "text", {"description": 42}, {"ocr_text": 42}])
def test_parse_extraction_rejects_unusable_payloads(payload):
    with pytest.raises(ExtractionError):
        parse_extraction(payload, **META)


def test_prompt_uses_context_only_for_naming_not_as_visual_evidence():
    prompt = build_extraction_prompt("D1:IMG_002\nGrace has a bag", 1, 3).lower()
    assert "d1:img_002\ngrace has a bag" in prompt
    assert "second" in prompt and "3" in prompt
    assert "only" in prompt and "visible" in prompt
    assert "do not" in prompt and "location" in prompt and "time" in prompt
    assert "time_hint" not in prompt
    assert "no surrounding text" in build_extraction_prompt("", 0, 1).lower()


def test_fake_uses_injected_mapping_without_network_and_tracks_calls():
    extractor = FakeVisionExtractor({_media_id(): {"description": "A red bag."}}, **META)
    assert isinstance(extractor, VisionExtractor)
    assert _extract(extractor).description == "A red bag."
    assert _extract(extractor, image_data_uri=_uri(b"other")).description == ""
    assert extractor.calls == 2 and extractor.model_name == "test"


def test_fake_strict_mode_rejects_missing_media():
    with pytest.raises(ExtractionError):
        _extract(FakeVisionExtractor({}, strict=True, **META))


def test_fake_fixture_rejects_gold_metadata_from_file_and_mapping(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({_media_id(): {"description": "x", "answer": "gold"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        FakeVisionExtractor.from_json_file(path, **META)
    with pytest.raises(ValueError, match="forbidden"):
        FakeVisionExtractor({_media_id(): {"description": "x", "evidence_ids": ["id"]}}, **META)
    path.write_text(json.dumps({_media_id(): {"description": "from file"}}), encoding="utf-8")
    assert _extract(FakeVisionExtractor.from_json_file(path, **META)).description == "from file"


def test_cache_key_includes_all_declared_components():
    original = cache_key("img_a", "context", "m", "p", "s")
    for changed in [
        ("img_b", "context", "m", "p", "s"), ("img_a", "other", "m", "p", "s"),
        ("img_a", "context", "m2", "p", "s"), ("img_a", "context", "m", "p2", "s"),
        ("img_a", "context", "m", "p", "s2"),
    ]:
        assert original != cache_key(*changed)
    assert context_digest("a") == context_digest("a") != context_digest("b")


def test_cache_roundtrip_sharding_and_per_media_delete(tmp_path):
    cache = ExtractionCache(tmp_path / "user-1")
    original = Extraction(description="hello", model="test")
    cache.put(_media_id(), "k1", original)
    cache.put(_media_id(b"other"), "k2", Extraction(description="other"))
    assert cache.get(_media_id(), "k1") == original
    assert next((tmp_path / "user-1").rglob("k1.json")).relative_to(tmp_path / "user-1").parts[:2] == (
        _media_id()[4:6], _media_id()[6:8]
    )
    assert len(cache) == 2 and cache.delete_for_media(_media_id()) == 1
    assert cache.get(_media_id(), "k1") is None
    assert cache.get(_media_id(b"other"), "k2") is not None


def test_cache_corruption_and_wrong_media_are_misses(tmp_path):
    cache = ExtractionCache(tmp_path)
    cache.put("img_abc", "k1", Extraction(description="hello"))
    path = next(tmp_path.rglob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["media_id"] = "img_different"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert cache.get("img_abc", "k1") is None
    path.write_text("{invalid", encoding="utf-8")
    assert cache.get("img_abc", "k1") is None


class _CountingExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, *, image_data_uri, context_text, image_position, image_count):
        self.calls += 1
        return Extraction(description=f"call {self.calls}", model="counter")


def test_cache_hits_only_with_identical_prompt_inputs(tmp_path):
    inner = _CountingExtractor()
    wrapper = CachingExtractor(inner, ExtractionCache(tmp_path), **META)
    assert _extract(wrapper).description == "call 1"
    assert _extract(wrapper).description == "call 1"
    assert wrapper.hits == 1 and wrapper.misses == 1
    for changed in [dict(context_text="else"), dict(image_position=1, image_count=2),
                    dict(image_position=0, image_count=2), dict(image_data_uri=_uri(b"other"))]:
        _extract(wrapper, **changed)
    assert inner.calls == 5 and wrapper.cache_size() == 5


def test_user_roots_do_not_share_cache_entries(tmp_path):
    first = CachingExtractor(_CountingExtractor(), ExtractionCache(tmp_path / "u1"), **META)
    second_inner = _CountingExtractor()
    second = CachingExtractor(second_inner, ExtractionCache(tmp_path / "u2"), **META)
    _extract(first)
    assert _extract(second).description == "call 1" and second_inner.calls == 1


def test_record_observation_does_not_affect_cache_key(tmp_path):
    inner = _CountingExtractor()
    cache = ExtractionCache(tmp_path)
    wrapper = CachingExtractor(inner, cache, **META)
    _extract(wrapper)
    wrapper.record_observation(_uri(), usage={"total_tokens": 10}, latency_ms=5.0)
    assert _extract(wrapper).description == "call 1"
    assert wrapper.hits == 1
    assert json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))["usage"] == {
        "total_tokens": 10
    }
