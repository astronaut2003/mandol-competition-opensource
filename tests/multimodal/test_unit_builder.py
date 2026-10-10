from mandol_competition.multimodal.render import Location, RenderImage
from mandol_competition.multimodal.unit_builder import build_memory_unit, build_unit_uid
from mandol_competition.multimodal.vlm.base import Extraction


def build(**overrides):
    args = {
        "user_id": "alice",
        "request_id": "add-1",
        "session_id": "session-1",
        "message_index": 0,
        "role": "user",
        "timestamp_text": "2024-07-01",
        "dialogue": "A picture of Mimi",
        "content_parts": [
            {"type": "text", "text": "A picture of Mimi"},
            {"type": "image", "media_id": "img_123"},
        ],
        "images": [
            RenderImage(
                "img_123", Extraction(description="A cat on a sofa"), Location("1,2", "exif")
            )
        ],
        "extractor_model": "mock",
        "prompt_version": "v1",
        "schema_version": "1",
    }
    args.update(overrides)
    return build_memory_unit(**args)


def test_unit_is_stable_text_searchable_and_preserves_evidence():
    first = build()
    second = build()
    assert first.uid == second.uid
    assert first.raw_data == second.raw_data
    assert first.uid != build_unit_uid("alice", "add-1", 1)
    assert first.raw_data["original_content"] == "A picture of Mimi"
    assert "A cat on a sofa" in first.raw_data["text_content"]
    assert first.raw_data["content_parts"] == [
        {"type": "text", "text": "A picture of Mimi"},
        {"type": "image", "media_id": "img_123"},
    ]
    assert first.raw_data["multimodal_info"]["images"][0]["media_id"] == "img_123"
    assert first.metadata["modality"] == "text_image"
    assert first.metadata["session_id"] == "session-1"
    assert all(
        key not in first.raw_data
        for key in ("image_path", "image", "image_url", "media_path", "image_bytes")
    )


def test_text_and_image_only_units():
    text = build(images=[], content_parts=[{"type": "text", "text": "hello"}], dialogue="hello")
    image = build(dialogue="", content_parts=[{"type": "image", "media_id": "img_123"}])
    assert text.metadata["modality"] == "text"
    assert image.metadata["modality"] == "image"
    assert "[Dialogue]" not in image.raw_data["text_content"]
