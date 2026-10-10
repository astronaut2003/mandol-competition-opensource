import base64
import hashlib
import io

from PIL import Image

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.media_store import MediaStore, UserMediaStore, compute_media_id
from mandol_competition.multimodal.protocol import ImagePart, TextPart
from mandol_competition.multimodal.validation import DecodedImage, ExifInfo, decode_message_images


def image_bytes(format_name="JPEG", color=(10, 20, 30)):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format=format_name)
    return buffer.getvalue()


def decoded(raw, mime="image/jpeg"):
    return DecodedImage(raw, mime, 4, 4, ExifInfo())


def test_hash_named_sharded_storage_and_metadata(tmp_path):
    original = image_bytes()
    store = MediaStore(tmp_path)
    record = store.put(decoded(original))
    digest = hashlib.sha256(original).hexdigest()
    assert record.media_id == compute_media_id(original) == f"img_{digest}"
    assert record.sha256 == digest
    assert record.storage_path == tmp_path / digest[:2] / digest[2:4] / f"{digest}.bin"
    assert record.storage_path.read_bytes() == original
    assert record.byte_size == len(original)
    assert record.mime_type == "image/jpeg"
    assert record.created_at.endswith("Z")
    assert store.path_for(record.media_id) == record.storage_path


def test_duplicate_bytes_deduplicate_within_user(tmp_path):
    store = MediaStore(tmp_path)
    original = image_bytes()
    first = store.put(decoded(original))
    second = store.put(decoded(original))
    assert first.media_id == second.media_id
    assert list(tmp_path.rglob("*.bin")) == [first.storage_path]
    assert not list(tmp_path.rglob("*.tmp"))
    assert compute_media_id(image_bytes(color=(80, 90, 100))) != first.media_id


def test_get_and_data_uri_recover_original_format_and_bytes(tmp_path):
    store = MediaStore(tmp_path)
    formats = [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")]
    for format_name, mime in formats:
        original = image_bytes(format_name)
        record = store.put(decoded(original, mime))
        assert store.get(record.media_id) == (original, mime)
        data_uri = store.to_data_uri(record.media_id)
        assert data_uri.startswith(f"data:{mime};base64,")
        assert base64.b64decode(data_uri.split(",", 1)[1]) == original


def test_missing_invalid_or_tampered_media_does_not_reconstruct(tmp_path):
    store = MediaStore(tmp_path)
    for media_id in ["img_" + "0" * 64, "img_../escape", "img_" + "G" * 64, "other"]:
        assert store.path_for(media_id) is None
        assert store.get(media_id) is None
        assert store.to_data_uri(media_id) is None

    record = store.put(decoded(image_bytes()))
    record.storage_path.with_suffix(".mime").unlink()
    assert store.get(record.media_id) is None
    store.put(decoded(image_bytes()))
    record.storage_path.write_bytes(b"corrupt")
    assert store.to_data_uri(record.media_id) is None


def test_user_directories_isolate_same_bytes_even_with_similar_ids(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path)
    original = image_bytes()
    first = UserMediaStore(config, "a/b")
    second = UserMediaStore(config, "a:b")
    first_record = first.put(decoded(original))
    second_record = second.put(decoded(original))
    assert first.root == config.user_root("a/b") / "media"
    assert second.root == config.user_root("a:b") / "media"
    assert first.root != second.root
    assert first_record.media_id == second_record.media_id
    assert first_record.storage_path != second_record.storage_path
    assert first_record.storage_path.read_bytes() == second_record.storage_path.read_bytes()
    first_record.storage_path.unlink()
    assert second.get(second_record.media_id) == (original, "image/jpeg")


def test_reopen_store_reconstructs_images_in_part_order(tmp_path):
    original_png = image_bytes("PNG", (1, 2, 3))
    original_jpeg = image_bytes("JPEG", (4, 5, 6))
    parts = [
        TextPart(type="text", text="first"),
        ImagePart(
            type="image_url",
            image_url={"url": "data:image/png;base64," + base64.b64encode(original_png).decode()},
        ),
        TextPart(type="text", text="second"),
        ImagePart(
            type="image_url",
            image_url={"url": "data:image/jpeg;base64," + base64.b64encode(original_jpeg).decode()},
        ),
    ]
    images = decode_message_images(parts, MultimodalConfig())
    store = UserMediaStore(MultimodalConfig(data_dir=tmp_path), "u1")
    media_ids = {index: store.put(image).media_id for index, image in images.items()}
    reopened = UserMediaStore(MultimodalConfig(data_dir=tmp_path), "u1")
    reconstructed = [
        part.text if isinstance(part, TextPart) else reopened.to_data_uri(media_ids[index])
        for index, part in enumerate(parts)
    ]
    assert reconstructed[0] == "first"
    assert reconstructed[2] == "second"
    assert [
        base64.b64decode(value.split(",", 1)[1])
        for value in (reconstructed[1], reconstructed[3])
    ] == [original_png, original_jpeg]
