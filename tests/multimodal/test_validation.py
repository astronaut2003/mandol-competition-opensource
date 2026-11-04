import base64
import io

import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.protocol import ImagePart, TextPart
from mandol_competition.multimodal.validation import (
    PayloadError,
    decode_data_uri,
    decode_message_images,
    enforce_request_image_budget,
)


def image_bytes(format_name="PNG", *, size=(4, 4)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format=format_name)
    return buffer.getvalue()


def uri(data, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@pytest.mark.parametrize(
    ("format_name", "mime"),
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")],
)
def test_decodes_supported_formats_without_changing_bytes(format_name, mime):
    original = image_bytes(format_name)
    decoded = decode_data_uri(uri(original, mime), MultimodalConfig())
    assert decoded.data == original
    assert decoded.mime_type == mime
    assert (decoded.width, decoded.height) == (4, 4)
    assert decoded.exif.timestamp is None
    assert decoded.exif.gps is None


@pytest.mark.parametrize(
    "bad_uri",
    [
        "https://example.com/a.png",
        "data:image/png,AAAA",
        "data:image/png;base64,",
        "data:image/png;base64,AAAA\n",
        "data:image/png;base64,!!!!",
        "data:image/png;base64,AA=A",
        "data:image/png;base64,AAAA,AAAA",
        None,
    ],
)
def test_rejects_non_strict_data_uri(bad_uri):
    with pytest.raises(PayloadError) as excinfo:
        decode_data_uri(bad_uri, MultimodalConfig())
    assert excinfo.value.status_code == 422
    assert excinfo.value.reason


def test_rejects_unsupported_mime():
    with pytest.raises(PayloadError, match="image/gif"):
        decode_data_uri(uri(image_bytes(), "image/gif"), MultimodalConfig())


def test_rejects_mime_mismatch():
    with pytest.raises(PayloadError, match="does not match"):
        decode_data_uri(uri(image_bytes(), "image/jpeg"), MultimodalConfig())


@pytest.mark.parametrize("data", [b"not an image", image_bytes()[:35]])
def test_rejects_invalid_or_truncated_image(data):
    with pytest.raises(PayloadError):
        decode_data_uri(uri(data), MultimodalConfig())


def test_decoded_size_limit_is_inclusive():
    original = image_bytes()
    assert MultimodalConfig().max_image_bytes == 10 * 1024 * 1024
    assert (
        decode_data_uri(uri(original), MultimodalConfig(max_image_bytes=len(original))).data
        == original
    )
    with pytest.raises(PayloadError, match="byte per-image limit"):
        decode_data_uri(uri(original), MultimodalConfig(max_image_bytes=len(original) - 1))


def test_rejects_pixel_bomb_before_loading():
    with pytest.raises(PayloadError, match="pixel limit"):
        decode_data_uri(uri(image_bytes(size=(100, 100))), MultimodalConfig(max_pixels=100))


def test_pillow_bomb_guard_is_a_payload_error_before_pixel_decode(monkeypatch):
    # A tiny, valid image and a deliberately small Pillow guard exercise its
    # real open-time exception without allocating a decompression-bomb image.
    data = image_bytes(size=(20, 20))
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(PayloadError) as error:
        decode_data_uri(uri(data), MultimodalConfig())
    assert error.value.status_code == 422


def test_exif_timestamp_survives_and_missing_gps_is_none():
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[306] = "2024:09:01 12:34:56"
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG", exif=exif)
    decoded = decode_data_uri(uri(buffer.getvalue(), "image/jpeg"), MultimodalConfig())
    assert decoded.exif.timestamp == "2024:09:01 12:34:56"
    assert decoded.exif.gps is None


def test_exif_gps_is_extracted_without_changing_original_bytes():
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[34853] = {
        1: "S",
        2: (IFDRational(50, 1), IFDRational(54, 1), IFDRational(21, 1)),
        3: "W",
        4: (IFDRational(1, 1), IFDRational(25, 1), IFDRational(5, 1)),
    }
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG", exif=exif)
    original = buffer.getvalue()
    decoded = decode_data_uri(uri(original, "image/jpeg"), MultimodalConfig())
    assert decoded.data == original
    assert decoded.exif.gps == (-50.905833, -1.418056)


def test_exif_read_errors_do_not_reject_valid_image(monkeypatch):
    def broken_exif(self):
        raise ValueError("broken EXIF")

    monkeypatch.setattr(Image.Image, "getexif", broken_exif)
    decoded = decode_data_uri(uri(image_bytes()), MultimodalConfig())
    assert decoded.exif.timestamp is None
    assert decoded.exif.gps is None


def test_image_positions_preserve_part_order():
    parts = [
        TextPart(type="text", text="before"),
        ImagePart(type="image_url", image_url={"url": uri(image_bytes())}),
        TextPart(type="text", text="between"),
        ImagePart(type="image_url", image_url={"url": uri(image_bytes("JPEG"), "image/jpeg")}),
    ]
    decoded = decode_message_images(parts, MultimodalConfig())
    assert list(decoded) == [1, 3]
    assert [image.mime_type for image in decoded.values()] == ["image/png", "image/jpeg"]
    assert decode_message_images(parts[:1], MultimodalConfig()) == {}


def test_budget_is_across_all_messages_not_each_message():
    original = image_bytes()
    limit = len(original) * 2 - 1
    config = MultimodalConfig(max_request_image_bytes=limit, image_byte_budget=limit)
    first = decode_message_images(
        [ImagePart(type="image_url", image_url={"url": uri(original)})], config
    )
    second = decode_message_images(
        [ImagePart(type="image_url", image_url={"url": uri(original)})], config
    )
    assert MultimodalConfig().max_request_image_bytes == 30 * 1024 * 1024
    enforce_request_image_budget(first.values(), config)
    enforce_request_image_budget(second.values(), config)
    with pytest.raises(PayloadError, match="decoded-image limit"):
        enforce_request_image_budget((*first.values(), *second.values()), config)


def test_budget_accepts_exact_limit():
    original = image_bytes()
    limit = len(original) * 2
    config = MultimodalConfig(max_request_image_bytes=limit, image_byte_budget=limit)
    decoded = decode_data_uri(uri(original), config)
    enforce_request_image_budget(iter([decoded, decoded]), config)
