"""Strict inline-image validation before an Add request has side effects."""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
from dataclasses import dataclass
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from .config import MultimodalConfig
from .protocol import ContentPart, ImagePart
from .profiling import stage, timed


_MIMES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_DATA_URI = re.compile(r"data:([a-zA-Z0-9.+/-]+);base64,([^,]*)", re.DOTALL)
_decode_base64 = timed(base64.b64decode, "base64_decode")


class PayloadError(ValueError):
    def __init__(self, reason: str, status_code: int = 422) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class ExifInfo:
    timestamp: str | None = None
    gps: tuple[float, float] | None = None


@dataclass(frozen=True)
class DecodedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    exif: ExifInfo


def _extract_exif(image: Image.Image) -> ExifInfo:
    try:
        exif = image.getexif()
    except Exception:
        return ExifInfo()

    try:
        original = exif.get_ifd(34665).get(36867)
        raw_time = original or exif.get(36867) or exif.get(306)
        timestamp = raw_time.strip() if isinstance(raw_time, str) and raw_time.strip() else None
    except Exception:
        timestamp = None

    try:
        gps = exif.get_ifd(34853)
        latitude_ref, longitude_ref = gps[1], gps[3]
        if isinstance(latitude_ref, bytes):
            latitude_ref = latitude_ref.decode("ascii")
        if isinstance(longitude_ref, bytes):
            longitude_ref = longitude_ref.decode("ascii")
        if latitude_ref not in ("N", "S") or longitude_ref not in ("E", "W"):
            raise ValueError("invalid GPS direction")

        def degrees(values: tuple) -> float:
            degree, minute, second = (float(value) for value in values)
            return degree + minute / 60 + second / 3600

        latitude, longitude = degrees(gps[2]), degrees(gps[4])
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError("non-finite GPS coordinate")
        if not (0 <= latitude <= 90 and 0 <= longitude <= 180):
            raise ValueError("GPS coordinate out of range")
        coordinates = (
            round(latitude * (-1 if latitude_ref == "S" else 1), 6),
            round(longitude * (-1 if longitude_ref == "W" else 1), 6),
        )
    except Exception:
        coordinates = None
    return ExifInfo(timestamp, coordinates)


def decode_data_uri(url: str, config: MultimodalConfig) -> DecodedImage:
    """Decode one JPEG/PNG/WebP Data URI and check its actual image content."""
    match = _DATA_URI.fullmatch(url) if isinstance(url, str) else None
    if match is None:
        raise PayloadError("image_url.url must be an inline Base64 Data URI")
    declared = match[1].lower()
    if declared not in _MIMES.values():
        raise PayloadError(f"unsupported image MIME type {declared!r}")

    encoded = match[2]
    if len(encoded) > 4 * ((config.max_image_bytes + 2) // 3):
        raise PayloadError(f"image exceeds the {config.max_image_bytes} byte per-image limit")
    try:
        data = _decode_base64(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PayloadError(f"invalid Base64 image payload: {exc}") from exc
    if not data:
        raise PayloadError("image payload is empty")
    if len(data) > config.max_image_bytes:
        raise PayloadError(f"image exceeds the {config.max_image_bytes} byte per-image limit")
    return _validate_image_bytes(data, declared, config)


@stage("image_validation")
def _validate_image_bytes(data: bytes, declared: str, config: MultimodalConfig) -> DecodedImage:
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual = _MIMES.get(image.format or "")
            if actual != declared:
                raise PayloadError(
                    f"declared MIME {declared!r} does not match image format {actual!r}"
                )
            width, height = image.size
            if width * height > config.max_pixels:
                raise PayloadError(f"image exceeds the {config.max_pixels} pixel limit")
            image.load()
            exif = _extract_exif(image)
    except PayloadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PayloadError(f"payload is not a decodable image: {exc}") from exc
    return DecodedImage(data, actual, width, height, exif)


def decode_message_images(
    parts: list[ContentPart], config: MultimodalConfig
) -> dict[int, DecodedImage]:
    """Keep each image at its original part index for ordered reconstruction."""
    return {
        index: decode_data_uri(part.image_url.url, config)
        for index, part in enumerate(parts)
        if isinstance(part, ImagePart)
    }


def enforce_request_image_budget(
    images: Iterable[DecodedImage], config: MultimodalConfig
) -> None:
    """Call once with images from every message of an Add request."""
    total = sum(len(image.data) for image in images)
    if total > config.max_request_image_bytes:
        raise PayloadError(
            f"request exceeds the {config.max_request_image_bytes} byte decoded-image limit"
        )
