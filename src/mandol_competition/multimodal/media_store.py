"""Per-user content-addressed storage of unchanged image bytes."""

from __future__ import annotations

from .profiling import stage

import base64
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import MultimodalConfig
from .validation import DecodedImage


_MEDIA_ID = re.compile(r"img_[0-9a-f]{64}")
_MIMES = {"image/jpeg", "image/png", "image/webp"}


def compute_media_id(data: bytes) -> str:
    return "img_" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class MediaRecord:
    media_id: str
    mime_type: str
    byte_size: int
    sha256: str
    storage_path: Path
    created_at: str


def _write_atomic(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=".media-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        parent = path.parent
        while True:
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            if parent == parent.parent:
                break
            parent = parent.parent
    finally:
        Path(name).unlink(missing_ok=True)


class MediaStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path_for(self, media_id: str) -> Path | None:
        if not isinstance(media_id, str) or _MEDIA_ID.fullmatch(media_id) is None:
            return None
        digest = media_id[4:]
        path = self.root / digest[:2] / digest[2:4] / f"{digest}.bin"
        return path if path.is_file() else None

    @stage("media_write")
    def put(self, image: DecodedImage) -> MediaRecord:
        """Store validated original bytes; deduplicate within this store only."""
        media_id = compute_media_id(image.data)
        digest = media_id[4:]
        path = self.root / digest[:2] / digest[2:4] / f"{digest}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        mime_path = path.with_suffix(".mime")
        if not mime_path.is_file() or mime_path.read_text(encoding="utf-8") != image.mime_type:
            _write_atomic(mime_path, image.mime_type.encode("ascii"))
        if not path.is_file() or path.read_bytes() != image.data:
            _write_atomic(path, image.data)
        return MediaRecord(
            media_id=media_id,
            mime_type=image.mime_type,
            byte_size=len(image.data),
            sha256=digest,
            storage_path=path,
            created_at=datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        )

    @stage("media_restore")
    def get(self, media_id: str) -> tuple[bytes, str] | None:
        path = self.path_for(media_id)
        if path is None:
            return None
        try:
            mime = path.with_suffix(".mime").read_text(encoding="utf-8")
            if mime not in _MIMES:
                return None
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        if compute_media_id(data) != media_id:
            return None
        return data, mime

    def to_data_uri(self, media_id: str) -> str | None:
        loaded = self.get(media_id)
        if loaded is None:
            return None
        data, mime = loaded
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


class UserMediaStore(MediaStore):
    def __init__(self, config: MultimodalConfig, user_id: str) -> None:
        super().__init__(config.user_root(user_id) / "media")
        self.config = config
        self.user_id = user_id
