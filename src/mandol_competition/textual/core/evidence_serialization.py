"""Opt-in source metadata serialization; never changes canonical text or retrieval.

Callers supply the canonical snapshot held by their publication lease. Speaker
names require an immutable, version-bound source binding, not model inference.
This module is independent of the byte-frozen Competition context renderer.
"""

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from mandol_substrate.state.models import digest


def renderer_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceEvidence:
    uid: str
    scope: tuple[str, str]
    version: int
    raw: str
    raw_sha256: str
    role: str | None = None
    source_mention_ms: int | None = None
    session_id: str | None = None
    speaker: str | None = None
    speaker_binding_sha256: str | None = None
    active: bool = True

    def validate(self, scope, version):
        if self.scope != tuple(scope) or not self.active or self.version != version:
            raise ValueError("Inactive, stale or cross-scope source")
        if self.version < 1 or text_sha256(self.raw) != self.raw_sha256:
            raise ValueError("Canonical content binding mismatch")
        if self.speaker and not self.speaker_binding_sha256:
            raise ValueError("Speaker requires immutable source binding")
        for label in (self.role, self.session_id, self.speaker):
            if label is not None and (not isinstance(label, str) or len(label) > 256 or any(c in label for c in "\r\n[]")):
                raise ValueError("Invalid single-line metadata label")


@dataclass(frozen=True)
class EvidenceSerializationPolicy:
    renderer_sha256: str
    source_binding_sha256: str
    mode: str = "raw"
    selected_k: int = 100
    max_chars: int = 24000
    ordering: str = "rank"
    schema: str = "mandol-evidence-serialization-v1"

    def __post_init__(self):
        if self.schema != "mandol-evidence-serialization-v1" or self.mode not in {
            "raw", "source_time", "source_role", "source_time_role",
            "source_time_speaker", "source_time_speaker_session",
        }:
            raise ValueError("Unknown evidence serialization policy")
        if type(self.selected_k) is not int or not 1 <= self.selected_k <= 100:
            raise ValueError("Invalid selected K")
        if type(self.max_chars) is not int or self.max_chars < 1:
            raise ValueError("Invalid character budget")
        if self.ordering not in {"rank", "selected_chronological"}:
            raise ValueError("Unknown ordering")
        for value in (self.renderer_sha256, self.source_binding_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Explicit SHA256 binding required")

    @property
    def hash(self):
        return digest(asdict(self))

    def render(self, hits, sources: Mapping[str, SourceEvidence], *, scope, versions, source_binding_sha256):
        """Return selected hits with content only changed; no score recalculation.

        Ordering is optionally applied *after* rank-prefix selection. Unknown
        mention times stay unknown and are placed last in chronological mode.
        created_at is deliberately not part of this interface.
        """
        if self.renderer_sha256 != renderer_sha256() or self.source_binding_sha256 != source_binding_sha256:
            raise ValueError("Renderer or source snapshot changed")
        selected = list(hits[:self.selected_k])
        if len({h["id"] for h in selected}) != len(selected):
            raise ValueError("Duplicate candidate UID")
        for hit in selected:
            src = sources[hit["id"]]
            if src.uid != hit["id"]:
                raise ValueError("Wrong source UID")
            src.validate(scope, versions[hit["id"]])
            if src.speaker and src.speaker_binding_sha256 != source_binding_sha256:
                raise ValueError("Speaker snapshot mismatch")
        if self.ordering == "selected_chronological":
            selected.sort(key=lambda h: (sources[h["id"]].source_mention_ms is None,
                                        sources[h["id"]].source_mention_ms or 0))
        output = []
        for hit in selected:
            src = sources[hit["id"]]
            labels = []
            if "time" in self.mode and src.source_mention_ms is not None:
                stamp = datetime.fromtimestamp(src.source_mention_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
                labels.append("Source mention time: " + stamp)
            if self.mode in {"source_role", "source_time_role"} and src.role in {"user", "assistant"}:
                labels.append("Source role: " + src.role)
            if "speaker" in self.mode:
                if src.speaker:
                    labels.append("Speaker: " + src.speaker)
                elif src.role in {"user", "assistant"}:
                    labels.append("Source role: " + src.role)
            if self.mode.endswith("session") and src.session_id:
                labels.append("Session: " + src.session_id)
            prefix = "[" + "; ".join(labels) + "]\n" if labels else ""
            output.append({**hit, "content": prefix + src.raw})
        return output

    def pack(self, rendered):
        """Explicit Answer-side prefix budget, with partial-record accounting.

        Search hits retain entire Raw. Only this downstream context can end in
        a partial record, which is reported rather than silently counted whole.
        """
        pieces, audit, used = [], [], 0
        for rank, hit in enumerate(rendered[:self.selected_k], 1):
            separator = 2 if pieces else 0
            available = self.max_chars - used - separator
            if available <= 0:
                break
            content = hit["content"][:available]
            pieces.append(content)
            used += separator + len(content)
            audit.append({"id": hit["id"], "selected_rank": rank,
                          "visible_chars": len(content), "record_chars": len(hit["content"]),
                          "fully_visible": len(content) == len(hit["content"])})
        return "\n\n".join(pieces), {"records": audit, "chars": used,
            "fully_visible": sum(x["fully_visible"] for x in audit),
            "partial_records": sum(not x["fully_visible"] for x in audit),
            "records_absent": len(rendered[:self.selected_k]) - len(audit)}
