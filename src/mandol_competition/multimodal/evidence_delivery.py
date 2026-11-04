"""Factual, bounded response-delivery notes; no answer or relevance judgment.

Canonical memories and the atomic evidence gate are unchanged. A note is only
emitted when the wire response omitted an original image or the source already
records degraded extraction. Normal evidence receives no extra text.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json

POLICY = "source-grounded-delivery-v3"
SCHEMA = "search-image-delivery-v1"
MAX_NOTE_BYTES = 8 * 1024
HEADER = "[Search delivery metadata]\n"
_DELIVERY_PRIORITY = {"attached_here": 2, "shared_in_response": 1}
_OMISSION_REASONS = {"image_count_budget", "single_image_byte_budget",
                     "response_image_byte_budget", "media_unavailable"}


@dataclass
class SourceDelivery:
    memory_id: str
    recorded_extraction_degraded: bool
    images: dict[str, str] = field(default_factory=dict)

    def record(self, media_id: str, state: str) -> None:
        if state not in _DELIVERY_PRIORITY and state not in _OMISSION_REASONS:
            raise ValueError("Unknown image delivery state")
        previous = self.images.get(media_id)
        # Repeated parts can refer to the image attached earlier in this same
        # memory. Such a reference must not erase the actual attachment.
        if previous is None or _DELIVERY_PRIORITY.get(state, 0) > _DELIVERY_PRIORITY.get(previous, 0):
            self.images[media_id] = state

    def entry(self) -> dict | None:
        omitted = Counter(state for state in self.images.values() if state in _OMISSION_REASONS)
        if not omitted and not self.recorded_extraction_degraded:
            return None
        return dict(memory_id=self.memory_id, source_image_count=len(self.images),
                    attached_here=sum(state == "attached_here" for state in self.images.values()),
                    shared_in_response=sum(state == "shared_in_response" for state in self.images.values()),
                    not_attached=dict(sorted(omitted.items())),
                    recorded_extraction_degraded=self.recorded_extraction_degraded)


class ResponseDelivery:
    """Request-local state, updated at the actual image admission decisions."""

    def __init__(self) -> None:
        self.sources: list[SourceDelivery] = []

    def source(self, unit) -> SourceDelivery:
        result = SourceDelivery(unit.uid, unit.metadata.get("vlm_degraded") is True)
        self.sources.append(result)
        return result

    def render(self) -> str | None:
        entries = [entry for source in self.sources if (entry := source.entry()) is not None]
        if not entries:
            return None
        payload = dict(schema=SCHEMA, scope="this_search_response",
                       visual_description_origin="stored_model_extraction",
                       affected_sources=len(entries),
                       omitted_image_references=sum(sum(entry["not_attached"].values()) for entry in entries),
                       recorded_degraded_sources=sum(entry["recorded_extraction_degraded"] for entry in entries),
                       details_limited=False, sources=entries)

        def encode() -> str:
            # Source ids are data, even for an alternate backend. JSON quoting
            # prevents a value from adding a header or an unescaped instruction.
            return HEADER + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        rendered = encode()
        if len(rendered.encode("utf-8")) <= MAX_NOTE_BYTES:
            return rendered
        # Counts always describe all affected sources. Bounded per-source
        # details are explicitly incomplete; the underlying evidence is intact.
        payload["details_limited"] = True
        low, high = 0, len(entries)
        while low < high:
            middle = (low + high + 1) // 2
            payload["sources"] = entries[:middle]
            if len(encode().encode("utf-8")) <= MAX_NOTE_BYTES:
                low = middle
            else:
                high = middle - 1
        payload["sources"] = entries[:low]
        return encode()

    def attach(self, results: list[dict]) -> None:
        if results and (note := self.render()) is not None:
            # Keep the existing source header first. No invented memory id or
            # synthetic result consumes a top_k slot; scores/order stay intact.
            results[0]["content"].insert(1, {"type": "text", "text": note})
