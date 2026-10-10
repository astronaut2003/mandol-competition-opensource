"""Versioned visual field ablations. Original ordered content is never rewritten."""
from dataclasses import dataclass, replace

from .render import RenderImage, _image_block, _one_line, render_text_content, render_query_text

BASELINE = "mm-representation-v1"
FIELDS = ("description", "entities", "objects", "ocr_text", "events", "relations", "tags")


@dataclass(frozen=True)
class RepresentationPolicy:
    fields: tuple[str, ...]
    context: bool = False
    temporal: bool = False
    location: bool = False
    surrounding: bool = False
    ocr_first: bool = False


POLICIES = {
    BASELINE: RepresentationPolicy(FIELDS, context=True, temporal=True),
    "mm-representation-r1-v1": RepresentationPolicy(("description",)),
    "mm-representation-r2-v1": RepresentationPolicy(("description", "ocr_text")),
    "mm-representation-r3-v1": RepresentationPolicy(("description", "entities", "objects")),
    "mm-representation-r4-v1": RepresentationPolicy(("description", "ocr_text"), temporal=True, location=True),
    "mm-representation-r5-v1": RepresentationPolicy(FIELDS, temporal=True, location=True),
    "mm-representation-r6-v1": RepresentationPolicy(FIELDS, context=True, temporal=True, location=True, surrounding=True),
    "mm-representation-sgm-v1": RepresentationPolicy(FIELDS[1:], temporal=True, location=True),
    "mm-representation-ocr-first-v1": RepresentationPolicy(FIELDS, temporal=True, location=True, ocr_first=True),
    # 2026-10 public-study candidates. Existing R0/R1-R6 identities stay frozen.
    "mm-representation-a4-v1": RepresentationPolicy(("description", "ocr_text", "entities"), location=True),
    "mm-representation-a5-v1": RepresentationPolicy(("description", "ocr_text", "entities", "objects", "events")),
}


# Single-field diagnostics complement the bundled R1-R6 comparisons.
for _field in ("entities", "relations", "tags"):
    POLICIES[f"mm-representation-no-{_field}-v1"] = replace(
        POLICIES["mm-representation-r5-v1"], fields=tuple(f for f in FIELDS if f != _field))
POLICIES["mm-representation-no-location-v1"] = replace(POLICIES["mm-representation-r5-v1"], location=False)
POLICIES["mm-representation-no-time-v1"] = replace(POLICIES["mm-representation-r5-v1"], temporal=False)


def image_lines(image, policy):
    empty = {field: "" if field in {"description", "ocr_text"} else []
             for field in FIELDS if field not in policy.fields}
    extraction = replace(image.extraction, **empty)
    result = _image_block(RenderImage(image.media_id, extraction, image.location if policy.location else None))
    if policy.location and extraction.location_hint:
        result.append("Observed scene: " + _one_line(extraction.location_hint))
    if policy.ocr_first:
        result = result[:1] + [s for s in result[1:] if s.startswith("OCR:")] + [
            s for s in result[1:] if not s.startswith("OCR:")]
    return result


def memory_text(*, policy_id=BASELINE, dialogue, timestamp_text, images, surrounding_text=""):
    policy = POLICIES[policy_id]
    if policy_id == BASELINE or not images:
        return render_text_content(dialogue=dialogue, timestamp_text=timestamp_text, images=images)
    lines = []
    if policy.context and dialogue:
        lines.append("[Dialogue] " + dialogue)
    if policy.temporal and timestamp_text:
        lines.append("[Timestamp] " + _one_line(timestamp_text))
    for image in images:
        lines.extend(image_lines(image, policy))
    if policy.surrounding and surrounding_text:
        lines.append("[Surrounding source dialogue] " + surrounding_text)
    return "\n".join(lines)


QUERY_POLICIES = {"mm-query-v1", "mm-query-description-v1", "mm-query-sgm-v1", "mm-query-dm-ocr-entities-v1"}


def query_text(*, policy_id="mm-query-v1", query_text, images, options=None, use_options=True):
    if policy_id not in QUERY_POLICIES:
        raise ValueError("unknown multimodal query policy")
    options = options if use_options else None
    if policy_id == "mm-query-v1":
        return render_query_text(query_text=query_text, images=images, options=options)
    fields = {"mm-query-description-v1": ("description",),
              "mm-query-sgm-v1": FIELDS[1:],
              "mm-query-dm-ocr-entities-v1": ("description", "ocr_text", "entities")}[policy_id]
    lines = ["[Question] " + query_text] if query_text else []
    for image in images:
        lines.extend(image_lines(image, RepresentationPolicy(fields)))
    if options:
        lines += ["[Options]", *options]
    return "\n".join(lines)
