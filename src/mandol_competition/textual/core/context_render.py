"""Experimental candidate-preserving context rendering, separate from T0.

No generation, retrieval, state writes or benchmark labels. Source mention time
is explicitly distinct from event time and ingestion-created_at. The caller
provides only same-scope canonical hydration from its held publication view.
"""

from datetime import UTC, datetime

from mandol_substrate.state.models import StateUnavailable, digest


def render_context_items(scope, items, canonical, *, mode="raw", final_k=10):
    if mode not in {"raw", "source_mention", "source_mention_chronological"}:
        raise ValueError("Unknown versioned context representation")
    if type(final_k) is not int or not 1 <= final_k <= 100 or len(items) > 1024:
        raise ValueError("Invalid bounded context window")
    ids = [r["id"] for r in items]
    if len(ids) != len(set(ids)) or set(ids) != set(canonical):
        raise StateUnavailable("Context hydration candidate IDs differ")
    for item in items:
        memory = canonical[item["id"]]
        # Prototype supports only actual raw AML observations. Derived memories
        # need their own proven provenance path before entering this renderer.
        if memory.metadata.get("user_id") != scope.user_id or not memory.metadata.get(
            "aml"
        ):
            raise StateUnavailable("Context raw source lacks exact-user provenance")
        if (
            not memory.active
            or memory.lifecycle == "deleted"
            or memory.memory_level != "L0_Observation"
        ):
            raise StateUnavailable("Unsupported invisible/derived context memory")
        if item["content"] != memory.text_content:
            raise StateUnavailable("Input content differs from canonical raw text")
    selected = list(items[:final_k])
    if mode == "source_mention_chronological":
        # Reorder only the selected K; never bring a lower-ranked candidate into
        # the context by sorting all returned100. Unknown dates retain rank.
        indexed = list(enumerate(selected))
        indexed.sort(
            key=lambda pair: (
                canonical[pair[1]["id"]].business_timestamp_ms is None,
                canonical[pair[1]["id"]].business_timestamp_ms or 0,
                pair[0],
            )
        )
        selected = [row for _, row in indexed]
    output = []
    for item in selected + list(items[final_k:]):
        memory = canonical[item["id"]]
        content = memory.text_content
        if mode != "raw":
            fields = []
            if memory.business_timestamp_ms is not None:
                value = (
                    datetime.fromtimestamp(memory.business_timestamp_ms / 1000, UTC)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                fields.append("Source mention time: " + value)
            if memory.role in {"user", "assistant"}:
                fields.append("Source role: " + memory.role)
            if fields:
                content = "[" + "; ".join(fields) + "]\n" + content
        output.append({**item, "content": content})
    if {r["id"] for r in output[:final_k]} != set(ids[:final_k]) or [
        r["id"] for r in output[final_k:]
    ] != ids[final_k:]:
        raise StateUnavailable(
            "Context-only representation changed candidate selection"
        )
    return output, {
        "mode": mode,
        "schema": "mandol-raw-source-context-hypothesis-v1",
        "final_k": final_k,
        "input_ids": ids,
        "output_ids": [r["id"] for r in output],
        "source_hashes": {uid: digest(canonical[uid].text_content) for uid in ids},
        "source_text_preserved": True,
        "date_semantics": "MENTION_NOT_EVENT_OR_INGESTION",
        "input_chars": sum(len(r["content"]) for r in items[:final_k]),
        "output_chars": sum(len(r["content"]) for r in output[:final_k]),
        "context_budget_gate": "MUST_REMEASURE_ACTUAL_WIRE_CONTEXT_AND_QA",
    }
