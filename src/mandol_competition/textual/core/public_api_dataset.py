"""Source-only adapters for the later actual-API public quality baseline.

No benchmark labels, questions, answers or evidence flags enter these functions.
LongMemEval session IDs are hashed before Add because some public source IDs
contain answer markers. LoCoMo's two peers retain their explicit speaker names
in imported dialogue text; mapping the peers to API roles does not make either
one a model-generated assistant. These choices must be frozen before Q0.
"""

import re
from datetime import UTC, datetime

from mandol_substrate.state.models import Scope, digest
from .public_api_source import SourceTurn, prepare_session

IMPORT_POLICY = {
    "version": "public-api-dialogue-import-v1",
    "locomo_content": "source speaker name + ': ' + original source text",
    "locomo_roles": "source speaker_a=user; source speaker_b=assistant; two human peers",
    "longmemeval_content": "original native message content and role",
    "source_date": "session mention time, not inferred event time",
    "naive_timezone": "UTC by explicit local benchmark convention; source has no timezone",
    "search_context": "only actual API content; no evaluator-side source augmentation",
    "empty_sources": "explicit source-provenance exclusion; never fabricate content",
    "source_identifiers": "opaque session hashes on API; originals evaluator-only",
    "q1m": "unchanged; this import is only for new actual-API Q0 onward",
}


def session_timestamp(source_date, dataset):
    if not isinstance(source_date, str) or not source_date:
        raise ValueError("Missing source session date")
    formats = {
        "locomo": "%I:%M %p on %d %B, %Y",
        "longmemeval": "%Y/%m/%d (%a) %H:%M",
    }
    if dataset not in formats:
        raise ValueError("Unsupported public source format")
    parsed = datetime.strptime(source_date, formats[dataset]).replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def prepare_conversation(conversation, *, dataset, scope: Scope):
    """Accept only a source conversation, never the benchmark record or its QA.

    The returned source mapping is evaluator-side provenance and never an Add
    field. Original text hash and imported dialogue hash are both retained.
    Epoch timestamps represent source session/mention time with an explicit
    timezone convention, not the event time of every statement in the session.
    """
    allowed_metadata = {"speaker_a", "speaker_b"} if dataset == "locomo" else set()
    if dataset not in {"locomo", "longmemeval"}:
        raise ValueError("Unsupported public dataset")
    if any(
        not re.fullmatch(r"session_\d+(_date_time)?", key)
        and key not in allowed_metadata
        for key in conversation
    ):
        raise ValueError("Non-source fields in public conversation")
    sessions = sorted(
        (name for name in conversation if re.fullmatch(r"session_\d+", name)),
        key=lambda name: int(name.split("_")[1]),
    )
    if not sessions:
        raise ValueError("No source sessions")
    if dataset == "locomo":
        speakers = [conversation[key] for key in ("speaker_a", "speaker_b")]
        if (
            any(not isinstance(name, str) or not name.strip() for name in speakers)
            or len(set(speakers)) != 2
        ):
            raise ValueError("LoCoMo needs two explicit distinct source speakers")
        roles = dict(zip(speakers, ("user", "assistant"), strict=True))
    requests, provenance, skipped, source_ids = [], {}, [], set()
    session_manifest = []
    for name in sessions:
        date = conversation[name + "_date_time"]
        timestamp = session_timestamp(date, dataset)
        turns, original = [], {}
        for index, turn in enumerate(conversation[name]):
            allowed = {
                "dia_id",
                "text",
                "speaker",
                "img_url",
                "blip_caption",
                "query",
                "re-download",
            }
            if dataset == "longmemeval":
                allowed = {
                    "dia_id",
                    "text",
                    "speaker",
                    "session_id",
                    "session_occurrence",
                    "turn_index",
                }
            if not set(turn) <= allowed or not {"dia_id", "text", "speaker"} <= set(
                turn
            ):
                raise ValueError("Unexpected source-turn fields")
            uid, text, speaker = turn["dia_id"], turn["text"], turn["speaker"]
            if (
                not isinstance(uid, str)
                or not uid
                or uid in source_ids
                or not isinstance(text, str)
            ):
                raise ValueError("Invalid/duplicate source turn")
            source_ids.add(uid)
            if dataset == "locomo":
                if speaker not in roles:
                    raise ValueError("Unknown source speaker")
                role = roles[speaker]
                content = f"{speaker}: {text}" if text.strip() else text
            else:
                if speaker not in {"user", "assistant"}:
                    raise ValueError(
                        "Native role cannot enter the current public Add contract"
                    )
                role, content = speaker, text
            turns.append(SourceTurn(uid, role, content, timestamp))
            original[uid] = {
                "original_text_sha256": digest(text),
                "original_text_chars": len(text),
                "imported_text_sha256": digest(content),
                "source_speaker": speaker,
                "source_session_date": date,
                "source_turn_index": index,
                "source_session_index": int(name.split("_")[1]),
                "timestamp_semantics": "source session mention time; explicit naive UTC convention",
            }
        session_id = "source_session_" + digest([dataset, scope.key, name])
        prepared = prepare_session(scope, session_id, turns)
        requests.extend(prepared["requests"])
        for uid, mapping in prepared["uid_provenance"].items():
            if uid in provenance:
                raise ValueError("Duplicate canonical import UID")
            provenance[uid] = {**mapping, **original[mapping["source_uid"]]}
        skipped.extend(prepared["skipped_sources"])
        session_manifest.append(
            {
                "source_session": name,
                "api_session_id": session_id,
                "source_manifest_sha256": prepared["source_manifest_sha256"],
                "source_turns": prepared["source_turns"],
                "canonical_units": prepared["canonical_units"],
                "requests": len(prepared["requests"]),
            }
        )
    return {
        "schema": "public-api-conversation-import-v1",
        "dataset": dataset,
        "scope": {"namespace": scope.namespace, "user_id": scope.user_id},
        "import_policy": IMPORT_POLICY,
        "import_policy_sha256": digest(IMPORT_POLICY),
        "source_conversation_sha256": digest(conversation),
        "requests": requests,
        "uid_provenance": provenance,
        "skipped_sources": skipped,
        "sessions": session_manifest,
        "source_turns": len(source_ids),
        "canonical_units": len(provenance),
        "multimodal_policy": "Textual source only; images/captions are not synthesized into dialogue",
    }
