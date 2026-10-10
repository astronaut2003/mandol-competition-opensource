"""Evaluator-only public question/provenance binding, isolated from Add inputs."""

from mandol_substrate.state.models import digest
from .reranker_quality import _evidence


def bind_questions(record, prepared, dataset):
    if dataset not in {"locomo", "longmemeval"}:
        raise ValueError("Unsupported public question dataset")
    if digest(record["conversation"]) != prepared["source_conversation_sha256"]:
        raise ValueError("Questions and prepared source history differ")
    sources = {}
    for name, turns in record["conversation"].items():
        if name.startswith("session_") and isinstance(turns, list):
            for turn in turns:
                uid = turn["dia_id"]
                if uid in sources:
                    raise ValueError("Duplicate source identity")
                sources[uid] = turn.get("session_id", name)
    provenance = prepared["uid_provenance"]
    if any(row["source_uid"] not in sources for row in provenance.values()):
        raise ValueError("Canonical provenance is not part of this source history")
    queries, labels = [], []
    for index, original in enumerate(record["qa"]):
        qa = {**original, "dataset": dataset}
        if original.get("dataset", dataset) != dataset:
            raise ValueError("Question dataset differs")
        if not isinstance(qa["question"], str) or not qa["question"].strip():
            raise ValueError("Empty public question")
        evidence = _evidence(qa)
        queries.append({"query_index": index, "query": qa["question"]})
        labels.append(
            {
                "query_index": index,
                "qa": qa,
                "evidence_source_uids": evidence,
                "gold_source_ids_absent_from_history": sorted(
                    set(evidence) - set(sources)
                ),
                "gold_source_ids_not_imported": sorted(
                    set(evidence) - {row["source_uid"] for row in provenance.values()}
                ),
            }
        )
    if not queries:
        raise ValueError("Public history has no questions")
    return {
        "schema": "public-api-question-binding-v1",
        "dataset": dataset,
        "scope": prepared["scope"],
        "source_conversation_sha256": prepared["source_conversation_sha256"],
        "provenance_sha256": digest(provenance),
        "api_searches": queries,
        "evaluator_labels": labels,
        "canonical_uid_to_source_session": {
            uid: sources[row["source_uid"]] for uid, row in provenance.items()
        },
        "data_boundary": "Only api_searches query/index enter Search; labels/session mapping never enter Add or rerank",
    }
