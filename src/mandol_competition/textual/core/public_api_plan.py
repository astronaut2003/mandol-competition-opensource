"""Immutable development-only inputs for an actual API baseline runner.

This planner cannot select a reranker or admit validation/holdout data. Source
dialogue and evaluator labels remain separate files, checked one history at a
time. Each harness API instance owns one SQLite database and its corresponding
vector scope; lexical state stays inside that same canonical database.
"""

import json
from pathlib import Path

from mandol_substrate.state.models import digest
from .paired_quality import file_sha256
from .reranker_quality import LONGMEMEVAL_SOURCE_SHA256, SOURCE_SHA256


def checked_json(binding, hash_key="sha256"):
    path = Path(binding["file"])
    if file_sha256(path) != binding[hash_key]:
        raise ValueError("Frozen public API input changed")
    return json.loads(path.read_text())


def history_inputs(source, questions):
    prepared = checked_json(source, "file_sha256")
    bound = checked_json(questions)
    if (
        source["source_record_id"] != questions["source_record_id"]
        or source["user_id"] != questions["user_id"]
        or prepared["schema"] != "public-api-conversation-import-v1"
        or bound["schema"] != "public-api-question-binding-v1"
        or prepared["scope"] != bound["scope"]
        or prepared["scope"]["user_id"] != source["user_id"]
        or prepared["dataset"] != bound["dataset"]
        or prepared["source_conversation_sha256"] != bound["source_conversation_sha256"]
        or digest(prepared["uid_provenance"]) != bound["provenance_sha256"]
        or len(bound["api_searches"]) != questions["questions"]
    ):
        raise ValueError("Source/question/scope binding differs")
    indices = [row["query_index"] for row in bound["api_searches"]]
    labels = [row["query_index"] for row in bound["evaluator_labels"]]
    if len(set(indices)) != len(indices) or indices != labels:
        raise ValueError("Public question coverage differs")
    if any(
        request["user_id"] != source["user_id"]
        or set(request) != {"user_id", "session_id", "request_id", "messages"}
        for request in prepared["requests"]
    ):
        raise ValueError("Non-source data or wrong user in Add input")
    for query, label in zip(
        bound["api_searches"], bound["evaluator_labels"], strict=True
    ):
        if (
            set(query) != {"query_index", "query"}
            or query["query"] != label["qa"]["question"]
        ):
            raise ValueError("Search query differs from its public question")
    return prepared, bound


def development_inputs(source_manifest, question_manifest):
    """Require the already frozen whole public development cohort."""
    source_manifest, question_manifest = map(Path, (source_manifest, question_manifest))
    sources = json.loads(source_manifest.read_text())
    queries = json.loads(question_manifest.read_text())
    dataset = sources["dataset"]
    if (
        sources["schema"] != "mandol-public-source-import-manifest-v1"
        or queries["schema"] != "public-api-development-question-manifest-v1"
        or queries["status"] != "PASS"
        or sources["split"] != queries["split"]
        or sources["split"] != "development"
        or dataset != queries["dataset"]
        or Path(queries["source_manifest"]).resolve() != source_manifest.resolve()
        or queries["source_manifest_sha256"] != file_sha256(source_manifest)
    ):
        raise ValueError("Only bound public development manifests are accepted")
    expected = {
        "locomo": (SOURCE_SHA256, 5, 999),
        "longmemeval": (LONGMEMEVAL_SOURCE_SHA256, 500, 500),
    }
    if dataset not in expected:
        raise ValueError("Unapproved baseline dataset")
    source_hash, histories, questions = expected[dataset]
    source_rows = {row["source_record_id"]: row for row in sources["records"]}
    query_rows = {row["source_record_id"]: row for row in queries["records"]}
    if (
        sources["source_sha256"] != source_hash
        or len(source_rows) != len(sources["records"])
        or len(query_rows) != len(queries["records"])
        or len(source_rows) != histories
        or set(source_rows) != set(query_rows)
        or queries["histories"] != histories
        or queries["questions"] != questions
        or sum(row["questions"] for row in query_rows.values()) != questions
        or len({row["user_id"] for row in source_rows.values()}) != histories
    ):
        raise ValueError("Frozen development coverage/hash differs")
    if dataset == "locomo" and set(source_rows) != {
        "conv-26",
        "conv-30",
        "conv-41",
        "conv-42",
        "conv-43",
    }:
        raise ValueError("LoCoMo validation/holdout cannot enter Q0 development")
    return {
        "dataset": dataset,
        "split": "development",
        "namespace": sources["namespace"],
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest),
        "question_manifest": str(question_manifest.resolve()),
        "question_manifest_sha256": file_sha256(question_manifest),
        "source_sha256": source_hash,
        "histories": histories,
        "questions": questions,
        "records": [
            {"source": source_rows[key], "questions": query_rows[key]}
            for key in source_rows
        ],
        "holdout48_49_50": "UNTOUCHED",
    }


def verify_dataset_binding(binding):
    current = development_inputs(
        binding["source_manifest"], binding["question_manifest"]
    )
    if current != binding:
        raise ValueError("Frozen development manifests changed after plan freeze")


def verify_capture_reuse(directory, prepared, bound, attestation, budgets):
    """Validate a completed or partial capture before reusing its durable calls.

    A completed history is still passed through normal idempotent Add on resume;
    Search successes are never rerun merely to seek different model output.
    """
    directory = Path(directory)
    config = json.loads((directory / "api/configuration.json").read_text())
    if (
        config["runtime_attestation"] != attestation
        or config["add_requests_sha256"] != digest(prepared["requests"])
        or config["queries_sha256"] != digest(bound["api_searches"])
        or config["source_uids_sha256"] != digest(sorted(prepared["uid_provenance"]))
        or any(config[key] != value for key, value in budgets.items())
    ):
        raise ValueError("Prior API capture input/runtime/budget differs")
    return directory
