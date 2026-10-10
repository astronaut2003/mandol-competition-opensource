"""Actual-API public baseline helpers, after a reranker has been selected.

These helpers never choose a model, mutate storage ownership, or mark Q0 ready.
An owning runner supplies the real, initialized Competition app and its frozen
source inputs. Qualification labels cannot be inferred from this module's tests.
"""

import json
import time
from pathlib import Path

from mandol_substrate.state.models import Scope, digest
from .api_quality_capture import capture_user_api
from .api_quality_trace import ApiQualityTrace
from .paired_quality import append_call, file_sha256, write_json


def attest_runtime(service, selection):
    """Bind a selected model record to the already validated live service."""
    if (
        not service.ready
        or service.stats()["backend"] != "qdrant"
        or selection.get("schema") != "mandol-selected-reranker-v1"
        or selection.get("status") != "SELECTED"
        or getattr(service, "query_policy", None) is not None
    ):
        raise ValueError(
            "Actual Competition runtime and explicit reranker selection required"
        )
    expected = selection["formal_reranker"]
    actual = service.reranker.profile
    if (actual.model, actual.revision, actual.tokenizer_revision) != (
        expected["model"],
        expected["revision"],
        expected["tokenizer_revision"],
    ) or service.profile.hash != selection["quality_baseline_storage_profile_hash"]:
        raise ValueError("Selected reranker/profile differs from actual runtime")
    for required in ("downstream_qa", "evidence", "deployment"):
        references = selection["evidence_bindings"][required]
        if not references:
            raise ValueError("Selected model lacks completed selection evidence")
        for binding in references:
            if file_sha256(binding["path"]) != binding["sha256"]:
                raise ValueError("Selected model evidence changed")
    service.health()
    return {
        "backend": "qdrant",
        "profile_hash": service.profile.hash,
        "namespace": service.settings.evaluation_namespace,
        "reranker_model": actual.model,
        "reranker_revision": actual.revision,
        "reranker_tokenizer_revision": actual.tokenizer_revision,
        "selection_sha256": digest(selection),
        "db_uuid": service.state.db_uuid,
        "http_boundary": "real FastAPI ASGI routing; local TestClient transport if used",
        "timing": "observed API latency includes bounded diagnostic observer overhead",
    }


def audit_raw_source_state(service, prepared):
    """Read-only bounded canonical audit after the exact public Add requests.

    It is intentionally raw-L0-specific. Later extraction/HLM experiments need a
    new explicit audit that permits proven derived output, not a relaxed count.
    """
    scope = Scope(**prepared["scope"])
    mapping = prepared["uid_provenance"]
    if scope.namespace != service.settings.evaluation_namespace:
        raise ValueError("Prepared source namespace differs")
    expected_requests = {r["request_id"]: r for r in prepared["requests"]}
    counts = {"memory_units": 0, "L0": 0, "L1": 0, "L2": 0}
    with service.coordinator.gates.lease(scope):
        frontier = service.state.scope_state(scope)
        if (
            not frontier["canonical_seq"]
            == frontier["lexical_seq"]
            == frontier["published_seq"]
        ):
            raise ValueError("Raw API baseline has mixed publication")
        for batch in service.state.iter_memories(scope, batch_size=128):
            for memory in batch:
                source = mapping.get(memory.logical_uid)
                if source is None:
                    raise ValueError(
                        "Raw baseline contains unplanned or foreign memory"
                    )
                request = expected_requests[source["request_id"]]
                message = request["messages"][source["ordinal"]]
                if (
                    digest(memory.text_content) != source["canonical_text_sha256"]
                    or memory.raw_data.get("text_content") != message["content"].strip()
                    or memory.request_id != source["request_id"]
                    or memory.session_id != source["session_id"]
                    or memory.turn_index != source["ordinal"]
                    or memory.role != message["role"]
                    or memory.business_timestamp_ms != message.get("timestamp")
                    or not memory.active
                    or memory.lifecycle == "deleted"
                    or memory.version != 1
                    or memory.commit_seq > frontier["published_seq"]
                ):
                    raise ValueError("Raw source and durable canonical memory differ")
                counts["memory_units"] += 1
                level = {
                    "L0_Observation": "L0",
                    "L1_Summary": "L1",
                    "L2_Insight": "L2",
                }.get(memory.memory_level)
                if level is None:
                    raise ValueError("Unknown memory level in raw baseline")
                counts[level] += 1
        if counts["memory_units"] != len(mapping) or counts["L0"] != len(mapping):
            raise ValueError("Raw baseline source/unit/level count differs")
        ledger = []
        for request_id, request in expected_requests.items():
            row = service.state.request(scope, request_id)
            if row is None or row["status"] != "success":
                raise ValueError("Raw baseline has an unpublished Add ledger")
            messages = [
                dict(m, timestamp=m.get("timestamp")) for m in request["messages"]
            ]
            expected_digest = digest(
                [scope.user_id, request["session_id"], request_id, messages]
            )
            if (
                row["payload_digest"] != expected_digest
                or row["mutation_seq"] > frontier["published_seq"]
            ):
                raise ValueError("Raw baseline request ledger input differs")
            ledger.append(request_id)
    return {
        "status": "PASS",
        "scope": prepared["scope"],
        "frontier": {
            k: frontier[k] for k in ("canonical_seq", "lexical_seq", "published_seq")
        },
        "counts": counts,
        "request_count": len(ledger),
        "request_ids_sha256": digest(sorted(ledger)),
        "full_source_text_verified": True,
        "derived_memory_profile": "DISABLED_RAW_L0_BASELINE",
    }


class ObservedApiClient:
    """Drain one bounded trace immediately; delegate the real HTTP call once."""

    def __init__(self, client, observer, trace_file):
        self.client, self.observer = client, observer
        self.trace_file = Path(trace_file)
        self.searches = 0

    def post(self, path, *, json):
        response = self.client.post(path, json=json)
        if path == "/search":
            self.searches += 1
            try:
                trace = self.observer.take()
            except ValueError:
                # Validation/auth/startup can reject a request before Search.
                # Preserve that distinction, never fabricate a retrieval trace.
                trace = {"status": "NOT_ENTERED", "http_status": response.status_code}
            record = {
                "search_ordinal": self.searches,
                "request_sha256": digest(json),
                "trace": trace,
            }
            append_call(self.trace_file, {**record, "record_sha256": digest(record)})
            if response.status_code == 200 and trace["status"] != "PASS":
                raise ValueError(
                    "Successful API Search lacks its required quality trace"
                )
        return response


def capture_history(
    client,
    service,
    prepared,
    queries,
    *,
    attestation,
    output,
    requested_top_k=100,
    context_k=10,
    max_chars=24000,
    resume_from=None,
):
    """One already source-validated history through real Add/Search endpoints."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if prepared["scope"]["namespace"] != attestation["namespace"]:
        raise ValueError("Actual API source/namespace mismatch")
    user = prepared["scope"]["user_id"]
    started = time.monotonic()
    try:
        with ApiQualityTrace(service) as observer:
            observed = ObservedApiClient(
                client, observer, output / "search-traces.jsonl"
            )
            result = capture_user_api(
                observed,
                user_id=user,
                add_requests=prepared["requests"],
                queries=queries,
                source_uids=list(prepared["uid_provenance"]),
                runtime_attestation=attestation,
                output=output / "api",
                requested_top_k=requested_top_k,
                context_k=context_k,
                max_chars=max_chars,
                resume_from=Path(resume_from) / "api" if resume_from else None,
            )
        audit = audit_raw_source_state(service, prepared)
        write_json(output / "canonical-audit.json", audit)
        write_json(
            output / "result.json",
            {
                "status": "PASS",
                "api": result,
                "elapsed_seconds": time.monotonic() - started,
                "source_provenance_sha256": digest(prepared["uid_provenance"]),
                "reuse_trace_directory": str(Path(resume_from).resolve())
                if resume_from
                else None,
                "QUALITY_BASELINE_V0": "NOT_CLAIMED_BY_ONE_HISTORY",
            },
        )
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "status": "INCOMPLETE",
                "error_type": type(exc).__name__,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    return json.loads((output / "result.json").read_text())
