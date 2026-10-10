"""Public evaluation HTTP-boundary capture, independent of a reranker winner.

The caller must start the real Competition app and attest its selected profile.
This helper forwards only Add/Search contract fields. It cannot mark Q0 or a
formal profile ready and never reads benchmark labels to construct memory.
"""

import json
import time
from pathlib import Path

from mandol_competition.textual.core.jsonl_records import iter_jsonl_records

from .aml_quality_contract import validate_add_echo, wire_context
from .paired_quality import append_call, file_sha256, write_json
from .reranker_quality import fingerprint


def _hash_record(record):
    return {**record, "record_sha256": fingerprint(record)}


def _load_calls(directory, expected_config):
    # A second interruption can occur before the first resume has copied its
    # successful Search calls. Follow immutable prior runs rather than losing
    # those results and accidentally resampling a completed model operation.
    chain, seen = [], set()
    directory = Path(directory).resolve()
    while True:
        if directory in seen or len(chain) >= 256:
            raise ValueError("Cyclic or excessive API resume chain")
        seen.add(directory)
        config = json.loads((directory / "configuration.json").read_text())
        if config != expected_config:
            raise ValueError("API capture resume configuration differs")
        chain.append(directory)
        resume = directory / "resume.json"
        if not resume.exists():
            break
        binding = json.loads(resume.read_text())
        previous = Path(binding["previous"]).resolve()
        previous_calls = previous / "calls.jsonl"
        actual = file_sha256(previous_calls) if previous_calls.exists() else None
        if actual != binding["calls_sha256"]:
            raise ValueError("Prior API resume call binding differs")
        directory = previous
    calls, searches = [], {}
    for directory in reversed(chain):
        path = directory / "calls.jsonl"
        if not path.exists():
            continue
        local_searches = set()
        for call in iter_jsonl_records(path):
            if call.get("record_sha256") != fingerprint(
                {k: v for k, v in call.items() if k != "record_sha256"}
            ):
                raise ValueError("API call checksum mismatch")
            calls.append(call)
            if call["operation"] == "search" and call["status"] == "PASS":
                key = call["query_index"]
                if key in local_searches:
                    raise ValueError("Duplicate completed Search result")
                local_searches.add(key)
                if key in searches:
                    previous = searches[key]
                    keys = ("request_sha256", "response", "latency_ms", "http_status")
                    if call.get("execution") != "reused" or any(
                        call.get(k) != previous.get(k) for k in keys
                    ):
                        raise ValueError("Replayed Search differs from original result")
                searches[key] = call
    return calls, searches


def _request(client, path, payload, identity, output):
    started = time.perf_counter()
    record = {**identity, "request_sha256": fingerprint(payload)}
    try:
        response = client.post(path, json=payload)
        record["http_status"] = response.status_code
        if response.status_code != 200:
            record["status"] = "FAIL"
            # Arbitrary dependency error bodies and headers are not persisted.
            record["error_type"] = "HTTP_STATUS"
            raise ValueError("API returned non200")
        body = response.json()
        if identity["operation"] == "add":
            validate_add_echo(payload, body)
            safe_body = {
                key: body[key]
                for key in ("success", "request_id", "user_id", "session_id")
            }
        else:
            # Complete schema/scope validation occurs below before a PASS can
            # become a reusable Search result. Ignore unconsumed metadata fields.
            safe_body = body
        return safe_body, record
    except Exception as exc:
        record["status"] = "FAIL"
        record.setdefault("error_type", type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        append_call(output / "calls.jsonl", _hash_record(record))
        raise
    finally:
        record["latency_ms"] = (time.perf_counter() - started) * 1000


def capture_user_api(
    client,
    *,
    user_id,
    add_requests,
    queries,
    source_uids,
    runtime_attestation,
    output,
    requested_top_k=100,
    context_k=10,
    max_chars=24000,
    resume_from=None,
):
    """Forward a frozen user's requests through the actual API boundary.

    No automatic error retry. On explicit resume every Add is acknowledged again
    with the same request identity to re-establish durable state; already durable
    Search observations are replayed with their original latency and labelled as
    reused. Incomplete/failed Search attempts are retained and may be retried in
    the new run, without overwriting old evidence or selecting among successes.
    The strict input and runtime-attestation fingerprint must match on resume.
    """
    if not user_id or not isinstance(runtime_attestation, dict):
        raise ValueError("Missing exact user/runtime attestation")
    if any(
        type(v) is not int or v <= 0 for v in (requested_top_k, context_k, max_chars)
    ):
        raise ValueError("Invalid frozen context budget")
    required = {
        "backend",
        "profile_hash",
        "namespace",
        "reranker_model",
        "reranker_revision",
    }
    if (
        not required <= set(runtime_attestation)
        or runtime_attestation["backend"] != "qdrant"
    ):
        raise ValueError("Competition runtime attestation required")
    if any(
        not isinstance(runtime_attestation[k], str) or not runtime_attestation[k]
        for k in required
    ):
        raise ValueError("Empty runtime attestation field")
    allowed = set(source_uids)
    if len(allowed) != len(source_uids):
        raise ValueError("Duplicate same-scope UID provenance")
    query_ids, request_ids = [], []
    for payload in add_requests:
        if (
            set(payload) != {"user_id", "session_id", "request_id", "messages"}
            or payload["user_id"] != user_id
        ):
            raise ValueError("Add contains wrong scope or extra fields")
        if not payload["messages"] or len(payload["messages"]) > 20:
            raise ValueError("Public Add must have1–20 messages")
        for message in payload["messages"]:
            if (
                not {"role", "content"} <= set(message)
                or not set(message) <= {"role", "content", "timestamp"}
                or message["role"] not in {"user", "assistant"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
            ):
                raise ValueError("Add source contract malformed")
        request_ids.append(payload["request_id"])
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate frozen Add request_id")
    for query in queries:
        if (
            set(query) != {"query_index", "query"}
            or type(query["query_index"]) is not int
            or query["query_index"] < 0
            or not isinstance(query["query"], str)
            or not query["query"].strip()
        ):
            raise ValueError("Search input must contain only index and original query")
        query_ids.append(query["query_index"])
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate frozen Search identity")
    config = {
        "schema": "mandol-public-api-capture-v1",
        "user_id": user_id,
        "runtime_attestation": runtime_attestation,
        "add_requests_sha256": fingerprint(add_requests),
        "queries_sha256": fingerprint(queries),
        "source_uids_sha256": fingerprint(sorted(allowed)),
        "requested_top_k": requested_top_k,
        "context_k": context_k,
        "max_chars": max_chars,
        "implementation_sha256": file_sha256(__file__),
        "context_module_sha256": file_sha256(
            Path(__file__).with_name("aml_quality_contract.py")
        ),
        "scope": "API boundary capture only; no Q0 completion or storage qualification claim",
    }
    previous_calls, searches = (
        _load_calls(resume_from, config) if resume_from else ([], {})
    )
    if not set(searches) <= set(query_ids):
        raise ValueError("Resume contains unexpected query")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "input-contract.json", {"adds": add_requests, "searches": queries}
    )
    if resume_from:
        write_json(
            output / "resume.json",
            {
                "previous": str(Path(resume_from).resolve()),
                "prior_calls": len(previous_calls),
                "calls_sha256": file_sha256(Path(resume_from) / "calls.jsonl")
                if (Path(resume_from) / "calls.jsonl").exists()
                else None,
                "complete_searches": len(searches),
                "previous_failure_preserved": True,
            },
        )
    # Publish resumable configuration only after the parent binding is durable.
    # No model/API operation has happened before this point.
    write_json(output / "configuration.json", config)
    completed_adds, completed_searches, reused_searches, error = 0, 0, 0, None
    started = time.perf_counter()
    try:
        for payload in add_requests:
            body, call = _request(
                client,
                "/add",
                payload,
                {
                    "operation": "add",
                    "request_id": payload["request_id"],
                    "execution": "new",
                },
                output,
            )
            append_call(
                output / "calls.jsonl",
                _hash_record({**call, "response": body, "status": "PASS"}),
            )
            completed_adds += 1
        for query in queries:
            payload = {
                "query": query["query"],
                "user_id": user_id,
                "top_k": requested_top_k,
            }
            saved = searches.get(query["query_index"])
            if saved is not None:
                if saved["request_sha256"] != fingerprint(payload):
                    raise ValueError("Saved Search request differs")
                body = saved["response"]
                call = {
                    k: v
                    for k, v in saved.items()
                    if k not in {"record_sha256", "response"}
                }
                call["execution"] = "reused"
            else:
                body, call = _request(
                    client,
                    "/search",
                    payload,
                    {
                        "operation": "search",
                        "query_index": query["query_index"],
                        "execution": "new",
                    },
                    output,
                )
            try:
                context = wire_context(
                    body,
                    requested_top_k=requested_top_k,
                    context_k=context_k,
                    max_chars=max_chars,
                    allowed_uids=allowed,
                )
            except Exception as exc:
                append_call(
                    output / "calls.jsonl",
                    _hash_record(
                        {**call, "status": "FAIL", "error_type": type(exc).__name__}
                    ),
                )
                raise
            body = {
                "data": [
                    {
                        k: v
                        for k, v in hit.items()
                        if k in {"id", "content", "score", "created_at"}
                    }
                    for hit in body["data"]
                ]
            }
            append_call(
                output / "calls.jsonl",
                _hash_record({**call, "response": body, "status": "PASS"}),
            )
            # If interrupted here, the durable call can reconstruct this result
            # without another model-dependent Search on a later explicit resume.
            write_json(
                output / f"query-{query['query_index']}.json",
                {
                    "query_index": query["query_index"],
                    "query": query["query"],
                    "response": body,
                    "context": context,
                    "execution": call["execution"],
                    "api_latency_ms": call["latency_ms"],
                },
            )
            completed_searches += 1
            reused_searches += call["execution"] == "reused"
    except Exception as exc:
        error = type(exc).__name__
        raise
    finally:
        write_json(
            output / "result.json",
            {
                "status": "PASS"
                if error is None
                and completed_adds == len(add_requests)
                and completed_searches == len(queries)
                else "INCOMPLETE",
                "error_type": error,
                "acknowledged_adds": completed_adds,
                "completed_searches": completed_searches,
                "reused_searches": reused_searches,
                "elapsed_seconds": time.perf_counter() - started,
                "formal_quality_baseline": "NOT_CLAIMED_BY_THIS_HELPER",
            },
        )
    return json.loads((output / "result.json").read_text())
