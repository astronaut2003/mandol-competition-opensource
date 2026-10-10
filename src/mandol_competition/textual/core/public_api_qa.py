"""Public QA on verified API observations, without source-side augmentation.

The public AML prompt revision and dataset provenance are recorded separately.
Local QA results describe the selected dataset and model services; official AML
scores require leaderboard evaluation.
"""

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path

from mandol_competition.textual.core.jsonl_records import iter_jsonl_records

from .aml_quality_contract import validate_add_echo, wire_context
from .api_quality_capture import _load_calls
from .evaluation_protocol import DEFAULT_EVALUATION_PROTOCOL
from .paired_quality import (
    append_call,
    complete_stage,
    file_sha256,
    summarize,
    write_json,
)
from .public_native_qa import NativePublicQaContract
from .reranker_quality import _evidence, fingerprint, ranking_metrics


def load_api_observations(directory, provenance):
    """Recompute contexts from checksum-verified successful Search calls.

    Gold labels are not accepted by this function. Missing derived query files
    do not require another Search: the durable API call is sufficient. The
    supplied provenance belongs to this exact user and input preparation.
    """
    directory = Path(directory)
    config = json.loads((directory / "configuration.json").read_text())
    inputs = json.loads((directory / "input-contract.json").read_text())
    if (
        config["schema"] != "mandol-public-api-capture-v1"
        or config["source_uids_sha256"] != fingerprint(sorted(provenance))
        or config["add_requests_sha256"] != fingerprint(inputs["adds"])
        or config["queries_sha256"] != fingerprint(inputs["searches"])
    ):
        raise ValueError("API input/provenance binding differs")
    calls, searches = _load_calls(directory, config)
    expected_queries = {q["query_index"]: q for q in inputs["searches"]}
    if len(expected_queries) != len(inputs["searches"]):
        raise ValueError("Duplicate API query identity")
    if set(searches) != set(expected_queries):
        raise ValueError("API capture has incomplete Search observations")
    acknowledged = {
        call["request_id"]
        for call in calls
        if call["operation"] == "add" and call["status"] == "PASS"
    }
    if acknowledged != {p["request_id"] for p in inputs["adds"]}:
        raise ValueError("API capture has incomplete Add acknowledgments")
    requests = {p["request_id"]: p for p in inputs["adds"]}
    if len(requests) != len(inputs["adds"]):
        raise ValueError("Duplicate API Add identity")
    for call in calls:
        if call["operation"] != "add" or call["status"] != "PASS":
            continue
        payload = requests[call["request_id"]]
        if payload["user_id"] != config["user_id"] or call[
            "request_sha256"
        ] != fingerprint(payload):
            raise ValueError("API Add scope/request binding differs")
        validate_add_echo(payload, call["response"])
    observations = {}
    for index, saved in searches.items():
        query = expected_queries[index]["query"]
        request = {
            "user_id": config["user_id"],
            "query": query,
            "top_k": config["requested_top_k"],
        }
        if saved["request_sha256"] != fingerprint(request):
            raise ValueError("API Search request binding differs")
        context = wire_context(
            saved["response"],
            requested_top_k=config["requested_top_k"],
            context_k=config["context_k"],
            max_chars=config["max_chars"],
            allowed_uids=set(provenance),
        )
        observations[index] = {
            "user": config["user_id"],
            "query_index": index,
            "query": query,
            "context": context,
            "api_latency_ms": saved["latency_ms"],
        }
    return {
        "configuration": config,
        "binding": {
            "directory": str(directory.resolve()),
            "configuration_sha256": file_sha256(directory / "configuration.json"),
            "calls_sha256": file_sha256(directory / "calls.jsonl"),
            "inputs_sha256": file_sha256(directory / "input-contract.json"),
            "provenance_sha256": fingerprint(provenance),
        },
        "observations": observations,
    }


def source_evidence_metrics(context, provenance, evidence):
    """Distinguish any source-part presence from complete source retention.

    Public gold usually labels whole source turns, not spans. A retrieved chunk
    from a long turn is therefore a presence proxy, not proof of gold-span
    coverage. Even complete source retention is not a separate faithfulness test.
    """
    evidence = set(evidence)
    ids = context["context_ids"]
    if not evidence:
        return None
    if any(uid not in provenance for uid in context["returned_ids"]):
        raise ValueError("Evidence mapping lacks same-scope provenance")
    source_by_uid = {uid: row["source_uid"] for uid, row in provenance.items()}
    required, seen, whole = {}, {}, set()
    for row in provenance.values():
        source = row["source_uid"]
        count = row["parts"]
        if type(count) is not int or count < 1 or not 0 <= row["part"] < count:
            raise ValueError("Malformed source-part provenance")
        if source in required and required[source] != count:
            raise ValueError("Source-part count disagrees")
        required[source] = count
    for piece in context["pieces"]:
        row = provenance[piece["id"]]
        if piece["fully_visible"]:
            seen.setdefault(row["source_uid"], set()).add(row["part"])
    for source, parts in seen.items():
        if len(parts) == required[source]:
            whole.add(source)
    return {
        "returned_source_presence_proxy": ranking_metrics(
            context["returned_ids"], evidence, label_by_uid=source_by_uid
        ),
        "context_source_presence_proxy": ranking_metrics(
            ids, evidence, label_by_uid=source_by_uid
        ),
        "complete_source_context_recall": len(evidence & whole) / len(evidence),
        "fully_retained_gold_sources": sorted(evidence & whole),
        "partially_retained_gold_sources": sorted(
            evidence & {source_by_uid[uid] for uid in ids} - whole
        ),
        "gold_span_recall": "UNKNOWN_WITHOUT_SPAN_LABELS",
    }


async def generate_api_arm(
    observation,
    qa,
    provenance,
    contract,
    provider,
    protocol,
    identity,
    output,
    replay=None,
):
    """Only wire content enters Answer; labels enter Judge/evidence metrics."""
    context = observation["context"]
    if (
        context["context_k"] != protocol.context_k
        or context["max_chars"] != protocol.max_context_chars
    ):
        raise ValueError("Captured context budget differs from frozen QA protocol")
    if observation["query"] != qa["question"]:
        raise ValueError("QA question differs from the actual Search request")
    native = isinstance(contract, NativePublicQaContract)
    if not native and protocol.evaluation_protocol == DEFAULT_EVALUATION_PROTOCOL:
        raise ValueError(
            "Default official-public evaluation requires a hash-verified NativePublicQaContract"
        )
    if native and contract.evaluation_protocol != protocol.evaluation_protocol:
        raise ValueError("Contract and evaluation protocol differ")
    identity = {**identity, **contract.manifest}
    messages = (
        contract.answer_messages(qa, context["context"])
        if native
        else [
            {
                "role": "user",
                "content": contract.answer(
                    qa["question"],
                    context["context"],
                    speaker_1_name=qa.get("speaker_1_name", "speaker 1"),
                    speaker_2_name=qa.get("speaker_2_name", "speaker 2"),
                ),
            }
        ]
    )
    result = {
        "status": "FAIL",
        "evaluation_protocol": protocol.evaluation_protocol,
        "protocol_manifest": contract.manifest,
        "context": context["context"],
        "context_sha256": context["context_sha256"],
        "context_chars": context["context_chars"],
        "context_ids": context["context_ids"],
        "wire": context,
        "api_latency_ms": observation["api_latency_ms"],
        "evidence": source_evidence_metrics(
            context, provenance, _evidence(qa) if native else qa.get("evidence", [])
        ),
        "answer_request_sha256": fingerprint(messages),
    }
    generated, attempts = await complete_stage(
        provider,
        messages,
        protocol.answer_max_tokens,
        json_mode=False,
        protocol=protocol,
        identity={**identity, "stage": "generation"},
        output=output,
        replay=replay,
    )
    result.update(generation=generated, generation_attempts=attempts)
    if generated["status"] != "PASS":
        return {**result, "stage": "generation"}
    if native:
        result.update(contract.answer_metrics(qa, generated["content"]))
        judge_messages, json_mode = contract.judge_messages(qa, generated["content"])
    else:
        json_mode = False
        judge_messages = [
            {
                "role": "user",
                "content": contract.judge(
                    qa["question"], qa["answer"], generated["content"]
                ),
            }
        ]
    judge, attempts = await complete_stage(
        provider,
        judge_messages,
        protocol.judge_max_tokens,
        json_mode=json_mode,
        protocol=protocol,
        identity={**identity, "stage": "judge"},
        output=output,
        replay=replay,
    )
    result.update(
        judge=judge,
        judge_attempts=attempts,
        judge_request_sha256=fingerprint(judge_messages),
    )
    if judge["status"] != "PASS":
        return {**result, "stage": "judge"}
    try:
        score = (
            contract.parse_score(judge["content"])
            if native
            else float(contract.parse_judge_label(judge["content"]) == "CORRECT")
        )
    except (ValueError, TypeError):
        return {**result, "stage": "judge_schema", "error_type": "MalformedJudge"}
    if (
        native
        and qa["dataset"] == "longmemeval"
        and protocol.evaluation_protocol == "legacy_local_v1"
    ):
        result["longmemeval_native_judge_score"] = score
    return {
        **result,
        "status": "PASS",
        "semantic_accuracy": score,
        "metric": protocol.evaluation_protocol
        if native
        else "explicit-pinned-public-contract-diagnostic",
    }


def _load_replay_chain(previous, config, allowed, protocol):
    chain, seen = [], set()
    previous = Path(previous).resolve()
    while True:
        if previous in seen or len(chain) >= 256:
            raise ValueError("Cyclic or excessive API QA replay chain")
        seen.add(previous)
        if json.loads((previous / "configuration.json").read_text()) != config:
            raise ValueError("API QA resume configuration changed")
        chain.append(previous)
        path = previous / "resume.json"
        if not path.exists():
            break
        binding = json.loads(path.read_text())
        previous = Path(binding["previous"]).resolve()
        calls = previous / "calls.jsonl"
        if (file_sha256(calls) if calls.exists() else None) != binding["calls_sha256"]:
            raise ValueError("API QA ancestor call binding changed")
    replay = {}
    for previous in reversed(chain):
        calls = previous / "calls.jsonl"
        if not calls.exists():
            continue
        local = set()
        for call in iter_jsonl_records(calls):
            if call.get("record_sha256") != fingerprint(
                {k: v for k, v in call.items() if k != "record_sha256"}
            ):
                raise ValueError("API QA call checksum differs")
            key = tuple(
                call.get(k) for k in ("user", "query_index", "arm", "stage", "attempt")
            )
            if (
                key[:3] not in allowed
                or key[3] not in {"generation", "judge"}
                or key[4] not in range(protocol.length_retries + 1)
                or key in local
            ):
                raise ValueError("Unexpected/duplicate saved API QA call")
            local.add(key)
            if key in replay:
                fields = ("messages_sha256", "max_tokens", "json_mode", "result")
                if call.get("execution") != "reused" or any(
                    call.get(k) != replay[key].get(k) for k in fields
                ):
                    raise ValueError("API QA replay differs from its original call")
            replay[key] = call
    return replay


async def evaluate_api_qa(
    rows,
    contracts,
    provider,
    protocol,
    output,
    *,
    dataset_binding,
    connected_cohort=False,
    resume_from=None,
):
    """Evaluate fully bound API observations; replay all attempts on recovery.

    rows are evaluator-only {user, query_index, qa, provenance, arms}. Each arm
    is an actual validated API observation. The external dataset/split manifest
    must be frozen by the caller; this helper does not unlock a holdout.
    """
    selected = [(r["user"], r["query_index"]) for r in rows]
    if not rows or len(set(selected)) != len(selected):
        raise ValueError("Empty or duplicate API QA queries")
    names = sorted(rows[0]["arms"])
    if not names or any(sorted(r["arms"]) != names for r in rows):
        raise ValueError("Unequal API QA arm coverage")
    for row in rows:
        if row["qa"]["dataset"] not in contracts:
            raise ValueError("Missing pinned public prompt contract")
        for observation in row["arms"].values():
            if (observation["user"], observation["query_index"]) != (
                row["user"],
                row["query_index"],
            ):
                raise ValueError("API QA scope/query identity mismatch")
    # One source mapping per history, not a corpus-sized serialization repeated
    # for each of its questions. Context and query content remain fully bound.
    provenance_hashes = {}
    row_bindings = []
    for row in rows:
        scope = row["user"]
        provenance_hash = fingerprint(row["provenance"])
        previous = provenance_hashes.setdefault(scope, provenance_hash)
        if previous != provenance_hash:
            raise ValueError("Same API user has conflicting source provenance")
        row_bindings.append(
            {
                **{key: value for key, value in row.items() if key != "provenance"},
                "provenance_sha256": provenance_hash,
            }
        )
    config = {
        "schema": "mandol-public-api-qa-v1",
        "dataset_binding": dataset_binding,
        "rows_sha256": fingerprint(row_bindings),
        "source_provenance_sha256": provenance_hashes,
        "protocol": asdict(protocol),
        "provider": provider.metadata,
        "arms": names,
        "contracts": {key: c.manifest for key, c in contracts.items()},
        "implementation_sha256": file_sha256(__file__),
        "connected_cohort": connected_cohort,
        "official_cycle2_bundle": "UNKNOWN; this is a public proxy",
    }
    replay = {}
    if resume_from:
        allowed = {(u, i, a) for u, i in selected for a in names}
        replay = _load_replay_chain(resume_from, config, allowed, protocol)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if resume_from:
        previous = Path(resume_from).resolve()
        calls = previous / "calls.jsonl"
        write_json(
            output / "resume.json",
            {
                "previous": str(previous),
                "calls_sha256": file_sha256(calls) if calls.exists() else None,
                "replay_calls": len(replay),
            },
        )
    write_json(output / "configuration.json", config)
    semaphore, completed = asyncio.Semaphore(protocol.concurrency), []
    started = time.monotonic()

    async def one(source):
        async with semaphore:
            key = {k: source[k] for k in ("user", "query_index")}
            row = {
                **key,
                "question": source["qa"]["question"],
                "category": source["qa"].get("category"),
                "arms": {},
            }
            offset = int(fingerprint([protocol.seed, key])[:8], 16) % len(names)
            for name in names[offset:] + names[:offset]:
                row["arms"][name] = await generate_api_arm(
                    source["arms"][name],
                    source["qa"],
                    source["provenance"],
                    contracts[source["qa"]["dataset"]],
                    provider,
                    protocol,
                    {**key, "arm": name},
                    output,
                    replay,
                )
            row["row_sha256"] = fingerprint(row)
            append_call(output / "queries.jsonl", row)
            completed.append(row)

    await asyncio.gather(*(one(row) for row in rows))
    calls = list(iter_jsonl_records(output / "calls.jsonl"))
    result = summarize(
        completed, calls, names, connected_cohort=connected_cohort, seed=protocol.seed
    )
    result.update(
        elapsed_seconds=time.monotonic() - started,
        configuration_sha256=fingerprint(config),
        formal_quality_ready="NOT_CLAIMED_BY_THIS_HELPER",
    )
    write_json(output / "results.json", result)
    return result
