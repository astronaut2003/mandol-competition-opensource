"""Bounded scheduling partitions for one frozen public API cohort.

A partition is never a complete quality experiment. These helpers do not launch
workers or claim that concurrent model serving has been qualified.
"""

import json
import time
from pathlib import Path

from mandol_substrate.state.models import digest
from .paired_quality import file_sha256, write_json
from .public_api_plan import verify_dataset_binding


def partition_records(plan, index=None):
    count = plan.get("partition_count", 1)
    if type(count) is not int or count not in (1, 2):
        raise ValueError("Only one or two explicit capture partitions are supported")
    if index is None and count == 1:
        index = 0
    if type(index) is not int or not 0 <= index < count:
        raise ValueError("Explicit valid partition index required")
    records = [
        (ordinal, row)
        for ordinal, row in enumerate(plan["dataset"]["records"])
        if ordinal % count == index
    ]
    if not records:
        raise ValueError("Empty capture partition")
    descriptor = {
        "index": index,
        "count": count,
        "users_sha256": digest([row["source"]["user_id"] for _, row in records]),
    }
    return descriptor, records


def run_identity(plan_path, partition):
    identity = {
        "plan": str(Path(plan_path).resolve()),
        "plan_sha256": file_sha256(plan_path),
    }
    if partition["count"] > 1:
        identity["partition"] = partition
    return identity


def combine_partitions(plan_path, directories):
    """Revalidate every completed partition before returning a union."""
    from .public_api_experiment import bind_capture_run

    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    verify_dataset_binding(plan["dataset"])
    count = plan.get("partition_count", 1)
    directories = [Path(p).resolve() for p in directories]
    if count != 2 or len(directories) != count or len(set(directories)) != count:
        raise ValueError("Every distinct frozen partition is required for merge")
    combined, indexes, sources = {}, set(), []
    for directory in directories:
        run = json.loads((directory / "run.json").read_text())
        if (
            Path(run["plan"]).resolve() != plan_path
            or run["plan_sha256"] != file_sha256(plan_path)
            or "partition" not in run
        ):
            raise ValueError("Partition belongs to another frozen plan")
        binding = bind_capture_run(directory, allow_partition=True)
        index = run["partition"]["index"]
        histories = json.loads((directory / "histories.json").read_text())
        if index in indexes or set(histories) & set(combined):
            raise ValueError("Duplicate partition or overlapping capture users")
        indexes.add(index)
        combined.update(histories)
        sources.append(
            {
                "index": index,
                "directory": str(directory),
                "files": binding["files"],
            }
        )
    expected = {r["source"]["user_id"] for r in plan["dataset"]["records"]}
    if indexes != set(range(count)) or set(combined) != expected:
        raise ValueError("Merged capture does not cover the complete frozen cohort")
    return combined, {
        "schema": "mandol-actual-api-partition-merge-v1",
        "plan_sha256": file_sha256(plan_path),
        "users_sha256": digest(sorted(combined)),
        "partitions": sorted(sources, key=lambda r: r["index"]),
    }


def merge_runs(plan_path, directories, output):
    started = time.monotonic()
    index, provenance = combine_partitions(plan_path, directories)
    plan = json.loads(Path(plan_path).read_text())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "merged-inputs.json", provenance)
    write_json(
        output / "run.json",
        {
            "plan": str(Path(plan_path).resolve()),
            "plan_sha256": file_sha256(plan_path),
            "merge_sha256": file_sha256(output / "merged-inputs.json"),
        },
    )
    write_json(output / "histories.json", index)
    result = {
        "status": "PASS",
        "histories": len(index),
        "questions": plan["dataset"]["questions"],
        "plan_sha256": file_sha256(plan_path),
        "elapsed_seconds": time.monotonic() - started,
        "timing_scope": "merge verification only; API latencies remain in original partitions",
        "QUALITY_BASELINE_V0": "INCOMPLETE_UNTIL_REAL_QA_AND_AUDIT",
        "real_provider_calls": 0,
        "durable_state_retained": True,
    }
    write_json(output / "results.json", result)
    return result
