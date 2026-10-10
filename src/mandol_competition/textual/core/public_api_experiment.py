"""Bind complete actual-API development captures to paired native public QA."""

import json
from pathlib import Path

from mandol_substrate.state.models import digest
from .paired_quality import file_sha256
from .public_api_partition import combine_partitions, partition_records, run_identity
from .public_api_plan import checked_json, history_inputs, verify_dataset_binding
from .public_api_qa import load_api_observations


def file_binding(path):
    path = Path(path).resolve()
    return {"file": str(path), "sha256": file_sha256(path)}


def capture_files(directory):
    """Bind actual call ancestry, including partial prior runs and trace files."""
    paths, seen = set(), set()
    directory = Path(directory).resolve()
    while True:
        if directory in seen or len(seen) >= 256:
            raise ValueError("Cyclic or excessive API capture ancestry")
        seen.add(directory)
        for name in (
            "configuration.json",
            "input-contract.json",
            "calls.jsonl",
            "resume.json",
        ):
            path = directory / name
            if path.exists():
                paths.add(path)
        for name in (
            "result.json",
            "canonical-audit.json",
            "search-traces.jsonl",
            "lifecycle.json",
        ):
            path = directory.parent / name
            if path.exists():
                paths.add(path)
        if not (directory / "resume.json").exists():
            break
        directory = Path(
            json.loads((directory / "resume.json").read_text())["previous"]
        ).resolve()
    return [file_binding(path) for path in sorted(paths)]


def bind_capture_run(directory, *, allow_partition=False):
    directory = Path(directory).resolve()
    run = json.loads((directory / "run.json").read_text())
    plan_path = Path(run["plan"])
    if file_sha256(plan_path) != run["plan_sha256"]:
        raise ValueError("API capture plan changed")
    plan = json.loads(plan_path.read_text())
    if plan["schema"] != "mandol-actual-api-baseline-plan-v1":
        raise ValueError("Unexpected API capture plan")
    verify_dataset_binding(plan["dataset"])
    checked_json(plan["selection"])
    checked_json(plan["profile"])
    partition = run.get("partition")
    merged, merge_files = None, []
    if partition is not None:
        if not allow_partition:
            raise ValueError("A partial cohort cannot enter full baseline QA")
        descriptor, records = partition_records(plan, partition["index"])
        if descriptor["count"] == 1 or run != run_identity(plan_path, descriptor):
            raise ValueError("Capture partition differs from the frozen assignment")
        expected_histories = len(records)
        expected_questions = sum(r["questions"]["questions"] for _, r in records)
        expected_status = "PASS_PARTITION"
    else:
        if allow_partition:
            raise ValueError("Expected an actual partition, not another merged cohort")
        records = list(enumerate(plan["dataset"]["records"]))
        expected_histories = plan["dataset"]["histories"]
        expected_questions = plan["dataset"]["questions"]
        expected_status = "PASS"
        if plan.get("partition_count", 1) > 1:
            merge_path = directory / "merged-inputs.json"
            if file_sha256(merge_path) != run.get("merge_sha256"):
                raise ValueError("Missing or changed partition merge provenance")
            prior = json.loads(merge_path.read_text())
            merged, current = combine_partitions(
                plan_path, [p["directory"] for p in prior["partitions"]]
            )
            if prior != current:
                raise ValueError("Merged partition artifacts changed")
            merge_files = [file_binding(merge_path)] + [
                f for part in current["partitions"] for f in part["files"]
            ]
    result = json.loads((directory / "results.json").read_text())
    if (
        result["status"] != expected_status
        or result["plan_sha256"] != run["plan_sha256"]
        or result["histories"] != expected_histories
        or result["questions"] != expected_questions
        or (partition is not None and result.get("partition") != partition)
    ):
        raise ValueError("Incomplete actual API cohort cannot enter full baseline QA")
    index = json.loads((directory / "histories.json").read_text())
    users = {row["source"]["user_id"] for _, row in records}
    if set(index) != users or any(row["status"] != "PASS" for row in index.values()):
        raise ValueError("Actual API history coverage differs")
    if merged is not None and index != merged:
        raise ValueError("Merged history index differs from its verified partitions")
    files = {str(plan_path.resolve()): file_binding(plan_path)}
    files.update({f["file"]: f for f in merge_files})
    for name in ("run.json", "results.json", "histories.json"):
        files[str(directory / name)] = file_binding(directory / name)
    for user, row in index.items():
        history = Path(row["directory"])
        if file_sha256(history / "result.json") != row["result_sha256"]:
            raise ValueError("Completed API history changed")
        if json.loads((history / "result.json").read_text())["status"] != "PASS":
            raise ValueError("Failed API history cannot enter a completed cohort")
        audit = json.loads((history / "canonical-audit.json").read_text())
        if (
            audit["status"] != "PASS"
            or audit["scope"]["user_id"] != user
            or audit["scope"]["namespace"] != plan["dataset"]["namespace"]
        ):
            raise ValueError("Missing same-user canonical source audit")
        for binding in capture_files(history / "api"):
            files[binding["file"]] = binding
    return {
        "directory": str(directory),
        "files": sorted(files.values(), key=lambda b: b["file"]),
        "dataset": plan["dataset"],
        "budgets": plan["budgets"],
        "selection": plan["selection"],
        "profile": plan["profile"],
    }


def load_experiment_rows(arms):
    """Read one source history at a time; labels never augment wire context."""
    if not arms or any(not name for name in arms):
        raise ValueError("Named actual API arms required")
    first = next(iter(arms.values()))
    index_by_arm = {}
    for name, arm in arms.items():
        if arm["dataset"] != first["dataset"] or arm["budgets"] != first["budgets"]:
            raise ValueError(
                "Paired API arms need identical sources/questions/context budgets"
            )
        for binding in arm["files"]:
            if file_sha256(binding["file"]) != binding["sha256"]:
                raise ValueError("Frozen API capture artifact changed")
        index_by_arm[name] = json.loads(
            (Path(arm["directory"]) / "histories.json").read_text()
        )
        checked_json(arm["selection"])
        checked_json(arm["profile"])
    verify_dataset_binding(first["dataset"])
    rows, provenance_bindings = [], {}
    for record in first["dataset"]["records"]:
        source, questions = history_inputs(**record)
        user = source["scope"]["user_id"]
        observed = {}
        for name in arms:
            directory = Path(index_by_arm[name][user]["directory"]) / "api"
            loaded = load_api_observations(directory, source["uid_provenance"])
            if loaded["configuration"]["user_id"] != user:
                raise ValueError("Observed API result belongs to a different user")
            observed[name] = loaded["observations"]
        labels = {
            row["query_index"]: row["qa"] for row in questions["evaluator_labels"]
        }
        if any(set(values) != set(labels) for values in observed.values()):
            raise ValueError("Unequal paired API query coverage")
        provenance_bindings[user] = digest(source["uid_provenance"])
        for index, qa in labels.items():
            rows.append(
                {
                    "user": user,
                    "query_index": index,
                    "qa": qa,
                    "provenance": source["uid_provenance"],
                    "arms": {name: values[index] for name, values in observed.items()},
                }
            )
    if len(rows) != first["dataset"]["questions"]:
        raise ValueError("Full actual API QA question count differs")
    return rows, {
        "dataset": first["dataset"]["dataset"],
        "split": "development",
        "questions": len(rows),
        "histories": len(provenance_bindings),
        "provenance": provenance_bindings,
    }
