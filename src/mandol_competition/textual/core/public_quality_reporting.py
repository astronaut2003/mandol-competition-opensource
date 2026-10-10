"""Offline public quality summaries; benchmark labels never enter runtime policy."""

import itertools
import math
from collections import Counter
from statistics import mean

from .evaluation_protocol import DEFAULT_EVALUATION_PROTOCOL, LEGACY_EVALUATION_PROTOCOL
from .paired_quality import qa_change
from .reranker_quality import paired_bootstrap


def distribution(values):
    values = sorted(values)
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("Invalid measured resource observation")
    return {
        "n": len(values),
        "mean": mean(values) if values else None,
        **{
            name: values[max(0, math.ceil(p * len(values)) - 1)] if values else None
            for name, p in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
    }


def group_summary(rows, dataset):
    """Missing observations stay unavailable; no score or evidence is imputed."""
    names = sorted(rows[0]["arms"])
    native = (
        "locomo_native_score"
        if dataset == "locomo"
        else "longmemeval_native_judge_score"
    )
    protocols = {
        arm.get("evaluation_protocol", LEGACY_EVALUATION_PROTOCOL)
        for row in rows
        for arm in row["arms"].values()
    }
    if len(protocols) != 1:
        raise ValueError("Cannot aggregate different evaluation protocols")
    protocol = protocols.pop()
    # The AML public Judge is not the native LongMemEval yes/no evaluator.
    metrics = (
        ("semantic_accuracy",)
        if (protocol == DEFAULT_EVALUATION_PROTOCOL and dataset == "longmemeval")
        else ("semantic_accuracy", native)
    )
    result = {
        "questions": len(rows),
        "evaluation_protocol": protocol,
        "arms": {},
        "paired": {},
    }
    for name in names:
        all_arms = [r["arms"][name] for r in rows]
        arms = [arm for arm in all_arms if arm["status"] == "PASS"]
        for arm in arms:
            for metric in metrics:
                score = arm.get(metric)
                if (
                    type(score) not in (int, float)
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                ):
                    raise ValueError(
                        "Completed public QA lacks its true benchmark score"
                    )
        evidence = [
            arm["evidence"] for arm in all_arms if arm.get("evidence") is not None
        ]
        summary = {
            "complete": len(arms),
            "unavailable": len(rows) - len(arms),
            "scores_observed": {
                key: mean(a[key] for a in arms) if arms else None for key in metrics
            },
            "correct_over_all_questions": sum(a["semantic_accuracy"] == 1 for a in arms)
            / len(rows),
            "semantic_all_query_bounds": [
                sum(a["semantic_accuracy"] for a in arms) / len(rows),
                (sum(a["semantic_accuracy"] for a in arms) + len(rows) - len(arms))
                / len(rows),
            ],
            "evidence_labeled_questions": len(evidence),
            "evidence": {},
            "context_chars": distribution([a["context_chars"] for a in all_arms]),
            "search_latency_ms": distribution([a["api_latency_ms"] for a in all_arms]),
        }
        for stage in (
            "returned_source_presence_proxy",
            "context_source_presence_proxy",
        ):
            summary["evidence"][stage] = {
                metric: mean(e[stage][metric] for e in evidence) if evidence else None
                for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")
            }
        summary["evidence"]["complete_source_context_recall"] = (
            mean(e["complete_source_context_recall"] for e in evidence)
            if evidence
            else None
        )
        result["arms"][name] = summary
    for left, right in itertools.combinations(names, 2):
        complete = [
            r
            for r in rows
            if all(r["arms"][a]["status"] == "PASS" for a in (left, right))
        ]
        pair = {
            "available_pairs": len(complete),
            "unavailable_pairs": len(rows) - len(complete),
            "direction": "right minus left",
            "metrics": {},
            "qa_change_taxonomy": dict(
                Counter(qa_change(r["arms"][left], r["arms"][right]) for r in rows)
            ),
        }
        for metric in metrics:
            deltas = [
                r["arms"][right][metric] - r["arms"][left][metric] for r in complete
            ]
            pair["metrics"][metric] = {
                "mean_delta_observed": mean(deltas) if deltas else None,
                "wins": sum(v > 0 for v in deltas),
                "ties": sum(v == 0 for v in deltas),
                "losses": sum(v < 0 for v in deltas),
                "bootstrap": paired_bootstrap(
                    [
                        {
                            "user": r["user"],
                            left: r["arms"][left][metric],
                            right: r["arms"][right][metric],
                        }
                        for r in complete
                    ],
                    left,
                    right,
                )
                if dataset == "locomo" and complete
                else None,
            }
        result["paired"][left + "->" + right] = pair
    result["status"] = (
        "PASS"
        if all(not a["unavailable"] for a in result["arms"].values())
        else "INCOMPLETE"
    )
    return result


def summarize_public_rows(rows, labels, dataset, reporting_contract):
    """Consume independently replay-verified rows and evaluator-only labels."""
    if dataset not in {"locomo", "longmemeval"} or not rows:
        raise ValueError("Expected an approved nonempty public cohort")
    keys = [(r["user"], r["query_index"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(labels):
        raise ValueError("Public report question coverage differs")
    names = set(rows[0]["arms"])
    if not names or any(set(r["arms"]) != names for r in rows):
        raise ValueError("Public report arm coverage differs")
    categories = reporting_contract["primary_source_category_proxies"][dataset]
    for row in rows:
        qa = labels[row["user"], row["query_index"]]
        if row["category"] != qa["category"] or str(row["category"]) not in categories:
            raise ValueError(
                "Native public category changed or is unmapped in contract"
            )
    result = {
        "dataset": dataset,
        "overall": group_summary(rows, dataset),
        "by_native_category": {},
        "by_proxy": {},
        "official_aml_overall": "NOT_MEASURED",
        "G_rules": "NOT_MEASURED",
        "H_privacy": "REQUIRES_SEPARATE_ISOLATION_QUALIFICATION",
        "uncertainty": "LoCoMo conversation-clustered; shared-history LongMemEval has no independent-question CI",
        "runtime_label_use": "FORBIDDEN",
    }
    for category in sorted({str(r["category"]) for r in rows}):
        selected = [r for r in rows if str(r["category"]) == category]
        result["by_native_category"][category] = group_summary(selected, dataset)
    for proxy in sorted({categories[str(r["category"])]["proxy"] for r in rows}):
        selected = [r for r in rows if categories[str(r["category"])]["proxy"] == proxy]
        result["by_proxy"][proxy] = group_summary(selected, dataset)
    if dataset == "longmemeval":
        selected = [
            r for r in rows if labels[r["user"], r["query_index"]].get("unanswerable")
        ]
        result["unanswerable_overlap"] = (
            group_summary(selected, dataset) if selected else None
        )
    result["status"] = result["overall"]["status"]
    return result
