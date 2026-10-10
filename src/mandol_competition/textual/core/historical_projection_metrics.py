"""Offline aggregation of immutable paired projection query receipts."""

import numpy as np


def summarize(rows, *, warmup_per_history=3):
    if not rows:
        raise ValueError("paired query evidence is empty")
    identities = [(r["source_record"], r["query_index"]) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError("duplicate paired query receipt")
    measured = [r for r in rows if r["query_index"] >= warmup_per_history]
    annotated = [r for r in rows if r["gold"]]
    result = {}
    for leg in ("cosine", "splade", "hybrid"):
        result[leg] = {
            "max_abs_score_diff": max(
                r["legs"][leg]["max_abs_score_diff"] for r in rows
            ),
            "mean_kendall_common": float(
                np.mean([r["legs"][leg]["kendall_common"] for r in rows])
            ),
            "mean_overlap": {
                str(k): float(
                    np.mean([r["legs"][leg]["overlap"][str(k)] for r in rows])
                )
                for k in (1, 10, 100)
            },
            "latency_sample_count": len(measured),
            "latency_status": "MEASURED"
            if measured
            else "NOT_MEASURED_NO_POST_WARMUP_QUERIES",
            "annotated_query_count": len(annotated),
            "latency_ms": {
                n: {
                    f"p{p}": float(
                        np.percentile(
                            [r["legs"][leg]["latency_ms"][n] for r in measured], p
                        )
                    )
                    if measured
                    else None
                    for p in (50, 95, 99)
                }
                for n in ("milvus", "qdrant")
            },
            "evidence": {
                n: {
                    str(k): {
                        m: float(
                            np.mean(
                                [
                                    r["legs"][leg]["evidence"][n][str(k)][m]
                                    for r in annotated
                                ]
                            )
                        )
                        if annotated
                        else None
                        for m in ("recall", "mrr", "ndcg")
                    }
                    for k in (1, 10, 100)
                }
                for n in ("milvus", "qdrant")
            },
        }
    return result


def metric(hits, gold, provenance, k):
    ranked = [provenance[h.logical_uid]["source_uid"] for h in hits[:k]]
    # Distinct annotated source evidence; one source cannot contribute twice.
    seen, relevance = set(), []
    for source in ranked:
        relevance.append(int(source in gold and source not in seen))
        seen.add(source)
    recall = len(gold.intersection(ranked)) / len(gold) if gold else None
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(relevance))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(gold))))
    return {
        "recall": recall,
        "mrr": next((1 / (i + 1) for i, r in enumerate(relevance) if r), 0)
        if gold
        else None,
        "ndcg": float(dcg / ideal) if ideal else None,
    }
