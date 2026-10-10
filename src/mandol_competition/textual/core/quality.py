"""Public LoCoMo retrieval regression; QA evidence never enters storage.

Oracle arrays/full postings exist ONLY in this offline qualification harness.
Serving still uses bounded candidates. Native Mandol BM25 is the lexical oracle;
exact dense/sparse dot products use the same frozen real model outputs.
"""

import math


def metrics(ranking, evidence):
    gains = [int(uid in evidence) for uid in ranking]
    result = {f"recall@{k}": sum(gains[:k]) / len(evidence) for k in (5, 10, 50)}
    result["mrr"] = next((1 / (i + 1) for i, g in enumerate(gains) if g), 0)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:10]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(evidence))))
    result["ndcg@10"] = dcg / ideal if ideal else 0
    return result
