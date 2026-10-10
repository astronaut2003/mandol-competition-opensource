import ast
from collections import Counter
import math
from pathlib import Path
import re

import numpy as np
import pytest

from mandol_substrate.lexical.snapshot import Bm25Snapshot
from mandol_substrate.retrieval.candidates import Candidate, hydrate
from mandol_substrate.state import Mutation, StateUnavailable


def test_snapshot_matches_frozen_code_scoring_and_filtered_corpus():
    # Load only the archived pure class, not the old package or model loaders.
    import subprocess
    reference = Path(__file__).resolve().parents[1] / "oracles/code_lexical.py"
    assert __import__("hashlib").sha256(reference.read_bytes()).hexdigest() == "80aaeac82704476e46f1648444441abb5c68cbcfb24f77859fbd09f45cc25cd8"
    source = reference.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    env = {"Counter": Counter, "math": math, "re": re, "Mapping": dict}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "frozen-code-bm25", "exec"), env)
    docs = {"a": "alpha alpha Beta", "b": "alpha route", "c": "unrelated"}
    old = env["CodeBM25"](docs, code_aware=False)
    new = Bm25Snapshot(docs, tokenize=lambda s: re.findall(r"\b\w+\b", s.casefold()))
    for candidates in (None, [], ["a", "c"], ["b", "a", "missing"]):
        assert new.search("alpha beta", candidate_uids=candidates) == old.search("alpha beta", candidate_uids=candidates)


def test_hydration_rejects_stale_or_cross_scope(store, scope, memory):
    from mandol_substrate.state import Scope
    store.prepare(scope, "r", "d", Mutation(memories=(memory,)), {})
    store.publish(scope, 1)
    hit = Candidate(memory.logical_uid, .5, "dense", {"version": 1, "commit_seq": 1})
    assert hydrate(store, scope, [hit], 1)[memory.logical_uid].text_content == memory.text_content
    with pytest.raises(StateUnavailable):
        hydrate(store, Scope(scope.namespace, "other"), [hit], 1)
    with pytest.raises(StateUnavailable):
        hydrate(store, scope, [Candidate(memory.logical_uid, .5, "dense", {"version": 2})], 1)


def test_frozen_fusion_does_not_load_model_libraries():
    from mandol_substrate.representation import MemoryUnit
    from mandol_substrate.retrieval.fusion import ScoreFusion
    from mandol_substrate.retrieval.results import MultiRetrievalResults, RetrievalResult, RetrievalMethod
    import sys
    result = MultiRetrievalResults()
    a, b = MemoryUnit("a", {"text_content": "A"}), MemoryUnit("b", {"text_content": "B"})
    for unit, method in [(a, RetrievalMethod.BM25), (b, RetrievalMethod.BM25), (b, RetrievalMethod.SPLADE)]:
        result.add_result(RetrievalResult(unit, 1, method))
    assert ScoreFusion.rrf_fusion(result, k=60, top_k=2)[0][0].uid == "b"
    assert not {"torch", "transformers", "sentence_transformers"} & sys.modules.keys()
