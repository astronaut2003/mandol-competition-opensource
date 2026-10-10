"""Explicit bounded repair of selected *current* physical projections.

Not invoked by normal Add failure, and never a corpus/generation rebuild. The
operator chooses exact canonical UIDs; SQLite semantics/frontiers are unchanged.
"""

import math
import time

from mandol_substrate.state.models import StateUnavailable, digest


def vectors_match(row, dense, sparse):
    if row is None:
        return False
    length = math.sqrt(sum(float(v) ** 2 for v in dense))
    expected = [float(v) / length for v in dense]
    actual = row.get("dense_vector", ())
    if len(actual) != len(expected) or any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)
        for a, b in zip(actual, expected, strict=True)):
        return False
    actual_sparse = row.get("splade_sparse", {})
    return set(actual_sparse) == set(sparse) and all(math.isclose(actual_sparse[i], w,
        rel_tol=1e-5, abs_tol=1e-6) for i, w in sparse.items())


def repair_current_projections(run, scope, uids, *, retries=3, timeout=45):
    if not uids or len(uids) > run.vector.policy.batch_size or len(set(uids)) != len(uids):
        raise ValueError("explicit unique bounded repair UID batch required")
    if retries < 1 or timeout <= 0:
        raise ValueError("bounded positive repair policy required")
    with run.admission.lease(run.admission_scope), run.coordinator.gates.lease(scope, write=True):
        state = run.state.scope_state(scope)
        if not state["canonical_seq"] == state["lexical_seq"] == state["published_seq"]:
            raise StateUnavailable("pending semantic mutation must use publication recovery")
        run.state.lexical.verify_scope(scope)
        canonical = run.state.get_memories(scope, uids)
        if set(canonical) != set(uids) or any(m.lifecycle == "deleted" for m in canonical.values()):
            raise StateUnavailable("repair cannot invent or resurrect a canonical memory")
        vectors = run.coordinator.encoders.encode_documents([canonical[u].text_content for u in uids])
        if len(vectors) != len(uids):
            raise StateUnavailable("repair encoder returned partial output")
        rows = [run.vector.codec.encode(run.vector.codec.metadata(scope, canonical[u]), *v)
                for u, v in zip(uids, vectors, strict=True)]
        deadline = time.monotonic() + timeout
        errors = []
        for _ in range(retries):
            if time.monotonic() >= deadline:
                break
            try:
                run.vector.upsert_projection(scope, rows)
            except Exception as exc:  # noqa: BLE001 -- an RPC timeout is an unknown outcome
                errors.append(type(exc).__name__)
            try:
                found = run.vector.get_projection_metadata(scope, uids, vectors=True)
                if all(run.vector.codec.matches(found.get(u), row) and vectors_match(found.get(u), *v)
                       for u, row, v in zip(uids, rows, vectors, strict=True)):
                    return {"status": "REPAIRED_SELECTED_CURRENT_PROJECTIONS", "uids": list(uids),
                        "published_seq": state["published_seq"], "canonical_mutations": 0,
                        "prepared_vector_sha256": digest(vectors), "transport_errors": errors}
            except Exception as exc:  # noqa: BLE001 -- retain the failed verification; no fake success
                errors.append(type(exc).__name__)
        raise StateUnavailable("selected projection repair could not verify its vectors and metadata")
