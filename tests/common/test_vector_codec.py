from dataclasses import replace

import pytest

from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.state import Scope
from mandol_substrate.vector.codec import (
    MinimalProjectionCodec,
    point_id,
    validate_vectors,
)


def test_minimal_metadata_retains_full_source_hash_without_text(profile, scope, memory):
    codec = MinimalProjectionCodec(profile, QdrantProjectionPolicy().hash)
    source = replace(memory, text_content="中文🙂" * 40000)
    row = codec.metadata(scope, source)
    assert "text_content" not in row and "raw_data" not in row and "metadata" not in row
    assert len(str(row)) < 2000
    assert row == codec.metadata(scope, source)
    assert row["source_sha256"] != codec.metadata(scope, memory)["source_sha256"]
    assert row["projection_hash"] != codec.metadata(scope, replace(source, metadata={"x": 1}))["projection_hash"]


def test_immutable_uuid_separates_scope_version_and_frontier(profile, scope, memory):
    codec = MinimalProjectionCodec(profile, QdrantProjectionPolicy().hash)
    ids = [point_id(codec.metadata(s, m)["physical_id"]) for s, m in [
        (scope, memory), (Scope(scope.namespace, scope.user_id + " "), memory),
        (scope, replace(memory, version=2)), (scope, replace(memory, commit_seq=2))]]
    assert len(set(ids)) == 4
    assert ids[0] == point_id(codec.metadata(scope, memory)["physical_id"])


@pytest.mark.parametrize("dense,sparse", [([0, 0, 0], {}), ([1, float("nan"), 0], {}),
    ([1, 0], {}), ([1, 0, 0], {True: 1}), ([1, 0, 0], {1: -1}),
    ([1, 0, 0], {2**32: 1})])
def test_invalid_vectors_fail_before_any_rpc(dense, sparse):
    with pytest.raises((ValueError, RuntimeError)):
        validate_vectors(dense, sparse, 3)


def test_no_unqualified_approximation_or_idf():
    for args in [{"dense_exact": False}, {"sparse_modifier": "idf"}, {"quantization": "scalar"}]:
        with pytest.raises(ValueError):
            QdrantProjectionPolicy(**args)
