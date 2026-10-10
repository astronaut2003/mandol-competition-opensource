import json
from dataclasses import asdict

import pytest

from mandol_competition.textual.core.profile import formal_profile, load_profile
from mandol_substrate.state import ProfileMismatch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "key,value",
    [
        ("dense_index", "HNSW"),
        ("sparse_metric", "BM25"),
        ("projection_identity", "logical_uid-v1"),
        ("splade_microbatch", 32),
        ("lexical_schema", 99),
        ("hlm", "ENABLED"),
        ("reranker_required", False),
        ("cosine_k", 0),
    ],
)
def test_unsupported_frozen_profile_rejected(tmp_path, key, value):
    profile = asdict(formal_profile())
    profile["options"][key] = value
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileMismatch):
        load_profile(path)
