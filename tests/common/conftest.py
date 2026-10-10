from dataclasses import replace

import pytest

from mandol_substrate.text import MandolTextProcessor
from mandol_substrate.state import (
    CanonicalMemory,
    RetrievalProfile,
    Scope,
    SQLiteStateStore,
)


@pytest.fixture
def profile():
    return RetrievalProfile(
        "test/dense",
        "dense-revision-1",
        3,
        "tokenizer-1",
        "test/splade",
        "sparse-revision-1",
        MandolTextProcessor().fingerprint,
    )


@pytest.fixture
def scope():
    return Scope("test-evaluation", "exact:user/甲")


@pytest.fixture
def store(tmp_path, profile):
    store = SQLiteStateStore(tmp_path / "state.sqlite", profile)
    yield store
    store.close()


@pytest.fixture
def memory(scope):
    return CanonicalMemory(
        scope.memory_uid("s", "r", 0),
        "alpha beta",
        raw_data={"text_content": "alpha beta"},
        request_id="r",
        session_id="s",
    )


@pytest.fixture
def another(memory):
    return replace(memory, logical_uid="another", text_content="gamma")
