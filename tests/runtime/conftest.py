import pytest

from tests.support.scratch_qdrant import formal_stack_qdrant


@pytest.fixture
def isolated_formal_qdrant(tmp_path):
    with formal_stack_qdrant(tmp_path / "formal-qdrant") as endpoint:
        yield endpoint
