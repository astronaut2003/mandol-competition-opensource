import hashlib
import json
import os
import uuid
from pathlib import Path

import pytest

from mandol_substrate.qdrant.service import QdrantServiceManager


@pytest.fixture(scope="session")
def qdrant_server(tmp_path_factory):
    from types import SimpleNamespace
    endpoint = os.environ["MANDOL_TEST_QDRANT"]
    yield SimpleNamespace(uri=endpoint)
    from .backend_support import OWNED_COLLECTIONS
    import httpx
    for uri, collection in sorted(OWNED_COLLECTIONS):
        assert uri == endpoint
        httpx.delete(uri + "/collections/" + collection, trust_env=False, timeout=10).raise_for_status()


@pytest.fixture
def backend(tmp_path, qdrant_server):
    from .backend_support import runtime

    backend = runtime(tmp_path, qdrant_server.uri, "mm_" + uuid.uuid4().hex)
    try:
        backend.start()
        yield backend
    finally:
        backend.close()
        import httpx
        httpx.delete(qdrant_server.uri + "/collections/" + backend.config.qdrant_collection, trust_env=False).raise_for_status()
