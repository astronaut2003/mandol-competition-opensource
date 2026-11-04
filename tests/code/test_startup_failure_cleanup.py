import time

import pytest
from fastapi.testclient import TestClient

from mandol_competition.code.api.app import create_app
from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.coding_store import CodingEvidenceStore
from mandol_competition.code.api.config import Settings
from mandol_competition.code.native_sparse import POLICY
from mandol_substrate.state import ProfileMismatch


@pytest.mark.parametrize("close_raises", [False, True])
def test_failed_model_admission_releases_sqlite_owner_and_http_client(tmp_path, monkeypatch, close_raises):
    class WrongRecipe:
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            raise ProfileMismatch("injected wrong model recipe")

        def close(self):
            self.closed = True
            if close_raises:
                raise RuntimeError("injected transport close failure")

    monkeypatch.setattr(
        "mandol_competition.code.api.remote_embeddings.RecipeNativeEncoder", WrongRecipe
    )
    cfg = Settings(
        coding_db_path=str(tmp_path / "code.sqlite3"),
        coding_dense_base_url="http://127.0.0.1:1",
        inference_contract="aml-official-eligible-v1",
        qdrant_collection="code_aml_v4_v1",
        coding_raw_policy=POLICY,
    )
    service = CodingMemoryService(cfg)
    try:
        with TestClient(create_app(cfg, memory_service=service)) as client:
            until = time.monotonic() + 3
            while service.init_error is None:
                assert time.monotonic() < until
                time.sleep(.005)
            assert client.get("/health").status_code == 503
            assert client.get("/readyz").status_code == 503
            assert service.store.state is not None
            encoder = service._encoder
        assert service.store.state is None
        assert encoder.closed
        # No GC/process exit is needed to reopen the same canonical owner.
        reopened = CodingEvidenceStore(
            cfg.coding_db_path,
            inference_contract=cfg.inference_contract,
            namespace=cfg.namespace,
            collection=cfg.qdrant_collection,
        )
        try:
            reopened.initialize()
        finally:
            reopened.close()
    finally:
        service.shutdown()
