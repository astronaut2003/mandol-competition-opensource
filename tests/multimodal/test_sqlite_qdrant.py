import base64
import io
import json
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from mandol_substrate.state import Conflict, Mutation, Scope, StateUnavailable
from mandol_substrate.vector.codec import point_id
from PIL import Image
from qdrant_client import models

from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.vlm.base import ExtractionError

from .backend_support import Encoders, query, request, runtime, services


def image_uri(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(stream, "PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def image_part(color="red"):
    return {"type": "image_url", "image_url": {"url": image_uri(color)}}


def test_restart_without_document_embeddings_and_idempotency(backend):
    add, search = services(backend)
    expected = add.handle(request())
    assert add.handle(request()) == expected
    assert backend.encoders.documents == 1
    with pytest.raises(Conflict):
        add.handle(request(content="different"))
    before = search.handle(query())
    config = backend.config
    backend.close()
    encoders = Encoders(forbid_documents=True)
    restarted = runtime(config.data_dir, config.qdrant_url, config.qdrant_collection, encoders=encoders)
    try:
        restarted.start()
        add, search = services(restarted)
        assert add.handle(request()) == expected
        assert search.handle(query()) == before
        assert encoders.documents == 0
        assert restarted.state.lexical.inspect_stats(restarted.scope("alice"))["n_docs"] == 1
    finally:
        restarted.close()


@pytest.mark.parametrize("content", ["red backpack", [image_part()],
    [{"type": "text", "text": "backpack"}, image_part()], [image_part(), image_part("blue")]])
def test_modalities_canonical_payload_and_media_roundtrip(backend, content):
    add, search = services(backend)
    add.handle(request(content=content))
    scope = backend.scope("alice")
    memory = next(backend.state.iter_memories(scope))[0]
    assert {"original_content", "content_parts", "multimodal_info", "text_content"} <= memory.raw_data.keys()
    assert {"extractor_model", "extractor_prompt_version", "schema_version", "session_id", "request_id"} <= memory.metadata.keys()
    assert "data:image" not in json.dumps(memory.raw_data)
    rows = next(backend.projection.iter_metadata(scope))
    assert not {"raw_data", "text_content", "metadata", "original_content"} & rows[0].keys()
    hits = search.handle(query())["data"]
    assert hits and hits[0]["id"] == memory.logical_uid
    expected_images = [p["image_url"]["url"] for p in content if isinstance(p, dict) and p["type"] == "image_url"]
    actual = [p["image_url"]["url"] for p in hits[0]["content"] if p["type"] == "image_url"]
    assert actual == expected_images


def test_vlm_degraded_and_image_query(backend):
    class BrokenVision:
        model_name = "broken"

        def extract(self, **kwargs):
            raise ExtractionError("offline")

    add, search = services(backend, BrokenVision())
    add.handle(request(content=[image_part()]))
    memory = next(backend.state.iter_memories(backend.scope("alice")))[0]
    assert memory.metadata["vlm_degraded"] is True
    result = search.handle(query().model_copy(update={"query": request(content=[image_part()]).messages[0].parts()}))
    assert result["data"][0]["content"][0] == image_part()  # official original parts order


@pytest.mark.parametrize("phase", ["before_T1", "canonical_mutation_done", "lexical_new_document",
    "lexical_new_postings", "lexical_new_df_incremented", "lexical_corpus_incremented",
    "lexical_seq_advanced", "T1_pre_commit", "T1_post_commit", "after_T1",
    "before_projection_write", "after_projection_write", "after_projection_verify",
    "after_graph_prepare", "before_T2", "T2_pre_commit", "T2_post_commit", "after_T2_before_http_return"])
def test_process_exit_each_publication_phase(tmp_path, qdrant_server, phase):
    collection = "fault_" + uuid.uuid4().hex
    worker = Path(__file__).with_name("publication_worker.py")
    result = subprocess.run([sys.executable, str(worker), str(tmp_path), qdrant_server.uri,
                             collection, phase, "add"], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 86, result.stderr
    backend = runtime(tmp_path, qdrant_server.uri, collection)
    try:
        backend.start()
        add, search = services(backend)
        assert add.handle(request())["success"]
        assert len(search.handle(query())["data"]) == 1
        assert backend.state.scope_state(backend.scope("alice"))["published_seq"] == 1
        assert backend.verify()["clean"]
        if phase in {"after_projection_write", "after_projection_verify", "after_graph_prepare",
                     "before_T2", "T2_pre_commit", "T2_post_commit", "after_T2_before_http_return"}:
            assert backend.encoders.documents == 0
    finally:
        backend.close()


@pytest.mark.parametrize("outcome", ["before", "after", "always"])
def test_qdrant_timeout_reconciliation(backend, outcome):
    original = backend.projection.upsert_projection
    calls = []

    def uncertain(scope, rows):
        calls.append(rows)
        if outcome == "after":
            original(scope, rows)
        if len(calls) == 1 or outcome == "always":
            raise TimeoutError("uncertain delivery")
        return original(scope, rows)

    backend.projection.upsert_projection = uncertain
    add, search = services(backend)
    if outcome == "always":
        with pytest.raises(StateUnavailable):
            add.handle(request())
        with pytest.raises(StateUnavailable):
            search.handle(query())
        backend.projection.upsert_projection = original
        assert add.handle(request())["success"]
    else:
        assert add.handle(request())["success"]
    assert len(search.handle(query())["data"]) == 1
    assert backend.verify()["clean"]


def test_stale_revision_and_late_write_after_purge(backend):
    add, search = services(backend)
    add.handle(request())
    scope, p = backend.scope("alice"), backend.projection
    memory = next(backend.state.iter_memories(scope))[0]
    old = p.get_projection_metadata(scope, [memory.logical_uid], vectors=True)[memory.logical_uid]
    backend.coordinator.apply(scope, "update", "digest", Mutation(memories=(
        replace(memory, version=2, text_content="blue backpack"),)), {})
    p.upsert_projection(scope, [old])
    assert p.search_dense(scope, [1., 0., 0.], limit=1, published_seq=2)[0].metadata["version"] == 2
    assert backend.verify()["stale"] == 1
    assert p.gc_scope(scope) == 1
    old = p.get_projection_metadata(scope, [memory.logical_uid], vectors=True)[memory.logical_uid]
    assert old["version"] == 2
    backend.purge("alice")
    assert backend.verify()["clean"]
    assert add.handle(request())["success"]
    p.upsert_projection(scope, [old])
    current = next(backend.state.iter_memories(scope))[0]
    assert current.commit_seq > old["commit_seq"]
    assert p.search_dense(scope, [1., 0., 0.], limit=1,
                          published_seq=current.commit_seq)[0].metadata["commit_seq"] == current.commit_seq
    assert len(search.handle(query())["data"]) == 1
    assert backend.verify()["servable"]


def test_purge_user_scope_resources_and_request_reuse(backend):
    add, search = services(backend)
    content = [{"type": "text", "text": "backpack"}, image_part()]
    for user in ("alice", "alice "):
        add.handle(request(user=user, content=content))
    other = Scope("textual", "alice")
    memory = next(backend.state.iter_memories(backend.scope("alice")))[0]
    from mandol_substrate.state import ProfileMismatch
    with pytest.raises(ProfileMismatch):
        backend.coordinator.apply(other, "other", "d", Mutation(memories=(replace(memory, spaces=()),)), {})
    report = backend.purge("alice")
    assert report["units_removed"] == 1
    assert not backend.config.user_root("alice").exists()
    assert backend.config.user_root("alice ").exists()
    assert search.handle(query())["data"] == []
    assert len(search.handle(query(user="alice "))["data"]) == 1
    assert not list(backend.state.iter_memories(other))
    with pytest.raises(ProfileMismatch):
        backend.projection.ensure_schema(other)
    assert backend.state.request(backend.scope("alice"), "r1") is None
    assert backend.state.lexical.inspect_stats(backend.scope("alice"))["n_docs"] == 0
    assert backend.purge("alice")["units_removed"] == 0
    assert add.handle(request())["success"]


@pytest.mark.parametrize("phase", ["after_purge_canonical", "after_purge_projection",
                                   "after_purge_resources", "after_purge_publish"])
def test_purge_process_crash_recovery(backend, phase):
    services(backend)[0].handle(request(content=[image_part()]))
    config = backend.config
    backend.close()
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("publication_worker.py")),
        str(config.data_dir), config.qdrant_url, config.qdrant_collection, phase, "purge"],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 86, result.stderr
    recovered = runtime(config.data_dir, config.qdrant_url, config.qdrant_collection)
    try:
        recovered.start()
        assert services(recovered)[1].handle(query())["data"] == []
        assert not recovered.config.user_root("alice").exists()
        assert recovered.verify()["clean"]
        assert recovered.encoders.documents == 0
    finally:
        recovered.close()


def test_divergence_detection_explicit_repair_and_full_rebuild(backend):
    backend.projection.client.delete_collection(backend.projection.collection_name)
    assert backend.reconcile(rebuild=True)["clean"]  # the empty corpus is rebuildable too
    add, search = services(backend)
    add.handle(request())
    p, scope = backend.projection, backend.scope("alice")
    row = next(p.iter_metadata(scope))[0]
    p.client.delete(p.collection_name, models.PointIdsList(points=[point_id(row["physical_id"])]), wait=True)
    assert backend.verify()["missing"] == 1
    with pytest.raises(StateUnavailable):
        search.handle(query())
    assert backend.reconcile()["clean"]
    assert backend.encoders.documents == 2  # only the missing row was repaired
    p.client.delete_collection(p.collection_name)
    with pytest.raises(StateUnavailable):
        search.handle(query(user="new-user"))
    assert backend.reconcile(rebuild=True)["clean"]
    assert len(search.handle(query())["data"]) == 1


@pytest.mark.parametrize("payload", [{"projection_hash": "corrupt"}, {"user_id": "wrong"}, {"version": 999}])
def test_corruption_never_becomes_canonical_or_success(backend, payload):
    services(backend)[0].handle(request())
    p = backend.projection
    row = next(p.iter_metadata(backend.scope("alice")))[0]
    p.client.set_payload(p.collection_name, payload, points=[point_id(row["physical_id"])], wait=True)
    assert not backend.verify()["servable"]
    with pytest.raises(StateUnavailable):
        services(backend)[1].handle(query())
    with pytest.raises(StateUnavailable):
        backend.reconcile()


def test_concurrent_add_search_and_duplicate_request(backend):
    add, search = services(backend)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda _: add.handle(request()), range(8)))
        assert all(r == results[0] for r in results)
        assert backend.encoders.documents == 1
        futures = [pool.submit(add.handle, request(request_id=f"r{i}")) for i in range(2, 10)]
        futures += [pool.submit(search.handle, query()) for _ in range(8)]
        for future in futures:
            future.result(timeout=30)
    assert len(search.handle(query())["data"]) == 9
    assert backend.verify()["clean"]


def test_reranker_reads_sqlite_content(backend):
    calls = []

    class Reranker:
        def score(self, question, documents):
            calls.append(documents)
            return [0.75] * len(documents)

    backend.retrieval.reranker = Reranker()
    add, search = services(backend)
    add.handle(request())
    result = search.handle(query())
    assert result["data"][0]["score"] == .75
    assert calls == [[result["data"][0]["content"][0]["text"]]]


def test_http_contract_and_pending_503(backend):
    app = create_app(backend.config, extractor=services(backend)[0].extractor, backend=backend)
    with TestClient(app) as client:
        assert client.post("/add", json=request().model_dump()).json()["success"]
        assert client.get("/health").json() == {"status": "ok"}
        assert client.post("/search", json=query().model_dump()).json()["data"]
        assert client.post("/add", json=request(content="different").model_dump()).status_code == 409
        backend.projection.upsert_projection = lambda *args: (_ for _ in ()).throw(TimeoutError())
        assert client.post("/add", json=request(request_id="r2").model_dump()).status_code == 503
        assert client.post("/search", json=query().model_dump()).status_code == 503
        assert client.get("/health").status_code == 503
        assert client.post("/admin/purge", params={"user_id": "alice"}).status_code == 200


def test_no_legacy_graph_import_for_serving():
    code = "import mandol_competition.multimodal.api, sys; assert not any(n in sys.modules for n in ['mandol.core.semantic_graph', 'mandol.core.semantic_map', 'mandol_competition.multimodal.persistence', 'mandol_competition.multimodal.store', 'faiss', 'rocksdict'])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_partial_batch_crash_only_encodes_missing_targets(tmp_path, qdrant_server):
    collection = "partial_" + uuid.uuid4().hex
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("publication_worker.py")),
        str(tmp_path), qdrant_server.uri, collection, "after_projection_write", "add_many"],
        capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 86, result.stderr
    recovered = runtime(tmp_path, qdrant_server.uri, collection)
    try:
        recovered.start()
        assert recovered.encoders.documents == 1
        assert recovered.verify()["current"] == 129
        body = request()
        body = body.model_copy(update={"messages": body.messages * 129})
        assert services(recovered)[0].handle(body)["success"]
        assert recovered.encoders.documents == 1
    finally:
        recovered.close()


def test_other_user_progress_and_purge_waits_for_ingestion(backend):
    add, search = services(backend)
    entered, release = threading.Event(), threading.Event()
    original = add.prepare_units

    def blocked(body, decoded=None, preparation=None):
        if body.user_id == "alice":
            entered.set()
            assert release.wait(5)
        return original(body, decoded, preparation)

    add.prepare_units = blocked
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(add.handle, request())
        assert entered.wait(2)
        purge = pool.submit(backend.purge, "alice")
        try:
            assert pool.submit(add.handle, request(user="bob")).result(timeout=3)["success"]
            assert not purge.done()
        finally:
            release.set()
        first.result(timeout=3)
        assert purge.result(timeout=3)["units_removed"] == 1
    assert search.handle(query())["data"] == []
    assert search.handle(query(user="bob"))["data"]


def test_purge_unknown_delete_outcome_is_verified(backend):
    services(backend)[0].handle(request(content=[image_part()]))
    original = backend.projection.client.delete
    calls = []

    def lost_reply(*args, **kwargs):
        calls.append(1)
        original(*args, **kwargs)
        raise TimeoutError("applied delete but lost reply")

    backend.projection.client.delete = lost_reply
    assert backend.purge("alice")["units_removed"] == 1
    assert len(calls) == 1
    assert backend.verify()["clean"]


def test_missing_vector_and_lexical_corruption_fail_closed(backend):
    add, search = services(backend)
    add.handle(request())
    scope = backend.scope("alice")
    with backend.state.transaction() as db:
        db.execute("UPDATE lexical_corpus_stats SET n_docs=n_docs+1 WHERE scope=?", (scope.key,))
    with pytest.raises(StateUnavailable):
        search.handle(query())
    with pytest.raises(StateUnavailable):
        backend.verify()
    backend.state.lexical.rebuild_scope(scope)
    assert search.handle(query())["data"]
    p = backend.projection
    row = next(p.iter_metadata(scope))[0]
    p.client.delete_vectors(p.collection_name, vectors=["splade"],
                            points=[point_id(row["physical_id"])], wait=True)
    assert not backend.verify()["servable"]
    with pytest.raises(StateUnavailable):
        search.handle(query())


def test_collection_creation_unknown_outcome_resumes_schema(backend):
    p = backend.projection
    p.client.delete_collection(p.collection_name)
    original = p.client.create_collection

    def lost_reply(*args, **kwargs):
        original(*args, **kwargs)
        raise TimeoutError("collection created before payload index creation")

    p.client.create_collection = lost_reply
    add, search = services(backend)
    with pytest.raises(TimeoutError):
        add.handle(request())
    assert backend.state.request(backend.scope("alice"), "r1")["status"] == "pending"
    assert backend.encoders.documents == 0
    assert add.handle(request())["success"]
    assert search.handle(query())["data"]
    assert backend.verify()["clean"]


def test_default_app_uses_shared_backend_and_http_frozen_models(tmp_path, qdrant_server, monkeypatch):
    from mandol_competition.multimodal.config import MultimodalConfig
    from mandol_competition.inference.clients import HTTPEncoders, RecipeClient
    from mandol_competition.inference.http import HTTPTransport
    monkeypatch.setattr(HTTPTransport, "health", lambda self: True)
    monkeypatch.setattr(RecipeClient, "validate_profile", lambda self: True)  # transport attestation tested separately
    from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor

    frozen = tmp_path / "profile.json"
    result = subprocess.run([sys.executable, "-m", "mandol_competition.multimodal.profile", "--output", str(frozen)],
                             capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    config = MultimodalConfig(data_dir=tmp_path, retrieval_profile_path=frozen,
        qdrant_url=qdrant_server.uri, qdrant_collection="bootstrap_" + uuid.uuid4().hex,
        env="production", auth_token="test", vlm_backend="openai_compat")
    app = create_app(config, extractor=FakeVisionExtractor({}, model="fixture", prompt_version="v1", schema_version="1"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert isinstance(app.state.backend.encoders, HTTPEncoders)
        assert not {"torch", "transformers", "sentence_transformers"} & sys.modules.keys()
        assert not (tmp_path / "canonical.sqlite3").exists()
