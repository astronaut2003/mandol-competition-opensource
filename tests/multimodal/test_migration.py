import shutil
import sqlite3

import pytest
from mandol_substrate.state import StateUnavailable

from mandol_competition.multimodal.config import ConfigurationError
from mandol_competition.multimodal.migration import import_legacy
from mandol_competition.multimodal.persistence import CanonicalStore, canonical_body_hash

from .backend_support import Encoders, request, runtime, services
from .test_sqlite_qdrant import image_part


def legacy_source(backend, root):
    source = root / "canonical.sqlite3"
    store = CanonicalStore(source)
    body = request(content=[{"type": "text", "text": "backpack"}, image_part()])
    units = services(backend)[0].prepare_units(body)
    response = {"success": True, "request_id": body.request_id, "user_id": body.user_id,
                "session_id": body.session_id}
    body_hash = canonical_body_hash(body.model_dump())
    with store.hold(body.request_id, body_hash, body.user_id):
        store.stage(body.request_id, body.user_id, body_hash, units)
        store.mark_succeeded(body.request_id, body_hash, response)
    store.close()
    old_user = root / "users" / backend.config.user_key_for(body.user_id)
    old_user.parent.mkdir(parents=True)
    shutil.move(backend.config.user_root(body.user_id), old_user)
    return source, body, units, response


def test_import_legacy_canonical_and_images_is_idempotent(backend, tmp_path):
    source, body, units, expected = legacy_source(backend, tmp_path / "old")
    before = source.read_bytes()
    report = import_legacy(backend, source)
    assert report["projection"]["clean"] and report["imported_units"] == 1
    assert source.read_bytes() == before
    assert backend.encoders.documents == 1
    assert import_legacy(backend, source) == report
    assert backend.encoders.documents == 1
    actual = next(backend.state.iter_memories(backend.scope("alice")))[0]
    assert actual.raw_data == units[0].raw_data and actual.metadata == units[0].metadata
    assert services(backend)[0].handle(body) == expected


def test_import_failure_blocks_serving_and_resumes_without_vlm(backend, tmp_path):
    source, body, _, expected = legacy_source(backend, tmp_path / "old")

    class Crash(BaseException):
        pass

    def fault(point):
        if point == "after_projection_write":
            raise Crash()

    backend.coordinator.fault = fault
    with pytest.raises(Crash):
        import_legacy(backend, source)
    config = backend.config
    backend.close()
    recovered = runtime(config.data_dir, config.qdrant_url, config.qdrant_collection)
    try:
        with pytest.raises(StateUnavailable, match="import incomplete"):
            recovered.start()
        assert import_legacy(recovered, source)["projection"]["clean"]
        recovered.start()
        assert recovered.encoders.documents == 0
        assert services(recovered)[0].handle(body) == expected
    finally:
        recovered.close()


def test_missing_legacy_media_never_publishes(backend, tmp_path):
    source, _, _, _ = legacy_source(backend, tmp_path / "old")
    shutil.rmtree(source.parent / "users")
    with pytest.raises(StateUnavailable, match="image is missing"):
        import_legacy(backend, source)
    assert not list(backend.state.iter_memories(backend.scope("alice")))
    assert backend.state._rows("SELECT value FROM manifest WHERE key='legacy_import_status'")[0]["value"] == "pending"


def test_textual_v2_schema_upgrade_preserves_projection_without_encoding(backend):
    services(backend)[0].handle(request())
    config = backend.config
    uid = next(backend.state.iter_memories(backend.scope("alice")))[0].logical_uid
    expected = backend.projection.get_projection_metadata(backend.scope("alice"), [uid])
    backend.close()
    with sqlite3.connect(config.data_dir / "state.sqlite3") as db:
        for table in ("scope_purges", "request_preparations", "projection_frontiers", "runtime_events", "scope_attributes"):
            db.execute(f"DROP TABLE {table}")
        db.execute("UPDATE scopes SET schema_version=2")
        db.execute("PRAGMA user_version=2")
    recovered = runtime(config.data_dir, config.qdrant_url, config.qdrant_collection,
                        encoders=Encoders(forbid_documents=True))
    try:
        recovered.start()
        assert recovered.projection.get_projection_metadata(recovered.scope("alice"), [uid]) == expected
        assert recovered.state.scope_state(recovered.scope("alice"))["schema_version"] == 5
    finally:
        recovered.close()


def test_legacy_data_is_never_silently_bypassed(tmp_path):
    CanonicalStore(tmp_path / "canonical.sqlite3").close()
    with pytest.raises(ConfigurationError, match="explicit import"):
        runtime(tmp_path, "http://127.0.0.1:1", "unused")
    assert not (tmp_path / "state.sqlite3").exists()
