"""Offline, resumable import of succeeded legacy SQLite units; no graph reads."""

import hashlib
import json
import sqlite3
from pathlib import Path

from mandol_substrate.state import Mutation, Scope, StateUnavailable
from mandol_substrate.state.models import canonical_json

from mandol_substrate.representation import MemoryUnit

from ..backend import canonical_memory
from ..media_store import MediaStore, UserMediaStore, _write_atomic


def import_legacy(backend, source, *, media_root=None):
    source = Path(source).resolve()
    if source == backend.state.path:
        raise ValueError("legacy source must differ from target SQLite")
    media_root = Path(media_root) if media_root else source.parent
    with backend.maintenance(), sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as old:
        backend._started = False  # an interrupted import must revoke in-process readiness too
        old.row_factory = sqlite3.Row
        old.execute("PRAGMA query_only=ON")
        old.execute("BEGIN")  # one stable source snapshot for fingerprint and import
        hasher = hashlib.sha256()
        for table, order in (("requests", "request_id"), ("units", "uid")):
            for row in old.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                hasher.update(canonical_json(dict(row)).encode())
        identity = hasher.hexdigest()
        with backend.state.transaction() as db:
            previous = db.execute("SELECT value FROM manifest WHERE key='legacy_import_source'").fetchone()
            if previous and previous[0] != identity:
                raise StateUnavailable("legacy source changed since import began")
            db.execute("INSERT OR REPLACE INTO manifest VALUES ('legacy_import_source',?)", (identity,))
            db.execute("INSERT OR REPLACE INTO manifest VALUES ('legacy_import_status','pending')")
        backend.projection.ensure_schema(Scope(backend.config.namespace, "__import__"), create=True)
        backend.coordinator.recover()
        imported = 0
        for record in old.execute("SELECT * FROM requests WHERE status='succeeded' ORDER BY updated_at,request_id"):
            user = record["user_id"]
            scope = backend.scope(user)
            response = json.loads(record["response_json"])
            if response.get("user_id") != user or response.get("request_id") != record["request_id"]:
                raise ValueError("legacy request response identity mismatch")
            rows = old.execute("SELECT * FROM units WHERE request_id=? ORDER BY message_index,uid",
                               (record["request_id"],)).fetchall()
            if not rows or any(row["user_id"] != user for row in rows):
                raise ValueError("legacy succeeded request has missing/cross-user units")
            units = [MemoryUnit(row["uid"], json.loads(row["raw_data_json"]), json.loads(row["metadata_json"]))
                     for row in rows]
            memories = tuple(canonical_memory(unit, user) for unit in units)
            legacy_media = MediaStore(media_root / "users" / backend.config.user_key_for(user) / "media")
            destination = UserMediaStore(backend.config, user)
            for memory in memories:
                for part in memory.raw_data.get("content_parts", []):
                    if part.get("type") != "image":
                        continue
                    media_id = part["media_id"]
                    loaded = destination.get(media_id) or legacy_media.get(media_id)
                    if loaded is None:
                        raise StateUnavailable("legacy canonical image is missing or corrupt")
                    data, mime = loaded
                    sha = media_id[4:]
                    path = destination.root / sha[:2] / sha[2:4] / (sha + ".bin")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_atomic(path.with_suffix(".mime"), mime.encode("ascii"))
                    _write_atomic(path, data)
            backend.coordinator.apply(scope, record["request_id"], record["body_hash"],
                                       Mutation(memories=memories, spaces=("aml",)), response)
            imported += len(memories)
        from mandol_substrate.vector.verification import verify_projection

        report = verify_projection(backend.state, backend.projection)
        if not report["servable"]:
            raise StateUnavailable("import projection verification failed")
        with backend.state.transaction() as db:
            db.execute("UPDATE manifest SET value='complete' WHERE key='legacy_import_status'")
        return {"imported_units": imported, "source_sha256": identity, "projection": report}
