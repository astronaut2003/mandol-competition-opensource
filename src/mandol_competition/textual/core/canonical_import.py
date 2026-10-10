"""Explicit dataset convergence from disjoint canonical SQLite snapshots.

No vector storage/export, no semantic re-publication, no re-tokenization. Import
receipts live in SQLite in the same transaction as each source's tables, making
an interrupted administrative import resumable without duplicate Add/DF/edges.
"""

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path

from mandol_substrate.state.models import StateUnavailable, canonical_json, digest
from .run_storage import checked_path, checksum, fsync_file


def import_disjoint_canonical_sources(destination, sources, profile_hash, *, scratch,
                                      allow_scratch=False, fault=None):
    fault = fault or (lambda _: None)
    scratch = checked_path(scratch, allow_scratch=allow_scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    receipts = []
    with closing(sqlite3.connect(Path(destination).resolve().as_uri(), uri=True)) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name<>'manifest' ORDER BY name")]
        # Identifiers are the destination schema's own table names, never input SQL.
        if any(not name.replace("_", "").isalnum() for name in tables):
            raise StateUnavailable("unexpected canonical destination schema")
        for index, source in enumerate(sources):
            source = checked_path(source, allow_scratch=allow_scratch)
            if not source.is_file() or source == Path(destination).resolve():
                raise ValueError("separate existing canonical sources required")
            # Old staging files from a killed process are never considered truth.
            snapshot = scratch / f"source-{index:05d}-{uuid.uuid4().hex}.sqlite"
            with (
                closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original,
                closing(sqlite3.connect(snapshot)) as copy,
            ):
                original.backup(copy)
            fsync_file(snapshot)
            snapshot_hash = checksum(snapshot)
            db.execute("ATTACH DATABASE ? AS imported", (snapshot.as_uri() + "?mode=ro",))
            try:
                src_tables = [r[0] for r in db.execute("SELECT name FROM imported.sqlite_master WHERE type='table' AND name<>'manifest' ORDER BY name")]
                if src_tables != tables:
                    raise StateUnavailable("canonical source table set differs")
                if db.execute("PRAGMA imported.integrity_check").fetchall() != [("ok",)]:
                    raise StateUnavailable("source integrity check failed")
                scopes = db.execute("SELECT scope,namespace,user_id,canonical_seq,published_seq,profile_hash FROM imported.scopes ORDER BY scope").fetchall()
                if not scopes or any(r[3] != r[4] or r[5] != profile_hash for r in scopes):
                    raise StateUnavailable("source is unpublished, empty or belongs to another profile")
                bad_lexical = db.execute("SELECT count(*) FROM imported.scopes s LEFT JOIN imported.lexical_corpus_stats l USING(scope) "
                    "WHERE l.lexical_seq IS NULL OR l.lexical_seq<>s.published_seq").fetchone()[0]
                if bad_lexical or db.execute("SELECT count(*) FROM imported.mutations WHERE status='pending'").fetchone()[0]:
                    raise StateUnavailable("source lexical/publication frontier is incomplete")
                key = "canonical_import_" + digest([r[0] for r in scopes])
                old = db.execute("SELECT value FROM manifest WHERE key=?", (key,)).fetchone()
                if old:
                    receipt = json.loads(old[0])
                    if receipt["source_snapshot_sha256"] != snapshot_hash:
                        raise StateUnavailable("previously imported source changed")
                    receipts.append(receipt)
                    continue
                if db.execute("SELECT count(*) FROM scopes JOIN imported.scopes USING(scope)").fetchone()[0]:
                    raise StateUnavailable("canonical source scopes overlap; no automatic merge policy")
                counts = {}
                db.execute("BEGIN IMMEDIATE")
                db.execute("PRAGMA defer_foreign_keys=ON")
                for table in tables:
                    columns = [r[1:3] for r in db.execute(f'PRAGMA main.table_info("{table}")')]
                    if columns != [r[1:3] for r in db.execute(f'PRAGMA imported.table_info("{table}")')]:
                        raise StateUnavailable("canonical source columns differ")
                    counts[table] = db.execute(f'INSERT INTO main."{table}" SELECT * FROM imported."{table}"').rowcount
                receipt = {"source_snapshot_sha256": snapshot_hash, "scopes": scopes, "row_counts": counts}
                db.execute("INSERT INTO manifest(key,value) VALUES (?,?)", (key, canonical_json(receipt)))
                fault("canonical_import_before_commit")
                db.commit()
                fault("canonical_import_after_commit")
                receipts.append(receipt)
            except BaseException:
                db.rollback()
                raise
            finally:
                db.execute("DETACH DATABASE imported")
                # Only this function's ephemeral online backup; never a source.
                snapshot.unlink()
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise StateUnavailable("merged canonical foreign keys failed")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise StateUnavailable("merged canonical integrity failed")
    return receipts
