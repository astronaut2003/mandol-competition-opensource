"""Experimental legacy graph-adapter state and offline migration source.

Formal serving uses mandol_substrate.state.SQLiteStateStore through backend.py.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from mandol_substrate.representation import MemoryUnit


class RequestStatus(str, Enum):
    PREPARED = "prepared"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class BeginOutcome(str, Enum):
    ACQUIRED = "acquired"
    REPLAY = "replay"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class BeginResult:
    outcome: BeginOutcome
    response: dict[str, Any] | None = None


def canonical_body_hash(payload: object) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class CanonicalStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                body_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                response_json TEXT,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS units (
                uid TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                message_index INTEGER NOT NULL,
                raw_data_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_units_user ON units(user_id);
            CREATE INDEX IF NOT EXISTS idx_units_request ON units(request_id);
            """
        )
        self._db_lock = threading.RLock()
        self._request_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def close(self) -> None:
        with self._db_lock:
            self._db.close()

    def journal_mode(self) -> str:
        with self._db_lock:
            return str(self._db.execute("PRAGMA journal_mode").fetchone()[0])

    @contextmanager
    def hold(self, request_id: str, body_hash: str, user_id: str = "") -> Iterator[BeginResult]:
        with self._locks_guard:
            lock = self._request_locks.setdefault(request_id, threading.Lock())
        with lock:
            with self._db_lock:
                row = self._db.execute(
                    "SELECT body_hash, status, response_json FROM requests WHERE request_id = ?",
                    (request_id,),
                ).fetchone()
                if row is not None and row["body_hash"] != body_hash:
                    result = BeginResult(BeginOutcome.CONFLICT)
                elif row is not None and row["status"] == RequestStatus.SUCCEEDED.value:
                    result = BeginResult(BeginOutcome.REPLAY, json.loads(row["response_json"]))
                else:
                    self._db.execute(
                        "INSERT INTO requests VALUES (?, ?, ?, ?, NULL, ?) "
                        "ON CONFLICT(request_id) DO UPDATE SET "
                        "user_id = excluded.user_id, status = excluded.status, "
                        "response_json = NULL, updated_at = excluded.updated_at",
                        (request_id, user_id, body_hash, RequestStatus.PREPARED.value, time.time()),
                    )
                    result = BeginResult(BeginOutcome.ACQUIRED)
            yield result

    def stage(
        self, request_id: str, user_id: str, body_hash: str, units: list[MemoryUnit]
    ) -> None:
        with self._db_lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                row = self._db.execute(
                    "SELECT body_hash, status FROM requests WHERE request_id = ?", (request_id,)
                ).fetchone()
                if row is None or row["body_hash"] != body_hash or row["status"] != "prepared":
                    raise ValueError("request must be prepared with matching body")
                self._db.execute("DELETE FROM units WHERE request_id = ?", (request_id,))
                for index, unit in enumerate(units):
                    self._db.execute(
                        "INSERT INTO units VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            unit.uid,
                            request_id,
                            user_id,
                            index,
                            json.dumps(unit.raw_data, ensure_ascii=False),
                            json.dumps(unit.metadata, ensure_ascii=False),
                        ),
                    )
                self._db.execute(
                    "UPDATE requests SET user_id = ?, updated_at = ? WHERE request_id = ?",
                    (user_id, time.time(), request_id),
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def mark_succeeded(self, request_id: str, body_hash: str, response: dict[str, Any]) -> None:
        with self._db_lock:
            cursor = self._db.execute(
                "UPDATE requests SET status = ?, response_json = ?, updated_at = ? "
                "WHERE request_id = ? AND body_hash = ? AND status = ?",
                (
                    RequestStatus.SUCCEEDED.value,
                    json.dumps(response, ensure_ascii=False),
                    time.time(),
                    request_id,
                    body_hash,
                    RequestStatus.PREPARED.value,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("request is not prepared")

    def fail(self, request_id: str, body_hash: str, reason: str) -> None:
        with self._db_lock:
            self._db.execute(
                "UPDATE requests SET status = ?, response_json = ?, updated_at = ? "
                "WHERE request_id = ? AND body_hash = ? AND status = ?",
                (
                    RequestStatus.FAILED.value,
                    json.dumps({"error": reason}),
                    time.time(),
                    request_id,
                    body_hash,
                    RequestStatus.PREPARED.value,
                ),
            )

    def succeeded_units(self, user_id: str) -> list[MemoryUnit]:
        with self._db_lock:
            rows = self._db.execute(
                "SELECT u.uid, u.raw_data_json, u.metadata_json FROM units u "
                "JOIN requests r ON r.request_id = u.request_id "
                "WHERE u.user_id = ? AND r.status = ? "
                "ORDER BY r.updated_at, u.message_index, u.rowid",
                (user_id, RequestStatus.SUCCEEDED.value),
            ).fetchall()
        return [
            MemoryUnit(
                uid=row["uid"],
                raw_data=json.loads(row["raw_data_json"]),
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def delete_user(self, user_id: str) -> None:
        with self._db_lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute("DELETE FROM units WHERE user_id = ?", (user_id,))
                self._db.execute("DELETE FROM requests WHERE user_id = ?", (user_id,))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise

    def request_count(self) -> int:
        with self._db_lock:
            return int(self._db.execute("SELECT COUNT(*) FROM requests").fetchone()[0])

    def status_of(self, request_id: str) -> str | None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT status FROM requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        return str(row["status"]) if row else None
