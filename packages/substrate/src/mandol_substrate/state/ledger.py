"""Durable preparation checkpoints before adapter model work.

The existing requests table remains the publication/replay authority. This
ledger reserves the body digest earlier, so an expensive preparation cannot
accept a different request body before T1. Checkpoints contain adapter semantic
output and references only, never inline image bytes or inference credentials.
Callers serialize a request's preparation using the same scope write lease.
"""

import json
import time

from .models import Conflict, canonical_json


_MEDIA_FIELDS = frozenset({'image', 'image_uri', 'image_data_uri', 'image_data',
                           'image_url', 'url', 'data_uri'})


def _inline_media(value, field=None):
    """Reject media payload fields, without interpreting source text as media.

    A code snippet, Markdown string or OCR transcription may legitimately
    contain a Data URI. Only structured media fields are payload carriers;
    strings are never parsed a second time as JSON or markup here.
    """
    if isinstance(value, str):
        return field in _MEDIA_FIELDS and value.lstrip().lower().startswith('data:image/')
    if isinstance(value, dict):
        return any(_inline_media(child, key) for key, child in value.items())
    return isinstance(value, (list, tuple)) and any(_inline_media(child, field) for child in value)


class RequestLedger:
    def __init__(self, state):
        self.state = state

    def reserve(self, scope, request_id, payload_digest):
        if not request_id or not payload_digest:
            raise ValueError("request identity and digest required")
        self.state.ensure_scope(scope)
        self.state.assert_not_purging(scope)
        with self.state.transaction() as db:
            published = self.state.request(scope, request_id)
            if published and published["payload_digest"] != payload_digest:
                raise Conflict("same request_id with different payload")
            row = db.execute("SELECT * FROM request_preparations WHERE scope=? AND request_id=?",
                             (scope.key, request_id)).fetchone()
            if row and row["payload_digest"] != payload_digest:
                raise Conflict("same request_id with different payload")
            if not row:
                db.execute("INSERT INTO request_preparations VALUES (?,?,?,'{}',?)",
                           (scope.key, request_id, payload_digest, time.time_ns() // 1_000_000))
                return {}
            return json.loads(row["checkpoint_json"])

    def checkpoint(self, scope, request_id, payload_digest, value):
        encoded = canonical_json(value)
        if _inline_media(value):
            raise ValueError("inline media cannot enter request checkpoints")
        with self.state.transaction() as db:
            row = db.execute("SELECT payload_digest FROM request_preparations WHERE scope=? AND request_id=?",
                             (scope.key, request_id)).fetchone()
            if row is None or row[0] != payload_digest:
                raise Conflict("unreserved preparation identity")
            db.execute("UPDATE request_preparations SET checkpoint_json=?,updated_at_ms=? "
                       "WHERE scope=? AND request_id=?",
                       (encoded, time.time_ns() // 1_000_000, scope.key, request_id))


    def for_mutation(self, scope, seq):
        rows = self.state._rows("SELECT request_id,payload_digest FROM requests WHERE scope=? AND mutation_seq=? AND status<>'success'",
                                (scope.key, seq))
        if not rows:
            return None
        row = rows[0]
        return row["request_id"], row["payload_digest"], self.reserve(scope, row["request_id"], row["payload_digest"])
