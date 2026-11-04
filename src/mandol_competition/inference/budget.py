"""Content-free, process-shared admission against the provider's real windows.

This operational database is separate from all canonical stores. It contains
only model names, reservation times and conservative token counts. Request
texts, user identifiers and credentials never enter it.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from urllib.parse import urlsplit

from .errors import InferenceAdmissionDeferred, InferenceUnavailable


@dataclass(frozen=True)
class Window:
    usage_limit: int
    usage_period: float
    request_limit: int
    request_period: float

    @classmethod
    def parse(cls, row):
        values = (row['usage_limit'], row['usage_period'],
                  row['request_limit'], row['request_period'])
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('positive finite provider windows required')
        if type(values[0]) is not int or type(values[2]) is not int:
            raise ValueError('integer provider admission limits required')
        if max(values[1], values[3]) > 3600:
            raise ValueError('bounded provider window required')
        return cls(*values)


def estimate_tokens(model, payload):
    """Conservative UTF-8 charge; never truncate or transform dispatched input.

    Rerank scores each query/document pair, so the query is charged per pair.
    Oversized estimates are admitted exclusively, rather than rejecting valid
    text based on an approximate tokenizer.
    """
    value = (payload or {}).get('input', {})
    size = lambda text: len(text.encode('utf-8')) if isinstance(text, str) else 0
    if isinstance(value, dict) and isinstance(value.get('documents'), list):
        docs = value['documents']
        return max(1, sum(size(d) for d in docs) + (size(value.get('query')) + 64) * len(docs))
    texts = value.get('texts', []) if isinstance(value, dict) else value
    if isinstance(texts, str):
        texts = [texts]
    return max(1, sum(size(text) + 16 for text in texts) if isinstance(texts, list) else 1)


class SharedBudget:
    def __init__(self, path, windows, *, clock=time.time, monotonic=time.monotonic,
                 sleep=time.sleep):
        self.path = Path(path)
        self.windows = {name: Window.parse(row) for name, row in windows.items()}
        if not self.path.is_absolute() or not self.windows:
            raise ValueError('absolute operational budget path and models required')
        self.clock, self.monotonic, self.sleep = clock, monotonic, sleep
        self._lock = threading.Lock()
        self._initialized = False

    def _connect(self):
        # Deployment creates the private directory. Never create directories in
        # a request or silently fall back to unmetered provider dispatch.
        conn = sqlite3.connect(self.path, timeout=5., isolation_level=None)
        try:
            conn.execute('PRAGMA busy_timeout=5000')
            with self._lock:
                if not self._initialized:
                    # SQLite's initial WAL transition can report BUSY before
                    # its busy handler runs when several processes start on a
                    # new database together. Retry that transition only.
                    for attempt in range(8):
                        try:
                            mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
                            if mode.lower() != 'wal':
                                conn.execute('PRAGMA journal_mode=WAL')
                            break
                        except sqlite3.OperationalError as exc:
                            if (getattr(exc, 'sqlite_errorcode', None) not in
                                    (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) or attempt == 7):
                                raise
                            self.sleep(.01*(attempt+1))
                    conn.execute('CREATE TABLE IF NOT EXISTS reservations '
                                 '(model TEXT NOT NULL, created REAL NOT NULL, tokens INTEGER NOT NULL)')
                    conn.execute('CREATE INDEX IF NOT EXISTS reservation_window ON reservations(model,created)')
                    conn.execute('CREATE TABLE IF NOT EXISTS controls '
                                 '(model TEXT PRIMARY KEY, cooldown REAL NOT NULL, next_dispatch REAL NOT NULL)')
                    self._initialized = True
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def _expiry_wait(events, now, period, required, limit):
        used = sum(charge for stamp, charge in events)
        if used + required <= limit:
            return 0.
        for stamp, charge in events:
            used -= charge
            if used + required <= limit:
                return max(0., stamp + period - now)
        return period

    def acquire(self, model, tokens, *, until):
        policy = self.windows[model]
        charge = min(max(1, int(tokens)), policy.usage_limit)
        while True:
            now = self.clock()
            remaining = until - self.monotonic()
            if remaining <= 0:
                raise InferenceAdmissionDeferred('provider_budget_deadline', retry_after=policy.usage_period)
            try:
                conn = self._connect()
                try:
                    conn.execute('BEGIN IMMEDIATE')
                    now = self.clock()
                    horizon = max(policy.usage_period, policy.request_period)
                    conn.execute('DELETE FROM reservations WHERE model=? AND created<=?', (model, now-horizon))
                    conn.execute('INSERT OR IGNORE INTO controls VALUES(?,0,0)', (model,))
                    cooldown, next_dispatch = conn.execute(
                        'SELECT cooldown,next_dispatch FROM controls WHERE model=?', (model,)).fetchone()
                    events = conn.execute('SELECT created,tokens FROM reservations '
                                          'WHERE model=? ORDER BY created', (model,)).fetchall()
                    token_events = [(stamp, n) for stamp, n in events if stamp > now-policy.usage_period]
                    request_events = [(stamp, 1) for stamp, n in events if stamp > now-policy.request_period]
                    wait = max(0., cooldown-now, next_dispatch-now,
                        self._expiry_wait(token_events, now, policy.usage_period, charge, policy.usage_limit),
                        self._expiry_wait(request_events, now, policy.request_period, 1, policy.request_limit))
                    if wait <= 1e-6:
                        conn.execute('INSERT INTO reservations VALUES(?,?,?)', (model, now, charge))
                        # Smooth dispatches as well as enforcing real windows;
                        # four clients must not release a full token burst at once.
                        interval = max(charge*policy.usage_period/policy.usage_limit,
                                       policy.request_period/policy.request_limit)
                        conn.execute('UPDATE controls SET next_dispatch=? WHERE model=?', (now+interval, model))
                    conn.execute('COMMIT')
                finally:
                    conn.close()
            except sqlite3.Error:
                raise InferenceUnavailable('provider_budget_store_unavailable') from None
            if wait <= 1e-6:
                return
            remaining = until - self.monotonic()
            if wait >= remaining:
                raise InferenceAdmissionDeferred('provider_budget_wait', retry_after=wait)
            self.sleep(wait)

    def defer(self, model, seconds):
        if model not in self.windows:
            return
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            seconds = 60.
        try:
            conn = self._connect()
            try:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('INSERT OR IGNORE INTO controls VALUES(?,0,0)', (model,))
                conn.execute('UPDATE controls SET cooldown=max(cooldown,?) WHERE model=?',
                             (self.clock()+seconds, model))
                conn.execute('COMMIT')
            finally:
                conn.close()
        except sqlite3.Error:
            raise InferenceUnavailable('provider_budget_store_unavailable') from None


class Reliability:
    def __init__(self, config):
        self.seconds = config.get('retry_budget_seconds', 1740.)
        self.attempts = config.get('retry_max_attempts', 8)
        if (type(self.seconds) not in (int, float) or not math.isfinite(self.seconds)
                or not 0 < self.seconds <= 1740 or type(self.attempts) is not int
                or not 1 <= self.attempts <= 32):
            raise ValueError('bounded inference reliability policy required')
        self.origins = frozenset(config['provider_origins'])
        for origin in self.origins:
            parsed = urlsplit(origin)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment or parsed.path):
                raise ValueError('explicit credential-free provider origin required')
        self.budget = SharedBudget(config['state_path'], config['models'])

    def model(self, endpoint, payload):
        parsed = urlsplit(endpoint)
        origin = parsed.scheme+'://'+parsed.netloc
        model = (payload or {}).get('model')
        return model if origin in self.origins and model in self.budget.windows else None


_cache = {}
_cache_lock = threading.Lock()


def from_environment():
    path = os.getenv('MANDOL_RELIABILITY_CONFIG')
    if not path:
        return None
    with _cache_lock:
        if path not in _cache:
            _cache[path] = Reliability(json.loads(Path(path).read_text()))
        return _cache[path]
