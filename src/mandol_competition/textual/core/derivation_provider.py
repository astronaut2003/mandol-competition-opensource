"""Owned synchronous adapter to the existing audited GPT provider loop."""

import asyncio
import json
import threading
from pathlib import Path

from mandol_substrate.state import StateUnavailable
from .paired_quality import append_call


class DerivationModelBridge:
    def __init__(
        self,
        path,
        *,
        provider_factory,
        max_tokens=12288,
        receipt_paths=(),
        record_inputs=False,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        self.record_inputs = record_inputs
        self.lock = threading.Lock()
        from mandol_substrate.state.models import digest
        from .user_profile import checked_response

        self.cache = {}
        for source in (*receipt_paths, self.path):
            if not Path(source).exists():
                continue
            for line in Path(source).read_text().splitlines():
                row = json.loads(line)
                if "messages" not in row:
                    continue
                rec = row["response"]
                if rec.get("status") != "PASS":
                    continue
                checked_response(rec)
                key = digest([row["messages"], max_tokens])
                if row.get("input_hash", key) != key or row.get(
                    "response_hash", digest(rec)
                ) != digest(rec):
                    raise StateUnavailable("Derivation receipt digest mismatch")
                self.cache.setdefault(key, rec)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="hlm-gpt", daemon=True
        )
        self.thread.start()

        async def create():
            return provider_factory()

        try:
            self.provider = asyncio.run_coroutine_threadsafe(
                create(), self.loop
            ).result(timeout=30)
        except BaseException:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=30)
            if not self.thread.is_alive():
                self.loop.close()
            raise

    def _log(self, row):
        from mandol_substrate.state.models import digest

        row = {**row, "response_hash": digest(row["response"])}
        if not self.record_inputs:
            row = {
                **row,
                "response": {
                    k: v
                    for k, v in row["response"].items()
                    if k not in ("content", "failed_completion", "raw_completion")
                },
            }
            row.pop("messages", None)
        append_call(self.path, row)

    def __call__(self, messages):
        from mandol_substrate.state.models import digest

        key = digest([messages, self.max_tokens])
        with self.lock:
            existing = self.cache.get(key)
            if existing is not None:
                self._log(
                    {
                        "messages": messages,
                        "response": existing,
                        "replayed_exact_input": True,
                        "input_definition": "ordered messages + output token budget; original wire request SHA retained in response",
                        "input_hash": key,
                    },
                )
                return existing
        result = asyncio.run_coroutine_threadsafe(
            self.provider.complete(
                messages, max_tokens=self.max_tokens, json_mode=True
            ),
            self.loop,
        ).result(timeout=180)
        with self.lock:
            self._log(
                {
                    "messages": messages,
                    "response": result,
                    "input_hash": key,
                    "replayed_exact_input": False,
                },
            )
            if result.get("status") == "PASS":
                self.cache[key] = result
        return result

    def close(self):
        asyncio.run_coroutine_threadsafe(self.provider.close(), self.loop).result(
            timeout=30
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise StateUnavailable("Owned provider did not stop")
        self.loop.close()
