"""Exact durable GPT receipt replay for one public construction ablation.

No network fallback. Original model/schema/budget/input must be identical.
Only responses present in the source SQLite durable receipt ledger qualify.
"""

import json
import sqlite3
from pathlib import Path

from mandol_substrate.state.models import StateUnavailable, digest
from .heterogeneous_memory import SCHEMAS, terminal_length_rejection
from .paired_quality import append_call
from .reranker_quality import fingerprint
from .user_profile import checked_response


class DurableConstructionReplay:
    def __init__(self, ledger, source_sqlite, output):
        self.output = Path(output)
        self.records = {}
        with sqlite3.connect(
            f"file:{Path(source_sqlite).resolve()}?mode=ro", uri=True
        ) as db:
            durable = set()
            for (text,) in db.execute(
                "SELECT progress FROM build_jobs WHERE status='completed'"
            ):
                progress = json.loads(text)
                if "response" in progress:
                    durable.add(digest(progress["response"]))
        for line in Path(ledger).read_text().splitlines():
            row = json.loads(line)
            response = row["response"]
            if (
                response.get("status") != "PASS"
                and not terminal_length_rejection(response)
            ) or digest(response) not in durable:
                continue
            if not terminal_length_rejection(response):
                checked_response(response)
            key = response["request_sha256"]
            if key in self.records and self.records[key]["response"] != response:
                raise StateUnavailable(
                    "Ambiguous durable responses for identical request"
                )
            self.records[key] = row

    def __call__(self, messages):
        stage = json.loads(messages[-1]["content"])["stage"]
        request = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 6000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "heterogeneous_" + stage,
                    "strict": True,
                    "schema": SCHEMAS[stage].model_json_schema(),
                },
            },
        }
        key = fingerprint(request)
        saved = self.records.get(key)
        if saved is None or saved["messages"] != messages:
            raise StateUnavailable(
                "Missing exact durable construction receipt; network fallback forbidden"
            )
        append_call(
            self.output,
            {
                "messages": messages,
                "response": saved["response"],
                "execution": "REPLAYED_DURABLE_SOURCE",
                "source_response_sha256": digest(saved["response"]),
            },
        )
        return saved["response"]

    def close(self):
        pass
