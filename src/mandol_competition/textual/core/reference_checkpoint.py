"""Per-query artifacts for public model qualification, never a serving cache.

This wrapper only records/replays identical official-reference calls. Each run
uses a new directory and a fully bound public snapshot/profile manifest.
"""

import json
import math
import time
from pathlib import Path

from .paired_quality import file_sha256, write_json
from .reranker_quality import fingerprint


class ReferenceCheckpointFailure(RuntimeError):
    pass


class CheckpointedReference:
    def __init__(self, delegate, output, identity, *, previous=None):
        self.delegate = delegate
        self.output = Path(output)
        self.identity = json.loads(json.dumps(identity))
        if delegate.profile != identity["profile"]:
            raise ValueError("Reference profile differs from checkpoint declaration")
        self.previous = Path(previous) if previous else None
        if self.previous is not None:
            manifest = json.loads((self.previous / "manifest.json").read_text())
            if (
                manifest.get("schema") != "mandol-public-reference-checkpoint-v1"
                or manifest["identity"] != identity
            ):
                raise ValueError("Reference checkpoint source/profile/code changed")
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "calls").mkdir()
        write_json(
            self.output / "manifest.json",
            {
                "schema": "mandol-public-reference-checkpoint-v1",
                "identity": identity,
                "previous": str(self.previous.resolve()) if self.previous else None,
                "usage": "Public offline qualification only; not a production retrieval cache",
            },
        )
        self.last_metrics = {}
        self.last_batch_metrics = {}
        self.last_raw_logits = None
        self.new_calls = self.reused_calls = 0

    @property
    def profile(self):
        if self.delegate.profile != self.identity["profile"]:
            raise ValueError("Live official reference profile changed")
        return self.delegate.profile

    def score(self, query, documents):
        _ = self.profile
        request = {"query": query, "documents": list(documents)}
        key = fingerprint(request)
        path = self.output / "calls" / (key + ".json")
        previous = self.previous / "calls" / path.name if self.previous else None
        if path.exists():
            previous = path
        record, incomplete = None, None
        if previous is not None and previous.exists():
            record = json.loads(previous.read_text())
            if record.get("record_sha256") != fingerprint(
                {k: v for k, v in record.items() if k != "record_sha256"}
            ):
                raise ValueError("Reference checkpoint checksum mismatch")
            if (
                record["request_sha256"] != key
                or record["identity_sha256"] != fingerprint(self.identity)
                or record["document_count"] != len(documents)
            ):
                raise ValueError("Reference checkpoint belongs to different inputs")
            if record["status"] == "INTERRUPTED":
                if previous == path:
                    raise ValueError(
                        "Resume interrupted acquisition into a new directory"
                    )
                incomplete = {
                    "path": str(previous.resolve()),
                    "sha256": file_sha256(previous),
                }
                record = None
            elif record["status"] in {"PASS", "FAIL"}:
                if previous != path:
                    write_json(path, record)
                self.reused_calls += 1
                if record["status"] == "FAIL":
                    raise ReferenceCheckpointFailure(
                        "Preserved reference failure is not silently retried"
                    )
                execution = "reused"
            else:
                raise ValueError("Unknown reference checkpoint state")
        if record is None:
            started = time.monotonic()
            record = {
                "request_sha256": key,
                "identity_sha256": fingerprint(self.identity),
                "document_count": len(documents),
                "status": "INTERRUPTED",
                "previous_incomplete_attempt": incomplete,
            }
            self.new_calls += 1
            try:
                scores = self.delegate.score(query, documents)
                if len(scores) != len(documents) or any(
                    type(s) not in (int, float) or not math.isfinite(s) for s in scores
                ):
                    raise ValueError(
                        "Invalid reference output cannot be checkpointed as success"
                    )
                if self.identity["output_kind"] == "probability" and any(
                    not 0 <= s <= 1 for s in scores
                ):
                    raise ValueError("Reference probability outside declared range")
                record.update(
                    status="PASS",
                    scores=scores,
                    raw_logits=getattr(self.delegate, "last_raw_logits", None),
                    batch_metrics=getattr(self.delegate, "last_batch_metrics", {}),
                    native_metrics=getattr(self.delegate, "last_metrics", {}),
                )
            except Exception as error:
                record["status"] = "FAIL"
                record["error_type"] = type(error).__name__
                raise
            finally:
                record["reference_seconds"] = time.monotonic() - started
                record["record_sha256"] = fingerprint(record)
                write_json(path, record)
            execution = "new"
        self.last_metrics = {
            **record["native_metrics"],
            "qualification_checkpoint": {
                "execution": execution,
                "path": str(path.resolve()),
                "file_sha256": file_sha256(path),
                "original_reference_seconds": record["reference_seconds"],
                "latency_note": "Reused acquisition latency is not native inference latency",
            },
        }
        self.last_batch_metrics = record["batch_metrics"]
        self.last_raw_logits = record["raw_logits"]
        return record["scores"]

    def close(self, **kwargs):
        return self.delegate.close(**kwargs)
