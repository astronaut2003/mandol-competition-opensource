"""Small JSON manifest cache for incremental repository parsing."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .git_analyzer import DiffHunk, GitCommitFacts
from .parser import ParsedFile

BUILDER_VERSION = "6"


class BuilderCache:
    """Persist hashes and parsed facts; this is not a memory/storage backend."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {
            "builder_version": BUILDER_VERSION,
            "files": {},
            "last_processed_commit": None,
            "high_level_hashes": {},
            "working_evolutions": {},
            "git_facts": None,
            "graph_consistency_version": 0,
        }

    def load(self) -> BuilderCache:
        if not self.path.is_file():
            return self
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return self
        # v5 parsed facts are compatible. Keep ownership and working history so
        # the first v6 sync can repair the existing graph, even on a no-op tree.
        if loaded.get("builder_version") not in {"5", BUILDER_VERSION}:
            return self
        self.data.update(loaded)
        self.data["builder_version"] = BUILDER_VERSION
        self.data.setdefault("files", {})
        self.data.setdefault("high_level_hashes", {})
        self.data.setdefault("working_evolutions", {})
        self.data.setdefault("git_facts", None)
        return self

    def file_hash(self, relative_path: str) -> str | None:
        item = self.data["files"].get(relative_path)
        return item.get("content_hash") if item else None

    def parsed_file(self, relative_path: str) -> ParsedFile | None:
        item = self.data["files"].get(relative_path)
        if not item or not item.get("parsed"):
            return None
        try:
            return ParsedFile.from_dict(item["parsed"])
        except (TypeError, ValueError, KeyError):
            return None

    def unit_uids(self, relative_path: str) -> list[str]:
        item = self.data["files"].get(relative_path) or {}
        return list(item.get("unit_uids", []))

    def symbol_hashes(self, relative_path: str) -> dict[str, str]:
        item = self.data["files"].get(relative_path) or {}
        return dict(item.get("symbol_hashes", {}))

    def update_file(
        self,
        relative_path: str,
        content_hash: str,
        parsed: ParsedFile | None,
        unit_uids: list[str],
        symbol_hashes: dict[str, str],
    ) -> None:
        self.data["files"][relative_path] = {
            "content_hash": content_hash,
            "parsed": parsed.to_dict() if parsed else None,
            "unit_uids": list(unit_uids),
            "symbol_hashes": dict(symbol_hashes),
        }

    def remove_file(self, relative_path: str) -> None:
        self.data["files"].pop(relative_path, None)
        self.data["high_level_hashes"].pop(relative_path, None)

    def high_level_hash(self, relative_path: str) -> dict[str, Any] | None:
        value = self.data["high_level_hashes"].get(relative_path)
        return dict(value) if isinstance(value, dict) else None

    def update_high_level_hash(
        self,
        relative_path: str,
        *,
        source_hash: str,
        input_hash: str,
        policy_hash: str,
    ) -> None:
        self.data["high_level_hashes"][relative_path] = {
            "source_hash": source_hash,
            "input_hash": input_hash,
            "policy_hash": policy_hash,
        }

    def cached_git_facts(
        self, revision: str, max_commits: int | None
    ) -> list[GitCommitFacts] | None:
        cached = self.data.get("git_facts")
        if not isinstance(cached, dict):
            return None
        if cached.get("revision") != revision or cached.get("max_commits") != max_commits:
            return None
        try:
            return [
                GitCommitFacts(
                    **{
                        **payload,
                        "hunks": {
                            path: [DiffHunk(**hunk) for hunk in hunks]
                            for path, hunks in payload.get("hunks", {}).items()
                        },
                    }
                )
                for payload in cached.get("facts", [])
            ]
        except (TypeError, ValueError):
            return None

    def update_git_facts(
        self,
        revision: str,
        max_commits: int | None,
        facts: list[GitCommitFacts],
    ) -> None:
        self.data["git_facts"] = {
            "revision": revision,
            "max_commits": max_commits,
            "facts": [asdict(fact) for fact in facts],
        }

    @property
    def paths(self) -> set[str]:
        return set(self.data["files"])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)


__all__ = ["BUILDER_VERSION", "BuilderCache"]
