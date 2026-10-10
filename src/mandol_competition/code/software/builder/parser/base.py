"""Language-neutral parsed software facts used by repository builders."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


def normalized_source_hash(source: str) -> str:
    """Return a stable hash after newline and trailing-space normalization."""
    normalized = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ParsedCodeSegment:
    segment_id: str
    code_snippet: str
    segment_type: str
    start_line: int
    end_line: int
    content_hash: str
    start_column: int | None = None
    end_column: int | None = None
    source_order: int = 0
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    control_transfer: str = ""
    semantic_summary: str = ""

    @property
    def block_id(self) -> str:
        """Deprecated compatibility name."""
        return self.segment_id

    @property
    def statement_type(self) -> str:
        """Deprecated compatibility name."""
        return self.segment_type


@dataclass
class ParsedCall:
    """One statically observed call site in lexical/source order."""

    target: str
    source_order: int
    start_line: int
    end_line: int
    start_column: int | None = None
    end_column: int | None = None
    expression: str = ""
    enclosing_segment_id: str | None = None

    @property
    def enclosing_block_id(self) -> str | None:
        """Deprecated compatibility name."""
        return self.enclosing_segment_id


@dataclass
class ParsedFunction:
    name: str
    qualified_name: str
    signature: str
    file_path: str
    module_name: str
    source_code: str
    start_line: int
    end_line: int
    content_hash: str
    docstring: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    call_sites: list[ParsedCall] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    config_references: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    ast_summary: dict[str, Any] = field(default_factory=dict)
    class_qualified_name: str = ""
    is_async: bool = False
    is_test: bool = False
    segments: list[ParsedCodeSegment] = field(default_factory=list)
    symbol_identity: str = ""
    is_declaration: bool = False
    parent_function_qualified_name: str = ""
    language: str = ""

    @property
    def blocks(self) -> list[ParsedCodeSegment]:
        """Deprecated compatibility view of ``segments``."""
        return self.segments


@dataclass
class ParsedClass:
    name: str
    qualified_name: str
    file_path: str
    source_code: str
    start_line: int
    end_line: int
    content_hash: str
    base_classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    docstring: str = ""
    interfaces: list[str] = field(default_factory=list)
    kind: str = "class"
    is_test: bool = False
    annotations: list[str] = field(default_factory=list)
    symbol_identity: str = ""


@dataclass
class ParsedFile:
    path: str
    language: str
    module_name: str
    content_hash: str
    source_loc: int
    is_test: bool = False
    imports: list[str] = field(default_factory=list)
    import_aliases: dict[str, str] = field(default_factory=dict)
    exports: list[str] = field(default_factory=list)
    functions: list[ParsedFunction] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    parse_error: str = ""
    language_facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedFile:
        payload = dict(data)
        payload["functions"] = [
            cls._function_from_dict(item)
            for item in payload.get("functions", [])
        ]
        payload["classes"] = [
            ParsedClass(**item) for item in payload.get("classes", [])
        ]
        return cls(**payload)

    @classmethod
    def _function_from_dict(cls, data: dict[str, Any]) -> ParsedFunction:
        payload = dict(data)
        payload["call_sites"] = [
            cls._call_from_dict(call_site)
            for call_site in payload.get("call_sites", [])
        ]
        legacy_segments = payload.pop("blocks", [])
        payload["segments"] = [
            cls._segment_from_dict(segment)
            for segment in payload.get("segments", legacy_segments)
        ]
        return ParsedFunction(**payload)

    @staticmethod
    def _call_from_dict(data: dict[str, Any]) -> ParsedCall:
        payload = dict(data)
        if "enclosing_segment_id" not in payload:
            payload["enclosing_segment_id"] = payload.pop(
                "enclosing_block_id", None
            )
        return ParsedCall(**payload)

    @staticmethod
    def _segment_from_dict(data: dict[str, Any]) -> ParsedCodeSegment:
        payload = dict(data)
        if "segment_id" not in payload:
            payload["segment_id"] = payload.pop("block_id")
        if "segment_type" not in payload:
            payload["segment_type"] = payload.pop("statement_type", "statement_group")
        return ParsedCodeSegment(**payload)


class LanguageParser(Protocol):
    """Extensible parser contract for one source language."""

    language: str

    def parse_file(self, path: Path, relative_path: str) -> ParsedFile:
        """Parse a source file into deterministic repository facts."""
        ...


__all__ = [
    "LanguageParser",
    "ParsedBlock",
    "ParsedCodeSegment",
    "ParsedCall",
    "ParsedClass",
    "ParsedFile",
    "ParsedFunction",
    "normalized_source_hash",
]

# Import compatibility for external parsers; new code should use ParsedCodeSegment.
ParsedBlock = ParsedCodeSegment
