"""Shared deterministic helpers for tree-sitter based language parsers."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from .base import ParsedCall, ParsedCodeSegment, ParsedFile, normalized_source_hash

_IDENTIFIER_TYPES = {
    "identifier",
    "field_identifier",
    "namespace_identifier",
    "operator_name",
    "type_identifier",
}


def unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


class TreeSitterParserBase:
    """Small common layer; language-specific classes own structural extraction."""

    language = ""
    grammar_name = ""
    call_node_types = frozenset({"call_expression", "method_invocation"})
    nested_function_types = frozenset(
        {
            "function_definition",
            "method_declaration",
            "constructor_declaration",
        }
    )
    branch_node_types = frozenset(
        {"if_statement", "switch_statement", "conditional_expression"}
    )
    loop_node_types = frozenset(
        {
            "for_statement",
            "enhanced_for_statement",
            "while_statement",
            "do_statement",
        }
    )
    return_node_types = frozenset({"return_statement"})
    await_node_types: frozenset[str] = frozenset()
    try_node_types = frozenset({"try_statement", "try_with_resources_statement"})
    segment_control_types = {
        "if_statement": "branch",
        "switch_statement": "branch",
        "for_statement": "loop",
        "enhanced_for_statement": "loop",
        "while_statement": "loop",
        "do_statement": "loop",
        "try_statement": "exception",
        "try_with_resources_statement": "exception",
    }

    def __init__(
        self,
        *,
        build_code_segments: bool = True,
        max_segments_per_function: int = 8,
        build_semantic_blocks: bool | None = None,
        max_blocks_per_function: int | None = None,
    ) -> None:
        if build_semantic_blocks is not None:
            build_code_segments = build_semantic_blocks
        if max_blocks_per_function is not None:
            max_segments_per_function = max_blocks_per_function
        self.build_code_segments = build_code_segments
        self.max_segments_per_function = max(0, max_segments_per_function)
        self.build_semantic_blocks = self.build_code_segments
        self.max_blocks_per_function = self.max_segments_per_function
        self._thread_local = threading.local()

    def _parser(self):
        return self._parser_for(self.grammar_name or self.language)

    def _parser_for(self, grammar_name: str):
        """Return a thread-local parser, including for extension-selected grammars."""
        parsers = getattr(self._thread_local, "parsers", None)
        if parsers is None:
            parsers = {}
            self._thread_local.parsers = parsers
        parser = parsers.get(grammar_name)
        if parser is None:
            parser = get_parser(grammar_name)
            parsers[grammar_name] = parser
        return parser

    def parse_file(self, path: Path, relative_path: str) -> ParsedFile:
        source = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_source(source, relative_path)

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        raise NotImplementedError

    def _tree(self, source_bytes: bytes):
        return self._parser().parse(source_bytes)

    @staticmethod
    def _text(source: bytes, node: Node | None) -> str:
        if node is None:
            return ""
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source[start_byte:end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _start_line(node: Node) -> int:
        # Keep Point alive before reading a field.  tree-sitter 0.26 can return
        # a borrowed Point whose chained ``node.start_point.row`` access is
        # invalid on CPython 3.12.
        point = node.start_point
        return point.row + 1

    @staticmethod
    def _end_line(node: Node) -> int:
        point = node.end_point
        return point.row + 1

    @classmethod
    def _line_text(cls, source: bytes, node: Node) -> str:
        """Return complete lines so source_code agrees with public line spans."""
        text = source.decode("utf-8", errors="replace")
        return "\n".join(
            text.splitlines()[cls._start_line(node) - 1 : cls._end_line(node)]
        )

    @staticmethod
    def _walk(node: Node, *, stop_types: Sequence[str] = ()) -> Iterator[Node]:
        stack = [node]
        stop = set(stop_types)
        while stack:
            current = stack.pop()
            yield current
            if current is not node and current.type in stop:
                continue
            stack.extend(reversed(current.named_children))

    @classmethod
    def _nodes(cls, node: Node, *types: str) -> Iterator[Node]:
        allowed = set(types)
        for item in cls._walk(node):
            if item.type in allowed:
                yield item

    @staticmethod
    def _module_name(relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        return ".".join(path.with_suffix("").parts)

    @staticmethod
    def _is_test_path(relative_path: str) -> bool:
        path = PurePosixPath(relative_path)
        stem_lower = path.stem.lower()
        return (
            any(part.lower() in {"test", "tests"} for part in path.parts)
            or stem_lower.startswith("test_")
            or stem_lower.endswith(("_test", "test", "tests"))
            or stem_lower.endswith((".test", ".spec", "_spec"))
        )

    @classmethod
    def _identifier(cls, source: bytes, node: Node | None) -> str:
        if node is None:
            return ""
        if node.type in _IDENTIFIER_TYPES or node.type in {
            "destructor_name",
            "qualified_identifier",
            "scoped_identifier",
        }:
            return cls._text(source, node).strip()
        name = node.child_by_field_name("name")
        if name is not None:
            return cls._text(source, name).strip()
        declarator = node.child_by_field_name("declarator")
        if declarator is not None and declarator != node:
            return cls._identifier(source, declarator)
        for child in reversed(node.named_children):
            result = cls._identifier(source, child)
            if result:
                return result
        return ""

    @classmethod
    def _comment_before(cls, source: bytes, node: Node) -> str:
        # Avoid following ``prev_named_sibling`` through preprocessor nodes.
        # Some versions of the packaged native grammars expose unstable sibling
        # pointers for nodes nested inside ``#if`` trees.  Looking at the small
        # source window immediately before a declaration is both safer and more
        # faithful: only an adjacent comment can be its documentation.
        start_byte = node.start_byte
        prefix = source[max(0, start_byte - 32768) : start_byte].decode(
            "utf-8", errors="replace"
        )
        if not prefix or (prefix.endswith("\n\n") or prefix.endswith("\r\n\r\n")):
            return ""
        stripped = prefix.rstrip(" \t\r\n")
        raw = ""
        if stripped.endswith("*/"):
            start = stripped.rfind("/*")
            if start >= 0:
                raw = stripped[start:]
        else:
            lines = stripped.splitlines()
            comment_lines: list[str] = []
            for line in reversed(lines):
                if line.lstrip().startswith("//"):
                    comment_lines.append(line)
                    continue
                break
            if comment_lines:
                raw = "\n".join(reversed(comment_lines))
        if not raw:
            return ""
        raw = raw.strip()
        if raw.startswith("/*") and raw.endswith("*/"):
            raw = raw[2:-2]
        cleaned = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("//"):
                line = line[2:].lstrip()
            elif line.startswith("*"):
                line = line[1:].lstrip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)

    @classmethod
    def _call_name(cls, source: bytes, node: Node) -> str:
        if node.type == "method_invocation":
            name = cls._text(source, node.child_by_field_name("name")).strip()
            obj = cls._text(source, node.child_by_field_name("object")).strip()
            if obj and "(" not in obj and len(obj) < 100:
                return f"{obj}.{name}"
            return name
        if node.type == "macro_invocation":
            macro = node.child_by_field_name("macro")
            name = cls._text(source, macro).strip()
            return f"{name}!" if name else ""
        function = node.child_by_field_name("function")
        return cls._text(source, function).strip()

    def _call_sites(
        self,
        source: bytes,
        body: Node | None,
        segments: Sequence[ParsedCodeSegment] = (),
    ) -> list[ParsedCall]:
        """Extract source-ordered call sites without collapsing repeated calls."""
        if body is None:
            return []
        call_nodes = [
            node
            for node in self._walk(body, stop_types=tuple(self.nested_function_types))
            if node is not body and node.type in self.call_node_types
        ]
        call_nodes.sort(key=lambda node: (node.start_byte, node.end_byte, node.type))
        result: list[ParsedCall] = []
        for source_order, node in enumerate(call_nodes):
            target = self._call_name(source, node)
            if not target:
                continue
            start = node.start_point
            end = node.end_point
            enclosing = self._enclosing_segment(node, segments)
            result.append(
                ParsedCall(
                    target=target,
                    source_order=source_order,
                    start_line=start.row + 1,
                    end_line=end.row + 1,
                    start_column=start.column,
                    end_column=end.column,
                    expression=self._text(source, node).strip(),
                    enclosing_segment_id=(enclosing.segment_id if enclosing else None),
                )
            )
        return result

    @staticmethod
    def _enclosing_segment(
        node: Node,
        segments: Sequence[ParsedCodeSegment],
    ) -> ParsedCodeSegment | None:
        start = node.start_point
        end = node.end_point
        candidates = []
        for segment in segments:
            segment_start = (segment.start_line - 1, segment.start_column or 0)
            segment_end = (
                segment.end_line - 1,
                segment.end_column if segment.end_column is not None else 2**31,
            )
            if segment_start <= (start.row, start.column) and (
                end.row,
                end.column,
            ) <= segment_end:
                candidates.append(segment)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda segment: (
                segment.end_line - segment.start_line,
                (segment.end_column or 0) - (segment.start_column or 0),
                -segment.source_order,
            ),
        )

    @classmethod
    def _behavior(
        cls,
        source: bytes,
        body: Node | None,
        *,
        global_names: Iterable[str] = (),
    ) -> dict[str, list[str]]:
        if body is None:
            return {
                "reads": [],
                "writes": [],
                "calls": [],
                "side_effects": [],
                "exceptions": [],
                "config_references": [],
            }
        calls: set[str] = set()
        reads: set[str] = set()
        writes: set[str] = set()
        exceptions: set[str] = set()
        configs: set[str] = set()
        side_effects: set[str] = set()
        global_names = set(global_names)

        for node in cls._walk(
            body,
            stop_types=tuple(cls.nested_function_types),
        ):
            if node is not body and node.type in _IDENTIFIER_TYPES - {
                "type_identifier"
            }:
                reads.add(cls._text(source, node).strip())
            if node.type in cls.call_node_types:
                call = cls._call_name(source, node)
                if call:
                    calls.add(call)
                    lowered = call.lower()
                    if lowered in {
                        "open",
                        "fopen",
                        "freopen",
                        "read",
                        "write",
                        "files.readstring",
                        "files.writestring",
                    } or any(
                        token in lowered
                        for token in (
                            ".read_text",
                            ".write_text",
                            "ifstream",
                            "ofstream",
                        )
                    ):
                        side_effects.add("file_io")
                    if any(
                        token in lowered
                        for token in (
                            "socket",
                            "connect",
                            "send",
                            "recv",
                            "http",
                            "urlconnection",
                        )
                    ):
                        side_effects.add("network")
                    if any(
                        token in lowered
                        for token in (
                            "execute",
                            "query",
                            "preparestatement",
                            "commit",
                            "rollback",
                        )
                    ):
                        side_effects.add("database")
                    if lowered in {
                        "system",
                        "popen",
                        "exec",
                        "fork",
                        "runtime.exec",
                        "processbuilder",
                    } or lowered.endswith(".exec"):
                        side_effects.add("process")
                    if any(
                        token in lowered
                        for token in (
                            "printf",
                            "fprintf",
                            "logger.",
                            "logging.",
                            "system.out.",
                            "system.err.",
                        )
                    ):
                        side_effects.add("logging")
                    if lowered in {
                        "getenv",
                        "std::getenv",
                        "system.getenv",
                        "system.getproperty",
                    }:
                        string_node = next(
                            cls._nodes(node, "string_literal", "string_fragment"),
                            None,
                        )
                        if string_node is not None:
                            key = cls._text(source, string_node).strip("\"'")
                            if key:
                                configs.add(f"environment.{key}")
                        side_effects.add("environment")
            elif node.type == "assignment_expression":
                left = node.child_by_field_name("left")
                if left is not None:
                    for identifier in cls._walk(left):
                        if identifier.type in _IDENTIFIER_TYPES - {"type_identifier"}:
                            writes.add(cls._text(source, identifier).strip())
            elif node.type == "update_expression":
                for identifier in cls._walk(node):
                    if identifier.type in _IDENTIFIER_TYPES - {"type_identifier"}:
                        writes.add(cls._text(source, identifier).strip())
            elif node.type in {"throw_statement", "throw_expression"}:
                thrown = cls._text(source, node)
                match = re.search(r"throw\s+(?:new\s+)?([\w:.]+)", thrown)
                if match:
                    exceptions.add(match.group(1))

        if global_names.intersection(writes):
            side_effects.add("global_state")
        return {
            "reads": unique(reads),
            "writes": unique(writes),
            "calls": unique(calls),
            "side_effects": unique(side_effects),
            "exceptions": unique(exceptions),
            "config_references": unique(configs),
        }

    @classmethod
    def _ast_summary(cls, body: Node | None) -> dict[str, Any]:
        if body is None:
            return {
                "branches": 0,
                "loops": 0,
                "returns": 0,
                "awaits": 0,
                "has_try": False,
            }
        types = [node.type for node in cls._walk(body)]
        return {
            "branches": sum(item in cls.branch_node_types for item in types),
            "loops": sum(item in cls.loop_node_types for item in types),
            "returns": sum(item in cls.return_node_types for item in types),
            "awaits": sum(item in cls.await_node_types for item in types),
            "has_try": any(item in cls.try_node_types for item in types),
        }

    def _segments(
        self,
        source: bytes,
        body: Node | None,
        qualified_name: str,
    ) -> list[ParsedCodeSegment]:
        if (
            body is None
            or not self.build_code_segments
            or not self.max_segments_per_function
        ):
            return []
        control_types = self.segment_control_types
        result: list[ParsedCodeSegment] = []
        candidates = [
            node for node in self._walk(body) if node is not body and node.type in control_types
        ]
        candidates.sort(key=lambda node: (node.start_byte, node.end_byte, node.type))
        for source_order, node in enumerate(candidates[: self.max_segments_per_function]):
            if node.type not in control_types:
                continue
            snippet = self._line_text(source, node)
            facts = self._behavior(source, node)
            transfer = control_types[node.type]
            start = self._start_line(node)
            end = self._end_line(node)
            start_point = node.start_point
            end_point = node.end_point
            result.append(
                ParsedCodeSegment(
                    segment_id=(
                        f"{node.type}:{start}:{start_point.column}-"
                        f"{end}:{end_point.column}"
                    ),
                    code_snippet=snippet,
                    segment_type=node.type,
                    start_line=start,
                    end_line=end,
                    start_column=start_point.column,
                    end_column=end_point.column,
                    source_order=source_order,
                    content_hash=normalized_source_hash(snippet),
                    reads=facts["reads"],
                    writes=facts["writes"],
                    control_transfer=transfer,
                    semantic_summary=(
                        f"{node.type} segment in {qualified_name} at lines {start}-{end}."
                    ),
                )
            )
        return result

    def _blocks(
        self,
        source: bytes,
        body: Node | None,
        qualified_name: str,
    ) -> list[ParsedCodeSegment]:
        """Deprecated parser hook."""
        return self._segments(source, body, qualified_name)

    @staticmethod
    def _parse_error(root: Node) -> str:
        return "tree-sitter recovered from syntax errors" if root.has_error else ""

    def _empty_file(self, source: str, relative_path: str, root: Node) -> ParsedFile:
        return ParsedFile(
            path=PurePosixPath(relative_path).as_posix(),
            language=self.language,
            module_name=self._module_name(relative_path),
            content_hash=normalized_source_hash(source),
            source_loc=len(source.splitlines()),
            is_test=self._is_test_path(relative_path),
            # A recovered ERROR node is diagnostic coverage data, not a file
            # parse failure: preprocessor-dependent C/C++ frequently produces
            # such nodes while all top-level symbols remain usable.
            language_facts={"has_syntax_errors": bool(root.has_error)},
        )


__all__ = ["TreeSitterParserBase", "unique"]
