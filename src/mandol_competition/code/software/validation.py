"""Deterministic quality and persistence fingerprints for software memories."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .relations import SoftwareRelation
from .schemas import SoftwareUnitType


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RepositorySnapshotFingerprint:
    """Stable structural digest used to compare build and load results."""

    unit_count: int
    graph_node_count: int
    graph_edge_count: int
    dense_index_ntotal: int
    units_by_type: dict[str, int]
    relationships_by_type: dict[str, int]
    uid_hash: str
    raw_data_hash: str
    metadata_hash: str
    graph_edges_hash: str
    memory_space_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_graph(cls, semantic_graph) -> RepositorySnapshotFingerprint:
        units = sorted(
            semantic_graph.semantic_map.get_all_units(), key=lambda unit: unit.uid
        )
        units_by_type = Counter(
            str(unit.metadata.get("software_type", "unknown")) for unit in units
        )
        edges: list[dict[str, Any]] = []
        relationships = Counter()
        for edge_index in semantic_graph.rx_graph.edge_indices():
            source_index, target_index = (
                semantic_graph.rx_graph.get_edge_endpoints_by_index(edge_index)
            )
            data = semantic_graph.rx_graph.get_edge_data_by_index(edge_index) or {}
            relation = str(data.get("type", ""))
            relationships[relation] += 1
            edges.append(
                {
                    "source": semantic_graph._index_to_uid.get(source_index, ""),
                    "target": semantic_graph._index_to_uid.get(target_index, ""),
                    "data": data,
                }
            )
        edges.sort(
            key=lambda item: (
                item["source"],
                item["target"],
                str(item["data"].get("type", "")),
                _hash(item["data"]),
            )
        )
        spaces = {
            name: {
                "unit_uids": sorted(space.get_unit_uids()),
                "child_space_names": sorted(space.get_child_space_names()),
            }
            for name, space in sorted(semantic_graph.semantic_map.memory_spaces.items())
        }
        faiss_index = semantic_graph.semantic_map.faiss_index
        return cls(
            unit_count=len(units),
            graph_node_count=semantic_graph.rx_graph.num_nodes(),
            graph_edge_count=semantic_graph.rx_graph.num_edges(),
            dense_index_ntotal=int(faiss_index.ntotal)
            if faiss_index is not None
            else 0,
            units_by_type=dict(sorted(units_by_type.items())),
            relationships_by_type=dict(sorted(relationships.items())),
            uid_hash=_hash([unit.uid for unit in units]),
            raw_data_hash=_hash({unit.uid: unit.raw_data for unit in units}),
            metadata_hash=_hash({unit.uid: unit.metadata for unit in units}),
            graph_edges_hash=_hash(edges),
            memory_space_hash=_hash(spaces),
        )


@dataclass(frozen=True)
class RepositoryQualityMetrics:
    source_files: int
    parsed_files: int
    parse_success_rate: float
    unsupported_file_rate: float
    syntax_error_rate: float
    files_with_zero_symbols: int
    functions: int
    code_segments: int
    functions_with_valid_source_span_rate: float
    functions_with_nonempty_source_code_rate: float
    functions_with_signature_rate: float
    source_code_match_rate: float
    code_segments_with_valid_source_span_rate: float
    code_segment_source_match_rate: float
    calls_recorded: int
    calls_resolved: int
    calls_resolved_rate: float
    dangling_edge_count: int
    duplicate_uid_count: int
    invalid_source_span_count: int
    source_code_mismatch_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_repository_quality(
    semantic_code, repo_path: str | Path
) -> RepositoryQualityMetrics:
    """Measure basic graph integrity and source-location correctness."""
    graph = semantic_code.semantic_graph
    root = Path(repo_path)
    units = graph.semantic_map.get_all_units()
    files = [
        unit
        for unit in units
        if unit.metadata.get("software_type") == SoftwareUnitType.FILE.value
        and unit.raw_data.get("language") != "configuration"
    ]
    functions = [
        unit
        for unit in units
        if unit.metadata.get("software_type") == SoftwareUnitType.FUNCTION.value
    ]
    segments = [
        unit
        for unit in units
        if unit.metadata.get("software_type")
        in {
            SoftwareUnitType.CODE_SEGMENT.value,
            SoftwareUnitType.SEMANTIC_BLOCK.value,
        }
    ]
    parsed_files = sum(not unit.raw_data.get("parse_error") for unit in files)
    syntax_errors = sum(
        bool(unit.raw_data.get("parse_error"))
        or bool(unit.raw_data.get("language_facts", {}).get("has_syntax_errors"))
        for unit in files
    )
    report = getattr(semantic_code, "last_build_report", None)
    source_files = int(getattr(report, "source_files", len(files)))
    unsupported = int(getattr(report, "unsupported_files", 0))

    invalid_spans = 0
    mismatches = 0
    nonempty = 0
    signatures = 0
    valid_spans = 0
    calls_recorded = 0
    for function in functions:
        source_code = str(function.raw_data.get("source_code", ""))
        nonempty += bool(source_code)
        signatures += bool(function.raw_data.get("signature"))
        calls_recorded += len(set(function.raw_data.get("calls", [])))
        path = root / str(function.raw_data.get("file_path", ""))
        start = int(function.raw_data.get("start_line", 0) or 0)
        end = int(function.raw_data.get("end_line", 0) or 0)
        if not path.is_file() or start <= 0 or end < start:
            invalid_spans += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if end > len(lines):
            invalid_spans += 1
            continue
        valid_spans += 1
        actual = "\n".join(lines[start - 1 : end])
        if source_code and _normalized_text(actual) != _normalized_text(source_code):
            mismatches += 1

    function_mismatches = mismatches
    segment_valid_spans = 0
    segment_mismatches = 0
    for segment in segments:
        path = root / str(
            segment.raw_data.get("file_path")
            or segment.metadata.get("file_path", "")
        )
        start = int(segment.raw_data.get("start_line", 0) or 0)
        end = int(segment.raw_data.get("end_line", 0) or 0)
        if not path.is_file() or start <= 0 or end < start:
            invalid_spans += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if end > len(lines):
            invalid_spans += 1
            continue
        segment_valid_spans += 1
        actual = "\n".join(lines[start - 1 : end])
        snippet = str(segment.raw_data.get("code_snippet", ""))
        if snippet and _normalized_text(actual) != _normalized_text(snippet):
            mismatches += 1
            segment_mismatches += 1

    file_symbol_counts = Counter()
    call_edges = 0
    dangling = 0
    known_uids = {unit.uid for unit in units}
    for edge_index in graph.rx_graph.edge_indices():
        source_index, target_index = graph.rx_graph.get_edge_endpoints_by_index(
            edge_index
        )
        source = graph._index_to_uid.get(source_index, "")
        target = graph._index_to_uid.get(target_index, "")
        data = graph.rx_graph.get_edge_data_by_index(edge_index) or {}
        relation = data.get("type")
        if source not in known_uids or target not in known_uids:
            dangling += 1
        if relation in {
            SoftwareRelation.CONTAINS.value,
            SoftwareRelation.HAS_METHOD.value,
        }:
            if source.startswith("file:") and target.startswith(
                ("function:", "class:")
            ):
                file_symbol_counts[source] += 1
        if relation == SoftwareRelation.CALLS.value:
            call_edges += 1

    function_count = len(functions)
    return RepositoryQualityMetrics(
        source_files=source_files,
        parsed_files=parsed_files,
        parse_success_rate=_rate(parsed_files, source_files),
        unsupported_file_rate=_rate(unsupported, source_files),
        syntax_error_rate=_rate(syntax_errors, source_files),
        files_with_zero_symbols=sum(
            file_symbol_counts[f"file:{unit.raw_data.get('path')}"] == 0
            for unit in files
        ),
        functions=function_count,
        code_segments=len(segments),
        functions_with_valid_source_span_rate=_rate(valid_spans, function_count),
        functions_with_nonempty_source_code_rate=_rate(nonempty, function_count),
        functions_with_signature_rate=_rate(signatures, function_count),
        source_code_match_rate=_rate(
            valid_spans - function_mismatches,
            valid_spans,
        ),
        code_segments_with_valid_source_span_rate=_rate(
            segment_valid_spans, len(segments)
        ),
        code_segment_source_match_rate=_rate(
            segment_valid_spans - segment_mismatches,
            segment_valid_spans,
        ),
        calls_recorded=calls_recorded,
        calls_resolved=call_edges,
        calls_resolved_rate=_rate(call_edges, calls_recorded),
        dangling_edge_count=dangling,
        duplicate_uid_count=len(units) - len({unit.uid for unit in units}),
        invalid_source_span_count=invalid_spans,
        source_code_mismatch_count=mismatches,
    )


def relationship_samples(
    semantic_graph, *, samples_per_type: int = 5
) -> dict[str, list[str]]:
    """Return deterministic human-readable relationship examples."""
    grouped: dict[str, list[str]] = {}
    rows: list[tuple[str, str, str]] = []
    for edge_index in semantic_graph.rx_graph.edge_indices():
        source_index, target_index = (
            semantic_graph.rx_graph.get_edge_endpoints_by_index(edge_index)
        )
        data = semantic_graph.rx_graph.get_edge_data_by_index(edge_index) or {}
        rows.append(
            (
                str(data.get("type", "")),
                semantic_graph._index_to_uid.get(source_index, ""),
                semantic_graph._index_to_uid.get(target_index, ""),
            )
        )
    for relation, source, target in sorted(rows):
        bucket = grouped.setdefault(relation, [])
        if len(bucket) < samples_per_type:
            bucket.append(f"{source} --{relation}--> {target}")
    return grouped


def spot_check_samples(semantic_code, repo_path: str | Path) -> dict[str, Any]:
    """Return deterministic source-backed examples suitable for manual review."""
    root = Path(repo_path)
    units = semantic_code.semantic_graph.semantic_map.get_all_units()
    functions = _sample_evenly(
        sorted(
            (
                unit
                for unit in units
                if unit.metadata.get("software_type") == SoftwareUnitType.FUNCTION.value
            ),
            key=lambda unit: (str(unit.raw_data.get("file_path", "")), unit.uid),
        ),
        10,
    )
    classes = _sample_evenly(
        sorted(
            (
                unit
                for unit in units
                if unit.metadata.get("software_type") == SoftwareUnitType.CLASS.value
            ),
            key=lambda unit: (str(unit.raw_data.get("file_path", "")), unit.uid),
        ),
        5,
    )
    function_rows = []
    for unit in functions:
        path = root / str(unit.raw_data.get("file_path", ""))
        start = int(unit.raw_data.get("start_line", 0) or 0)
        end = int(unit.raw_data.get("end_line", 0) or 0)
        source_match = False
        if path.is_file() and start > 0 and end >= start:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if end <= len(lines):
                actual = "\n".join(lines[start - 1 : end])
                source_match = _normalized_text(actual) == _normalized_text(
                    str(unit.raw_data.get("source_code", ""))
                )
        function_rows.append(
            {
                "uid": unit.uid,
                "qualified_name": unit.raw_data.get("qualified_name"),
                "file_path": unit.raw_data.get("file_path"),
                "span": [start, end],
                "signature": unit.raw_data.get("signature"),
                "calls": unit.raw_data.get("calls", []),
                "exceptions": unit.raw_data.get("exceptions", []),
                "source_match": source_match,
                "content_hash": unit.raw_data.get("content_hash"),
            }
        )
    class_rows = [
        {
            "uid": unit.uid,
            "name": unit.raw_data.get("name"),
            "file_path": unit.raw_data.get("file_path"),
            "methods": unit.raw_data.get("methods", []),
            "base_classes": unit.raw_data.get("base_classes", []),
            "interfaces": unit.raw_data.get("interfaces", []),
            "kind": unit.raw_data.get("kind"),
        }
        for unit in classes
    ]
    relationship_rows = relationship_samples(
        semantic_code.semantic_graph,
        samples_per_type=10,
    )
    return {
        "functions": function_rows,
        "classes": class_rows,
        "calls": relationship_rows.get(SoftwareRelation.CALLS.value, [])[:10],
        "imports": relationship_rows.get(SoftwareRelation.IMPORTS.value, [])[:5],
        "inheritance": (
            relationship_rows.get(SoftwareRelation.INHERITS_FROM.value, [])
            + relationship_rows.get(SoftwareRelation.IMPLEMENTS.value, [])
        )[:5],
    }


def query_snapshot(semantic_code, query: str | None = None) -> dict[str, Any]:
    """Capture graph and retrieval query UIDs for persistence comparison."""
    functions = sorted(
        (
            unit
            for unit in semantic_code.semantic_graph.semantic_map.get_all_units()
            if unit.metadata.get("software_type") == SoftwareUnitType.FUNCTION.value
            and not unit.raw_data.get("is_declaration")
        ),
        key=lambda unit: unit.uid,
    )
    selected = next(
        (unit for unit in functions if semantic_code.get_callees(unit.uid)),
        functions[0] if functions else None,
    )
    if selected is None:
        return {"function_uid": None, "search": []}
    search_query = query or str(
        selected.raw_data.get("semantic_summary")
        or selected.raw_data.get("signature")
        or selected.raw_data.get("name")
    )
    return {
        "function_uid": selected.uid,
        "get_function": getattr(semantic_code.get_function(selected.uid), "uid", None),
        "callers": [unit.uid for unit in semantic_code.get_callers(selected.uid)],
        "callees": [unit.uid for unit in semantic_code.get_callees(selected.uid)],
        "traverse_calls": [
            unit.uid
            for unit in semantic_code.traverse(
                selected.uid,
                SoftwareRelation.CALLS,
                direction="outgoing",
            )
        ],
        "history": [unit.uid for unit in semantic_code.get_history(selected.uid)],
        "file_memory": getattr(
            semantic_code.get_file_memory(str(selected.raw_data.get("file_path", ""))),
            "uid",
            None,
        ),
        "code_segments": [
            unit.uid for unit in semantic_code.get_function_segments(selected.uid)
        ],
        "search": [unit.uid for unit in semantic_code.search(search_query, top_k=5)],
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 1.0


def _normalized_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _sample_evenly(values: list[Any], count: int) -> list[Any]:
    if len(values) <= count:
        return values
    if count <= 1:
        return values[:count]
    indices = {round(index * (len(values) - 1) / (count - 1)) for index in range(count)}
    return [values[index] for index in sorted(indices)]


__all__ = [
    "RepositoryQualityMetrics",
    "RepositorySnapshotFingerprint",
    "evaluate_repository_quality",
    "query_snapshot",
    "relationship_samples",
    "spot_check_samples",
]
