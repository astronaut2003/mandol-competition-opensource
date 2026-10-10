"""Software-domain schemas that serialize to standard Mandol memory units."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

from mandol_substrate.representation import MemoryUnit


class SoftwareUnitType(str, Enum):
    """Stable software entity kinds stored in ``MemoryUnit.metadata``."""

    REPOSITORY = "repository"
    DIRECTORY = "directory"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    CODE_SEGMENT = "code_segment"
    # Read compatibility for checkpoints produced before builder version 4.
    SEMANTIC_BLOCK = "semantic_block"
    ISSUE = "issue"
    COMMIT = "commit"
    CONFIG = "config"
    LOG = "log"
    EXECUTION_TRACE = "execution_trace"
    EVOLUTION = "evolution"
    FILE_MEMORY = "file_memory"
    SYMBOL_MEMORY = "symbol_memory"
    REPAIR_MEMORY = "repair_memory"
    MODULE_MEMORY = "module_memory"
    ARCHITECTURE_MEMORY = "architecture_memory"
    ENGINEERING_EPISODE = "engineering_episode"


def _require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _text_content(*parts: Any) -> str:
    lines: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            value = ", ".join(str(item) for item in part if str(item).strip())
        else:
            value = str(part).strip()
        if value:
            lines.append(value)
    return "\n".join(lines)


def _memory_unit(
    uid: str,
    software_type: SoftwareUnitType,
    raw_data: dict[str, Any],
    text_content: str,
    metadata: dict[str, Any] | None,
) -> MemoryUnit:
    payload = {"software_type": software_type.value, **raw_data}
    payload["text_content"] = text_content or f"{software_type.value} {uid}"
    unit_metadata = dict(metadata or {})
    unit_metadata["software_type"] = software_type.value
    return MemoryUnit(uid=uid, raw_data=payload, metadata=unit_metadata)


@dataclass
class RepositorySchema:
    name: str
    path: str = ""
    url: str = ""
    default_branch: str = "main"
    description: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    revision: str = "HEAD"
    language_statistics: dict[str, int] = field(default_factory=dict)
    file_count: int = 0
    source_file_count: int = 0
    test_file_count: int = 0

    def to_memory_unit(self) -> MemoryUnit:
        name = _require_text(self.name, "name")
        uid = self.uid or f"repo:{name}"
        return _memory_unit(
            uid,
            SoftwareUnitType.REPOSITORY,
            {
                "name": name,
                "path": self.path,
                "url": self.url,
                "default_branch": self.default_branch,
                "description": self.description,
                "revision": self.revision,
                "language_statistics": dict(self.language_statistics),
                "file_count": int(self.file_count),
                "source_file_count": int(self.source_file_count),
                "test_file_count": int(self.test_file_count),
            },
            _text_content(name, self.path, self.url, self.description),
            self.metadata,
        )


@dataclass
class DirectorySchema:
    path: str
    name: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        path = _require_text(self.path, "path")
        name = self.name or path.rstrip("/").rsplit("/", 1)[-1]
        return _memory_unit(
            self.uid or f"directory:{path}",
            SoftwareUnitType.DIRECTORY,
            {"path": path, "name": name},
            _text_content(path, name),
            self.metadata,
        )


@dataclass
class FileSchema:
    path: str
    language: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    source_loc: int = 0
    is_test: bool = False
    is_config: bool = False
    module_name: str = ""
    parse_error: str = ""
    language_facts: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        path = _require_text(self.path, "path")
        return _memory_unit(
            self.uid or f"file:{path}",
            SoftwareUnitType.FILE,
            {
                "path": path,
                "language": self.language,
                "imports": list(self.imports),
                "exports": list(self.exports),
                "semantic_summary": self.semantic_summary,
                "content_hash": self.content_hash,
                "source_loc": int(self.source_loc),
                "is_test": bool(self.is_test),
                "is_config": bool(self.is_config),
                "module_name": self.module_name,
                "parse_error": self.parse_error,
                "language_facts": dict(self.language_facts),
            },
            _text_content(
                path, self.language, self.imports, self.exports, self.semantic_summary
            ),
            self.metadata,
        )


@dataclass
class ClassSchema:
    name: str
    qualified_name: str
    base_classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    content_hash: str = ""
    language: str = ""
    kind: str = "class"
    interfaces: list[str] = field(default_factory=list)
    docstring: str = ""
    symbol_identity: str = ""

    def to_memory_unit(self) -> MemoryUnit:
        name = _require_text(self.name, "name")
        qualified_name = _require_text(self.qualified_name, "qualified_name")
        return _memory_unit(
            self.uid or f"class:{qualified_name}",
            SoftwareUnitType.CLASS,
            {
                "name": name,
                "qualified_name": qualified_name,
                "base_classes": list(self.base_classes),
                "methods": list(self.methods),
                "semantic_summary": self.semantic_summary,
                "file_path": self.file_path,
                "start_line": int(self.start_line),
                "end_line": int(self.end_line),
                "content_hash": self.content_hash,
                "language": self.language,
                "kind": self.kind,
                "interfaces": list(self.interfaces),
                "docstring": self.docstring,
                "symbol_identity": self.symbol_identity or qualified_name,
            },
            _text_content(
                qualified_name,
                name,
                self.base_classes,
                self.methods,
                self.semantic_summary,
            ),
            self.metadata,
        )


@dataclass
class FunctionSchema:
    name: str
    signature: str
    language: str
    file_path: str
    qualified_name: str
    docstring: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    call_sites: list[dict[str, Any]] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_code: str = ""
    start_line: int = 0
    end_line: int = 0
    content_hash: str = ""
    exceptions: list[str] = field(default_factory=list)
    ast_summary: dict[str, Any] | None = None
    config_references: list[str] = field(default_factory=list)
    symbol_identity: str = ""
    is_declaration: bool = False

    def to_memory_unit(self) -> MemoryUnit:
        name = _require_text(self.name, "name")
        qualified_name = _require_text(self.qualified_name, "qualified_name")
        return _memory_unit(
            self.uid or f"function:{qualified_name}",
            SoftwareUnitType.FUNCTION,
            {
                "name": name,
                "qualified_name": qualified_name,
                "signature": self.signature,
                "language": self.language,
                "file_path": self.file_path,
                "docstring": self.docstring,
                "parameters": list(self.parameters),
                "return_type": self.return_type,
                "reads": list(self.reads),
                "writes": list(self.writes),
                "calls": list(self.calls),
                "call_sites": [dict(call_site) for call_site in self.call_sites],
                "side_effects": list(self.side_effects),
                "preconditions": list(self.preconditions),
                "postconditions": list(self.postconditions),
                "semantic_summary": self.semantic_summary,
                "source_code": self.source_code,
                "start_line": int(self.start_line),
                "end_line": int(self.end_line),
                "content_hash": self.content_hash,
                "exceptions": list(self.exceptions),
                "ast_summary": dict(self.ast_summary or {}),
                "config_references": list(self.config_references),
                "symbol_identity": self.symbol_identity or qualified_name,
                "is_declaration": bool(self.is_declaration),
            },
            _text_content(
                qualified_name,
                self.signature,
                self.docstring,
                self.semantic_summary,
            ),
            self.metadata,
        )


@dataclass
class CodeSegmentSchema:
    """A bounded, source-backed code region owned by a software entity.

    ``block_id`` and ``statement_type`` are accepted only as deprecated input
    compatibility. New serialization always uses segment terminology and the
    ``segment:<parent_uid>:<segment_id>`` UID namespace.
    """

    segment_id: str = ""
    segment_type: str = ""
    language: str = ""
    file_path: str = ""
    parent_uid: str = ""
    function_uid: str = ""
    class_uid: str = ""
    code_snippet: str = ""
    block_id: str = ""
    statement_type: str = ""
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    control_transfer: str = ""
    semantic_summary: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    start_line: int = 0
    end_line: int = 0
    start_column: int | None = None
    end_column: int | None = None
    content_hash: str = ""

    def to_memory_unit(self) -> MemoryUnit:
        segment_id = _require_text(self.segment_id or self.block_id, "segment_id")
        parent_uid = _require_text(
            self.parent_uid or self.function_uid or self.class_uid,
            "parent_uid",
        )
        segment_type = self.segment_type or self.statement_type or "statement_group"
        return _memory_unit(
            self.uid or f"segment:{parent_uid}:{segment_id}",
            SoftwareUnitType.CODE_SEGMENT,
            {
                "segment_id": segment_id,
                "segment_type": segment_type,
                "language": self.language,
                "file_path": self.file_path,
                "parent_uid": parent_uid,
                "function_uid": self.function_uid,
                "class_uid": self.class_uid,
                "code_snippet": self.code_snippet,
                "reads": list(self.reads),
                "writes": list(self.writes),
                "control_transfer": self.control_transfer,
                "semantic_summary": self.semantic_summary,
                "start_line": int(self.start_line),
                "end_line": int(self.end_line),
                "start_column": self.start_column,
                "end_column": self.end_column,
                "content_hash": self.content_hash,
            },
            _text_content(
                self.code_snippet, segment_type, self.semantic_summary
            ),
            self.metadata,
        )


@dataclass
class IssueSchema:
    issue_id: str | int
    title: str
    body: str = ""
    state: str = "open"
    labels: list[str] = field(default_factory=list)
    created_at: str = ""
    closed_at: str | None = None
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        issue_id = _require_text(str(self.issue_id), "issue_id")
        return _memory_unit(
            self.uid or f"issue:{issue_id}",
            SoftwareUnitType.ISSUE,
            {
                "issue_id": issue_id,
                "title": self.title,
                "body": self.body,
                "state": self.state,
                "labels": list(self.labels),
                "created_at": self.created_at,
                "closed_at": self.closed_at,
            },
            _text_content(self.title, self.body, self.labels, self.state),
            self.metadata,
        )


@dataclass
class CommitSchema:
    commit_hash: str
    message: str
    author: str = ""
    timestamp: str = ""
    changed_files: list[str] = field(default_factory=list)
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        commit_hash = _require_text(self.commit_hash, "commit_hash")
        return _memory_unit(
            self.uid or f"commit:{commit_hash}",
            SoftwareUnitType.COMMIT,
            {
                "commit_hash": commit_hash,
                "message": self.message,
                "author": self.author,
                "timestamp": self.timestamp,
                "changed_files": list(self.changed_files),
            },
            _text_content(commit_hash, self.message, self.author, self.changed_files),
            self.metadata,
        )


@dataclass
class ConfigSchema:
    key: str
    value: Any
    source_file: str = ""
    scope: str = ""
    semantic_summary: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        key = _require_text(self.key, "key")
        return _memory_unit(
            self.uid or f"config:{key}",
            SoftwareUnitType.CONFIG,
            {
                "key": key,
                "value": self.value,
                "source_file": self.source_file,
                "scope": self.scope,
                "semantic_summary": self.semantic_summary,
            },
            _text_content(
                key, self.value, self.source_file, self.scope, self.semantic_summary
            ),
            self.metadata,
        )


@dataclass
class LogSchema:
    level: str
    message: str
    timestamp: str
    service: str = ""
    stack_trace: str = ""
    log_id: str | None = None
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        timestamp = _require_text(self.timestamp, "timestamp")
        log_key = self.log_id or f"{self.service or 'unknown'}:{timestamp}"
        return _memory_unit(
            self.uid or f"log:{log_key}",
            SoftwareUnitType.LOG,
            {
                "level": self.level,
                "message": self.message,
                "timestamp": timestamp,
                "service": self.service,
                "stack_trace": self.stack_trace,
            },
            _text_content(self.level, self.message, self.service, self.stack_trace),
            self.metadata,
        )


@dataclass
class ExecutionTraceSchema:
    trace_id: str
    entry_function: str
    call_sequence: list[str] = field(default_factory=list)
    error_info: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        trace_id = _require_text(self.trace_id, "trace_id")
        return _memory_unit(
            self.uid or f"trace:{trace_id}",
            SoftwareUnitType.EXECUTION_TRACE,
            {
                "trace_id": trace_id,
                "entry_function": self.entry_function,
                "call_sequence": list(self.call_sequence),
                "error_info": self.error_info,
                "metrics": dict(self.metrics),
            },
            _text_content(
                trace_id, self.entry_function, self.call_sequence, self.error_info
            ),
            self.metadata,
        )


@dataclass
class EvolutionSchema:
    subject_uid: str
    change_type: str
    timestamp: str
    diff_summary: str = ""
    source_evidence: list[str] = field(default_factory=list)
    prior_evolution_uid: str | None = None
    impacted_units: list[str] = field(default_factory=list)
    confidence: float = 1.0
    evolution_id: str | None = None
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        subject_uid = _require_text(self.subject_uid, "subject_uid")
        timestamp = _require_text(self.timestamp, "timestamp")
        subject_key = subject_uid.split(":", 1)[-1]
        if self.evolution_id:
            evolution_id = self.evolution_id
        elif self.source_evidence:
            evolution_id = self.source_evidence[0].rsplit(":", 1)[-1]
        else:
            identity = "|".join(
                (subject_uid, self.change_type, timestamp, self.diff_summary)
            )
            evolution_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        return _memory_unit(
            self.uid or f"evolution:{subject_key}:{evolution_id}",
            SoftwareUnitType.EVOLUTION,
            {
                "subject_uid": subject_uid,
                "change_type": self.change_type,
                "timestamp": timestamp,
                "diff_summary": self.diff_summary,
                "source_evidence": list(self.source_evidence),
                "prior_evolution_uid": self.prior_evolution_uid,
                "impacted_units": list(self.impacted_units),
                "confidence": float(self.confidence),
            },
            _text_content(
                subject_uid, self.change_type, self.diff_summary, self.impacted_units
            ),
            self.metadata,
        )


@dataclass
class SymbolMemorySchema:
    subject_uid: str
    responsibility: str = ""
    contract: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    structural_context: dict[str, Any] = field(default_factory=dict)
    evolution_context: dict[str, Any] = field(default_factory=dict)
    semantic_summary: str = ""
    derived_from: list[str] = field(default_factory=list)
    confidence: float = 1.0
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        subject_uid = _require_text(self.subject_uid, "subject_uid")
        subject_key = subject_uid.split(":", 1)[-1]
        return _memory_unit(
            self.uid or f"symbol_memory:{subject_key}",
            SoftwareUnitType.SYMBOL_MEMORY,
            {
                "subject_uid": subject_uid,
                "responsibility": self.responsibility,
                "contract": dict(self.contract),
                "behavior": dict(self.behavior),
                "structural_context": dict(self.structural_context),
                "evolution_context": dict(self.evolution_context),
                "semantic_summary": self.semantic_summary,
                "derived_from": list(self.derived_from),
                "confidence": float(self.confidence),
            },
            _text_content(
                subject_uid,
                self.responsibility,
                self.contract,
                self.behavior,
                self.structural_context,
                self.evolution_context,
                self.semantic_summary,
            ),
            self.metadata,
        )


@dataclass
class RepairMemorySchema:
    symptom: str
    root_cause: str = ""
    affected_symbols: list[str] = field(default_factory=list)
    repair_strategy: str = ""
    changed_symbols: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    lessons: list[str] = field(default_factory=list)
    related_issue: str | None = None
    related_commit: str | None = None
    semantic_summary: str = ""
    derived_from: list[str] = field(default_factory=list)
    confidence: float = 0.7
    repair_id: str | None = None
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional task-experience fields; older payloads remain readable. Failed
    # attempts stay attached to the repair instead of creating a parallel type.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str = ""
    negative_evidence: list[str] = field(default_factory=list)
    violated_contract: str = ""
    corrective_strategy: str = ""
    applicability: str = ""
    outcome: str = "unknown"
    verification_level: str = "INFERRED"

    def to_memory_unit(self) -> MemoryUnit:
        symptom = _require_text(self.symptom, "symptom")
        repair_id = self.repair_id
        if not repair_id and self.related_commit:
            repair_id = self.related_commit.rsplit(":", 1)[-1]
        if not repair_id:
            repair_id = hashlib.sha256(symptom.encode("utf-8")).hexdigest()[:12]
        return _memory_unit(
            self.uid or f"repair_memory:{repair_id}",
            SoftwareUnitType.REPAIR_MEMORY,
            {
                "symptom": symptom,
                "root_cause": self.root_cause,
                "affected_symbols": list(self.affected_symbols),
                "repair_strategy": self.repair_strategy,
                "changed_symbols": list(self.changed_symbols),
                "validation": list(self.validation),
                "constraints": list(self.constraints),
                "lessons": list(self.lessons),
                "related_issue": self.related_issue,
                "related_commit": self.related_commit,
                "semantic_summary": self.semantic_summary,
                "derived_from": list(self.derived_from),
                "confidence": float(self.confidence),
                "attempts": list(self.attempts),
                "failure_reason": self.failure_reason,
                "negative_evidence": list(self.negative_evidence),
                "violated_contract": self.violated_contract,
                "corrective_strategy": self.corrective_strategy,
                "applicability": self.applicability,
                "outcome": self.outcome,
                "verification_level": self.verification_level,
            },
            _text_content(
                symptom,
                self.root_cause,
                self.repair_strategy,
                self.validation,
                self.lessons,
                self.semantic_summary,
                self.attempts,
                self.failure_reason,
                self.violated_contract,
                self.corrective_strategy,
                self.applicability,
            ),
            self.metadata,
        )


@dataclass
class FileMemorySchema:
    file_uid: str
    path: str
    language: str = ""
    responsibility: str = ""
    semantic_summary: str = ""
    key_symbols: list[str] = field(default_factory=list)
    public_interfaces: list[str] = field(default_factory=list)
    important_classes: list[str] = field(default_factory=list)
    important_functions: list[str] = field(default_factory=list)
    important_segments: list[str] = field(default_factory=list)
    important_dependencies: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    important_state: list[str] = field(default_factory=list)
    config_references: list[str] = field(default_factory=list)
    important_control_flow: list[str] = field(default_factory=list)
    important_data_flow: list[str] = field(default_factory=list)
    known_risks: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    confidence: float = 0.8
    source_hash: str = ""
    content_hash: str = ""
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_memory_unit(self) -> MemoryUnit:
        file_uid = _require_text(self.file_uid, "file_uid")
        path = _require_text(self.path, "path")
        return _memory_unit(
            self.uid or f"file_memory:{path}",
            SoftwareUnitType.FILE_MEMORY,
            {
                "file_uid": file_uid,
                "path": path,
                "language": self.language,
                "responsibility": self.responsibility,
                "semantic_summary": self.semantic_summary,
                "key_symbols": list(self.key_symbols),
                "public_interfaces": list(self.public_interfaces),
                "important_classes": list(self.important_classes),
                "important_functions": list(self.important_functions),
                "important_segments": list(self.important_segments),
                "important_dependencies": list(self.important_dependencies),
                "imports": list(self.imports),
                "exports": list(self.exports),
                "important_state": list(self.important_state),
                "config_references": list(self.config_references),
                "important_control_flow": list(self.important_control_flow),
                "important_data_flow": list(self.important_data_flow),
                "known_risks": list(self.known_risks),
                "side_effects": list(self.side_effects),
                "derived_from": list(self.derived_from),
                "confidence": float(self.confidence),
                "source_hash": self.source_hash,
                "content_hash": self.content_hash or self.source_hash,
            },
            _text_content(
                path,
                self.responsibility,
                self.semantic_summary,
                self.public_interfaces,
                self.important_dependencies,
                self.important_control_flow,
                self.known_risks,
            ),
            self.metadata,
        )


@dataclass
class ModuleMemorySchema:
    module_name: str
    responsibilities: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    important_dependencies: list[str] = field(default_factory=list)
    known_risks: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    evolution_summary: str = ""
    derived_from: list[str] = field(default_factory=list)
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    public_interfaces: list[str] = field(default_factory=list)
    important_configs: list[str] = field(default_factory=list)
    important_tests: list[str] = field(default_factory=list)
    recent_important_changes: list[str] = field(default_factory=list)

    def to_memory_unit(self) -> MemoryUnit:
        module_name = _require_text(self.module_name, "module_name")
        return _memory_unit(
            self.uid or f"module_memory:{module_name}",
            SoftwareUnitType.MODULE_MEMORY,
            {
                "module_name": module_name,
                "responsibilities": list(self.responsibilities),
                "key_entities": list(self.key_entities),
                "important_dependencies": list(self.important_dependencies),
                "public_interfaces": list(self.public_interfaces),
                "important_configs": list(self.important_configs),
                "important_tests": list(self.important_tests),
                "known_risks": list(self.known_risks),
                "recent_important_changes": list(self.recent_important_changes),
                "semantic_summary": self.semantic_summary,
                "evolution_summary": self.evolution_summary,
                "derived_from": list(self.derived_from),
            },
            _text_content(
                module_name,
                self.responsibilities,
                self.important_dependencies,
                self.public_interfaces,
                self.important_configs,
                self.important_tests,
                self.known_risks,
                self.recent_important_changes,
                self.semantic_summary,
                self.evolution_summary,
            ),
            self.metadata,
        )


@dataclass
class ArchitectureMemorySchema:
    scope: str
    components: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    critical_paths: list[str] = field(default_factory=list)
    semantic_summary: str = ""
    derived_from: list[str] = field(default_factory=list)
    uid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    public_interfaces: list[str] = field(default_factory=list)
    important_configurations: list[str] = field(default_factory=list)
    important_tests: list[str] = field(default_factory=list)
    supporting_infrastructure: list[str] = field(default_factory=list)

    def to_memory_unit(self) -> MemoryUnit:
        scope = _require_text(self.scope, "scope")
        return _memory_unit(
            self.uid or f"architecture_memory:{scope}",
            SoftwareUnitType.ARCHITECTURE_MEMORY,
            {
                "scope": scope,
                "components": list(self.components),
                "responsibilities": list(self.responsibilities),
                "dependencies": list(self.dependencies),
                "critical_paths": list(self.critical_paths),
                "public_interfaces": list(self.public_interfaces),
                "important_configurations": list(self.important_configurations),
                "important_tests": list(self.important_tests),
                "supporting_infrastructure": list(self.supporting_infrastructure),
                "semantic_summary": self.semantic_summary,
                "derived_from": list(self.derived_from),
            },
            _text_content(
                scope,
                self.components,
                self.responsibilities,
                self.dependencies,
                self.critical_paths,
                self.public_interfaces,
                self.important_configurations,
                self.important_tests,
                self.supporting_infrastructure,
                self.semantic_summary,
            ),
            self.metadata,
        )


SoftwareSchema = Union[
    RepositorySchema,
    DirectorySchema,
    FileSchema,
    ClassSchema,
    FunctionSchema,
    CodeSegmentSchema,
    IssueSchema,
    CommitSchema,
    ConfigSchema,
    LogSchema,
    ExecutionTraceSchema,
    EvolutionSchema,
    SymbolMemorySchema,
    RepairMemorySchema,
    FileMemorySchema,
    ModuleMemorySchema,
    ArchitectureMemorySchema,
]

# Concise domain names are aliases; old block names remain import-compatible.
CodeSegment = CodeSegmentSchema
SemanticBlockSchema = CodeSegmentSchema
SemanticBlock = CodeSegment
EvolutionNode = EvolutionSchema
SymbolMemory = SymbolMemorySchema
RepairMemory = RepairMemorySchema
FileMemory = FileMemorySchema
ModuleMemory = ModuleMemorySchema
ArchitectureMemory = ArchitectureMemorySchema


__all__ = [
    "ArchitectureMemory",
    "ArchitectureMemorySchema",
    "ClassSchema",
    "CodeSegment",
    "CodeSegmentSchema",
    "CommitSchema",
    "ConfigSchema",
    "DirectorySchema",
    "EvolutionNode",
    "EvolutionSchema",
    "ExecutionTraceSchema",
    "FileSchema",
    "FileMemory",
    "FileMemorySchema",
    "FunctionSchema",
    "IssueSchema",
    "LogSchema",
    "ModuleMemory",
    "ModuleMemorySchema",
    "RepairMemory",
    "RepairMemorySchema",
    "RepositorySchema",
    "SemanticBlock",
    "SemanticBlockSchema",
    "SoftwareSchema",
    "SoftwareUnitType",
    "SymbolMemory",
    "SymbolMemorySchema",
]
