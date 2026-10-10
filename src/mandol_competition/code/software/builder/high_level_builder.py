"""Rule-based agglomeration of software facts into high-level memories."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
)

from mandol_substrate.representation import MemoryUnit
from ..relations import SoftwareRelation
from ..schemas import (
    ArchitectureMemorySchema,
    FileMemorySchema,
    ModuleMemorySchema,
    RepairMemorySchema,
    SoftwareUnitType,
    SymbolMemorySchema,
)
from ..spaces import SoftwareSpace
from .cache import BuilderCache
from .git_analyzer import GitCommitFacts
from .relation_builder import SoftwareEdge

if TYPE_CHECKING:
    from ..facade import SemanticCode


SemanticEnricher = Callable[[str, dict[str, Any]], str | dict[str, Any] | None]


@dataclass
class HighLevelBuildResult:
    units: list[MemoryUnit] = field(default_factory=list)
    spaces: list[list[str]] = field(default_factory=list)
    edges: list[SoftwareEdge] = field(default_factory=list)
    rebuilt_symbol_uids: list[str] = field(default_factory=list)
    rebuilt_module_names: list[str] = field(default_factory=list)
    rebuilt_file_paths: list[str] = field(default_factory=list)
    semantic_enrichment_calls: int = 0
    semantic_enrichment_cache_hits: int = 0
    semantic_enrichment_failures: int = 0
    file_memory_build_time: float = 0.0


class HighLevelMemoryBuilder:
    """Agglomerate existing graph evidence without requiring an LLM."""

    def __init__(
        self,
        semantic_code: SemanticCode,
        repo_uid: str,
        *,
        enricher: SemanticEnricher | None = None,
        cache: BuilderCache | None = None,
        repo_path: str | Path | None = None,
        max_file_context_chars: int = 40_000,
        enrichment_config_key: str = "",
        enrich_symbols: bool = False,
        enrich_repairs: bool = False,
    ) -> None:
        self.semantic_code = semantic_code
        self.graph = semantic_code.semantic_graph
        self.repo_uid = repo_uid
        self.enricher = enricher
        self.cache = cache
        self.repo_path = Path(repo_path).resolve() if repo_path else None
        self.max_file_context_chars = max(2_000, int(max_file_context_chars))
        self.enrichment_config_key = enrichment_config_key
        self.enrich_symbols = enrich_symbols
        self.enrich_repairs = enrich_repairs
        self._related_cache: dict[
            tuple[str, SoftwareRelation, str], list[MemoryUnit]
        ] = {}
        self._history_by_subject: dict[str, list[MemoryUnit]] = {}
        self._commits_cache: dict[str, list[MemoryUnit]] = {}
        self._issues_cache: dict[str, list[MemoryUnit]] = {}
        enricher_identity = "none"
        if enricher is not None:
            enricher_identity = ".".join(
                filter(
                    None,
                    (
                        getattr(enricher, "__module__", ""),
                        getattr(enricher, "__qualname__", type(enricher).__qualname__),
                    ),
                )
            )
        self.policy_hash = self._hash(
            {
                "version": 2,
                "max_file_context_chars": self.max_file_context_chars,
                "enricher": enricher_identity,
                "config": enrichment_config_key,
            }
        )

    def build(
        self,
        *,
        commit_facts: Sequence[GitCommitFacts],
        affected_symbol_uids: Iterable[str] | None = None,
        affected_modules: Iterable[str] | None = None,
        changed_file_paths: Iterable[str] | None = None,
        force_all: bool = False,
    ) -> HighLevelBuildResult:
        result = HighLevelBuildResult()
        repo_units = self._repo_units()
        self._prepare_history_index(repo_units)
        changed_paths = set(changed_file_paths or ())
        file_units = [
            unit
            for unit in repo_units
            if self._type(unit) is SoftwareUnitType.FILE
            and unit.metadata.get("source_kind") == "source"
        ]

        members_by_file: dict[str, list[MemoryUnit]] = defaultdict(list)
        members_by_module: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in repo_units:
            if self._type(unit) in {
                SoftwareUnitType.CLASS,
                SoftwareUnitType.FUNCTION,
                SoftwareUnitType.CODE_SEGMENT,
                SoftwareUnitType.SEMANTIC_BLOCK,
            }:
                path = str(
                    unit.raw_data.get("file_path")
                    or unit.metadata.get("file_path")
                    or ""
                )
                if path:
                    members_by_file[path].append(unit)
            module = self._module(unit)
            if module:
                members_by_module[module].append(unit)
        for members in members_by_file.values():
            members.sort(
                key=lambda unit: (
                    int(unit.raw_data.get("start_line", 0) or 0),
                    unit.uid,
                )
            )

        file_started = time.perf_counter()
        for file_unit in sorted(file_units, key=lambda unit: unit.uid):
            path = str(file_unit.raw_data.get("path", ""))
            source_hash = str(file_unit.raw_data.get("content_hash", ""))
            memory_uid = f"file_memory:{path}"
            cached = self.cache.high_level_hash(path) if self.cache else None
            existing = self.graph.semantic_map.get_unit(memory_uid)
            reusable = bool(
                not force_all
                and path not in changed_paths
                and existing is not None
                and cached
                and cached.get("source_hash") == source_hash
                and cached.get("policy_hash") == self.policy_hash
            )
            if reusable:
                if self.enricher is not None:
                    result.semantic_enrichment_cache_hits += 1
                continue
            memory, input_hash, enrichment_status = self._file_memory(
                file_unit, members_by_file.get(path, [])
            )
            self._append(result, memory, SoftwareSpace.HIGH_LEVEL_FILE)
            result.rebuilt_file_paths.append(path)
            self._provenance_edges(result, memory)
            if enrichment_status == "called":
                result.semantic_enrichment_calls += 1
            elif enrichment_status == "failed":
                result.semantic_enrichment_calls += 1
                result.semantic_enrichment_failures += 1
            if self.cache:
                self.cache.update_high_level_hash(
                    path,
                    source_hash=source_hash,
                    input_hash=input_hash,
                    policy_hash=self.policy_hash,
                )
        result.file_memory_build_time = time.perf_counter() - file_started

        symbol_units = [
            unit
            for unit in repo_units
            if self._type(unit) in {SoftwareUnitType.FUNCTION, SoftwareUnitType.CLASS}
        ]
        symbol_filter = set(affected_symbol_uids or ())
        if force_all or not symbol_filter:
            targets = symbol_units if force_all else []
        else:
            targets = [unit for unit in symbol_units if unit.uid in symbol_filter]

        for subject in targets:
            memory = self._symbol_memory(subject)
            self._append(result, memory, SoftwareSpace.HIGH_LEVEL_SYMBOL)
            result.rebuilt_symbol_uids.append(subject.uid)
            self._provenance_edges(result, memory)

        for commit in commit_facts:
            if not self._is_repair(commit.message):
                continue
            repair_uid = f"repair_memory:{commit.commit_hash}"
            if (
                not force_all
                and self.graph.semantic_map.get_unit(repair_uid) is not None
            ):
                continue
            memory = self._repair_memory(commit)
            if memory is None:
                continue
            self._append(result, memory, SoftwareSpace.HIGH_LEVEL_REPAIR)
            self._provenance_edges(result, memory, SoftwareRelation.SUPPORTED_BY)

        module_names = sorted(
            {
                self._module(unit)
                for unit in repo_units
                if self._type(unit)
                in {
                    SoftwareUnitType.FILE,
                    SoftwareUnitType.FUNCTION,
                    SoftwareUnitType.CLASS,
                }
                and self._module(unit)
            }
        )
        requested_modules = set(affected_modules or ())
        rebuilt_paths = set(result.rebuilt_file_paths)
        requested_modules.update(
            self._module(file_unit)
            for file_unit in file_units
            if file_unit.raw_data.get("path") in rebuilt_paths
            and self._module(file_unit)
        )
        module_targets = (
            module_names if force_all else sorted(requested_modules & set(module_names))
        )
        all_file_memories = self._file_memories(result)
        file_memories_by_module: dict[str, list[MemoryUnit]] = defaultdict(list)
        for memory in all_file_memories:
            module = self._module(memory)
            if module:
                file_memories_by_module[module].append(memory)
        for module_name in module_targets:
            memory = self._module_memory(
                module_name,
                members_by_module.get(module_name, []),
                commit_facts,
                file_memories_by_module.get(module_name, []),
            )
            self._append(result, memory, SoftwareSpace.HIGH_LEVEL_MODULE)
            result.rebuilt_module_names.append(module_name)
            self._provenance_edges(result, memory)
            # Only core module symbols participate in the high-level memory
            # hierarchy. Tests/configs remain direct provenance evidence, but
            # are not promoted to peer module components.
            for derived_uid in memory.raw_data.get("key_entities", []):
                if derived_uid.startswith("function:") or derived_uid.startswith(
                    "class:"
                ):
                    symbol_memory_uid = f"symbol_memory:{derived_uid.split(':', 1)[-1]}"
                    if self.graph.semantic_map.get_unit(
                        symbol_memory_uid
                    ) is not None or any(
                        unit.uid == symbol_memory_uid for unit in result.units
                    ):
                        result.edges.append(
                            SoftwareEdge(
                                memory.uid,
                                symbol_memory_uid,
                                SoftwareRelation.DERIVED_FROM,
                            )
                        )

        # Incremental invalidation also removes the repository architecture
        # for root __init__.py/config/deletion changes without a named module.
        # Recreate a missing aggregate even when no module target was selected.
        architecture_uid = f"architecture_memory:{self.repo_uid.split(':', 1)[-1]}"
        if (force_all or module_targets or requested_modules
                or self.graph.semantic_map.get_unit(architecture_uid) is None):
            architecture = self._architecture_memory(module_names, repo_units)
            self._append(result, architecture, SoftwareSpace.HIGH_LEVEL_ARCHITECTURE)
            self._provenance_edges(result, architecture)

        return result

    def _file_memory(
        self,
        file_unit: MemoryUnit,
        file_members: Sequence[MemoryUnit],
    ) -> tuple[MemoryUnit, str, str]:
        path = str(file_unit.raw_data.get("path", ""))
        members = list(file_members)
        classes = [
            unit for unit in members if self._type(unit) is SoftwareUnitType.CLASS
        ]
        functions = [
            unit for unit in members if self._type(unit) is SoftwareUnitType.FUNCTION
        ]
        segments = [
            unit
            for unit in members
            if self._type(unit)
            in {SoftwareUnitType.CODE_SEGMENT, SoftwareUnitType.SEMANTIC_BLOCK}
        ]
        public = [
            unit.uid
            for unit in [*classes, *functions]
            if not str(unit.raw_data.get("name", "")).startswith("_")
            and not unit.metadata.get("is_test")
        ]
        configs = self._unique(
            config.uid
            for function in functions
            for config in self._related(
                function.uid,
                SoftwareRelation.AFFECTS_BEHAVIOR_OF,
                "predecessors",
            )
        )
        tests = self._unique(
            test.uid
            for function in functions
            for test in self._related(
                function.uid,
                SoftwareRelation.TESTED_BY,
                "successors",
            )
        )
        history = self._unique(
            evolution.uid
            for function in functions
            for evolution in self._history(function.uid)
        )
        commits = self._unique(
            commit.uid
            for function in functions
            for commit in self._related_commits(function.uid)
        )
        side_effects = self._unique(
            effect
            for function in functions
            for effect in function.raw_data.get("side_effects", [])
        )
        exceptions = self._unique(
            exception
            for function in functions
            for exception in function.raw_data.get("exceptions", [])
        )
        reads = self._unique(
            read
            for function in functions
            for read in function.raw_data.get("reads", [])
        )
        writes = self._unique(
            write
            for function in functions
            for write in function.raw_data.get("writes", [])
        )
        config_references = self._unique(
            reference
            for function in functions
            for reference in function.raw_data.get("config_references", [])
        )
        calls_summary = [
            {
                "caller_uid": function.uid,
                "calls": list(function.raw_data.get("calls", []))[:30],
                "resolved_call_sites": self.semantic_code.get_call_sites(function.uid)[
                    :30
                ],
            }
            for function in functions
            if function.raw_data.get("calls")
        ][:50]
        responsibility = str(file_unit.raw_data.get("semantic_summary", "")) or (
            f"{file_unit.raw_data.get('language', '')} source file {path}."
        )
        payload = {
            "file_uid": file_unit.uid,
            "path": path,
            "language": str(file_unit.raw_data.get("language", "")),
            "responsibility": responsibility,
            "semantic_summary": (
                f"{path} contains {len(classes)} class(es), {len(functions)} "
                f"function(s), and {len(segments)} code segment(s)."
            ),
            "key_symbols": [unit.uid for unit in [*classes, *functions]][:100],
            "public_interfaces": public[:50],
            "important_classes": [unit.uid for unit in classes[:30]],
            "important_functions": [unit.uid for unit in functions[:50]],
            "important_segments": [unit.uid for unit in segments[:50]],
            "important_dependencies": list(file_unit.raw_data.get("imports", []))[:50],
            "imports": list(file_unit.raw_data.get("imports", [])),
            "exports": list(file_unit.raw_data.get("exports", [])),
            "important_state": self._unique([*writes, *config_references])[:50],
            "config_references": config_references[:50],
            "important_control_flow": [
                f"{unit.uid}: {unit.raw_data.get('segment_type', unit.raw_data.get('statement_type', 'segment'))}"
                for unit in segments[:50]
            ],
            "important_data_flow": [
                *(f"reads {value}" for value in reads[:25]),
                *(f"writes {value}" for value in writes[:25]),
            ],
            "known_risks": exceptions[:50],
            "side_effects": side_effects[:50],
            "derived_from": self._unique(
                [file_unit.uid]
                + [unit.uid for unit in members]
                + configs
                + tests
                + history
                + commits
            ),
            "confidence": 0.85,
            "source_hash": str(file_unit.raw_data.get("content_hash", "")),
            "content_hash": str(file_unit.raw_data.get("content_hash", "")),
            "metadata": {
                "repo_uid": self.repo_uid,
                "module_name": self._module(file_unit),
                "file_path": path,
            },
        }
        context = self._file_context(
            file_unit,
            classes=classes,
            functions=functions,
            segments=segments,
            calls_summary=calls_summary,
            tests=tests,
            history=history,
            commits=commits,
        )
        input_hash = self._hash({"payload": payload, "context": context})
        enrichment_status = "not_configured"
        if self.enricher is not None:
            enrichment_status = self._enrich(
                "file",
                payload,
                extra={"context": context},
            )
        payload["content_hash"] = self._hash(
            {
                key: value
                for key, value in payload.items()
                if key not in {"content_hash", "metadata"}
            }
        )
        return (
            FileMemorySchema(**payload).to_memory_unit(),
            input_hash,
            enrichment_status,
        )

    def _file_context(
        self,
        file_unit: MemoryUnit,
        *,
        classes: Sequence[MemoryUnit],
        functions: Sequence[MemoryUnit],
        segments: Sequence[MemoryUnit],
        calls_summary: list[dict[str, Any]],
        tests: list[str],
        history: list[str],
        commits: list[str],
    ) -> dict[str, Any]:
        path = str(file_unit.raw_data.get("path", ""))
        source = ""
        if self.repo_path is not None:
            source_path = self.repo_path / path
            try:
                source = source_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                source = ""
        truncated = len(source) > self.max_file_context_chars
        if truncated:
            header_budget = self.max_file_context_chars // 3
            remaining = self.max_file_context_chars - header_budget
            excerpts: list[str] = []
            used = 0
            for unit in [*classes, *functions, *segments]:
                excerpt = str(
                    unit.raw_data.get("source_code")
                    or unit.raw_data.get("code_snippet")
                    or ""
                )
                if not excerpt:
                    continue
                excerpt = excerpt[: min(4_000, remaining - used)]
                if not excerpt:
                    break
                excerpts.append(
                    f"# {unit.uid} lines {unit.raw_data.get('start_line')}-"
                    f"{unit.raw_data.get('end_line')}\n{excerpt}"
                )
                used += len(excerpt)
                if used >= remaining:
                    break
            source_excerpt = source[:header_budget] + "\n\n" + "\n\n".join(excerpts)
            source_excerpt = source_excerpt[: self.max_file_context_chars]
        else:
            source_excerpt = source
        return {
            "file": {
                "uid": file_unit.uid,
                "path": path,
                "language": file_unit.raw_data.get("language", ""),
                "imports": file_unit.raw_data.get("imports", []),
                "exports": file_unit.raw_data.get("exports", []),
            },
            "classes": [
                {
                    "uid": unit.uid,
                    "qualified_name": unit.raw_data.get("qualified_name"),
                    "bases": unit.raw_data.get("base_classes", []),
                    "docstring": unit.raw_data.get("docstring", ""),
                }
                for unit in classes[:50]
            ],
            "functions": [
                {
                    "uid": unit.uid,
                    "signature": unit.raw_data.get("signature", ""),
                    "docstring": unit.raw_data.get("docstring", ""),
                    "reads": unit.raw_data.get("reads", [])[:30],
                    "writes": unit.raw_data.get("writes", [])[:30],
                    "exceptions": unit.raw_data.get("exceptions", [])[:20],
                    "side_effects": unit.raw_data.get("side_effects", [])[:20],
                    "config_references": unit.raw_data.get("config_references", [])[
                        :20
                    ],
                }
                for unit in functions[:100]
            ],
            "segments": [
                {
                    "uid": unit.uid,
                    "segment_type": unit.raw_data.get(
                        "segment_type", unit.raw_data.get("statement_type", "")
                    ),
                    "span": [
                        unit.raw_data.get("start_line"),
                        unit.raw_data.get("end_line"),
                    ],
                    "summary": unit.raw_data.get("semantic_summary", ""),
                }
                for unit in segments[:100]
            ],
            "calls": calls_summary,
            "tests": tests[:50],
            "evolution": history[-30:],
            "commits": commits[-30:],
            "source_excerpt": source_excerpt,
            "source_truncated": truncated,
            "max_source_chars": self.max_file_context_chars,
        }

    def _file_memories(self, result: HighLevelBuildResult) -> list[MemoryUnit]:
        pending = {
            unit.uid: unit
            for unit in result.units
            if self._type(unit) is SoftwareUnitType.FILE_MEMORY
        }
        for unit in self.graph.get_units_in_memory_space(
            SoftwareSpace.HIGH_LEVEL_FILE.value,
            recursive=True,
        ):
            if (
                unit.metadata.get("repo_uid") == self.repo_uid
                and unit.uid not in pending
            ):
                pending[unit.uid] = unit
        return [pending[uid] for uid in sorted(pending)]

    def _symbol_memory(self, subject: MemoryUnit) -> MemoryUnit:
        is_function = self._type(subject) is SoftwareUnitType.FUNCTION
        callers = (
            self._related(subject.uid, SoftwareRelation.CALLS, "predecessors")
            if is_function
            else []
        )
        callees = (
            self._related(subject.uid, SoftwareRelation.CALLS, "successors")
            if is_function
            else []
        )
        tests = self._related(subject.uid, SoftwareRelation.TESTED_BY, "successors")
        configs = self._related(
            subject.uid, SoftwareRelation.AFFECTS_BEHAVIOR_OF, "predecessors"
        )
        history = self._history(subject.uid)
        commits = self._related_commits(subject.uid)
        issues = self._related_issues(subject.uid)
        summary = subject.raw_data.get("semantic_summary") or (
            f"Condensed memory for {subject.raw_data.get('qualified_name', subject.uid)}."
        )
        payload = {
            "subject_uid": subject.uid,
            "responsibility": summary,
            "contract": {
                "inputs": subject.raw_data.get("parameters", []),
                "outputs": subject.raw_data.get("return_type", ""),
                "preconditions": subject.raw_data.get("preconditions", []),
                "postconditions": subject.raw_data.get("postconditions", []),
            },
            "behavior": {
                "reads": subject.raw_data.get("reads", []),
                "writes": subject.raw_data.get("writes", []),
                "side_effects": subject.raw_data.get("side_effects", []),
                "exceptions": subject.raw_data.get("exceptions", []),
            },
            "structural_context": {
                "important_callers": [unit.uid for unit in callers[:20]],
                "important_callees": [unit.uid for unit in callees[:20]],
                "ordered_call_sites": (
                    self.semantic_code.get_call_sites(subject.uid)[:50]
                    if is_function
                    else []
                ),
                "tests": [unit.uid for unit in tests[:20]],
                "configs": [unit.uid for unit in configs[:20]],
            },
            "evolution_context": {
                "important_changes": [
                    unit.raw_data.get("diff_summary", "") for unit in history[-10:]
                ],
                "related_commits": [unit.uid for unit in commits[-10:]],
                "fixed_issues": [unit.uid for unit in issues[-10:]],
                "known_failure_modes": [
                    unit.raw_data.get("title", "") for unit in issues[-10:]
                ],
                "design_rationale": "",
            },
            "semantic_summary": summary,
            "derived_from": self._unique(
                [subject.uid]
                + [
                    unit.uid
                    for unit in callers
                    + callees
                    + tests
                    + configs
                    + history
                    + commits
                    + issues
                ]
            ),
            "confidence": 0.9 if is_function else 0.85,
            "metadata": {
                "repo_uid": self.repo_uid,
                "module_name": self._module(subject),
            },
        }
        if self.enrich_symbols:
            self._enrich("symbol", payload)
        return SymbolMemorySchema(**payload).to_memory_unit()

    def _repair_memory(self, commit: GitCommitFacts) -> MemoryUnit | None:
        commit_uid = f"commit:{commit.commit_hash}"
        if self.graph.semantic_map.get_unit(commit_uid) is None:
            return None
        affected = [
            uid
            for uid in commit.modified_symbols
            if self.graph.get_unit(uid) is not None
        ]
        issue_units = self._related(
            commit_uid, SoftwareRelation.FIXED_BY, "predecessors"
        )
        validation: list[str] = []
        for symbol_uid in affected:
            validation.extend(
                unit.uid
                for unit in self._related(
                    symbol_uid, SoftwareRelation.TESTED_BY, "successors"
                )
            )
        related_issue = issue_units[0].uid if issue_units else None
        if issue_units:
            issue = issue_units[0]
            issue_title = str(issue.raw_data.get("title", "")).strip()
            issue_body = str(issue.raw_data.get("body", "")).strip()
            symptom = "\n\n".join(part for part in (issue_title, issue_body) if part)
        else:
            symptom = commit.message
        payload = {
            "symptom": symptom,
            # A commit title and diff identify a repair but do not, by
            # themselves, prove root cause. Optional enrichment may fill this.
            "root_cause": "",
            "affected_symbols": affected,
            "repair_strategy": commit.diff_summary,
            "changed_symbols": affected,
            "validation": self._unique(validation),
            "constraints": [],
            "lessons": [],
            "related_issue": related_issue,
            "related_commit": commit_uid,
            "semantic_summary": (
                f"Repair evidence for {symptom.splitlines()[0]} via {commit_uid}."
            ),
            "derived_from": self._unique(
                [commit_uid]
                + [unit.uid for unit in issue_units]
                + affected
                + validation
                + [
                    unit.uid
                    for uid in affected
                    for unit in self._history(uid)
                ]
            ),
            "confidence": 0.8 if issue_units else 0.6,
            "repair_id": commit.commit_hash,
            "metadata": {"repo_uid": self.repo_uid},
        }
        if self.enrich_repairs:
            self._enrich("repair", payload)
        return RepairMemorySchema(**payload).to_memory_unit()

    def _module_memory(
        self,
        module_name: str,
        module_members: Sequence[MemoryUnit],
        commits: Sequence[GitCommitFacts],
        file_memories: Sequence[MemoryUnit],
    ) -> MemoryUnit:
        members = list(module_members)
        symbols = [
            unit
            for unit in members
            if self._type(unit) in {SoftwareUnitType.FUNCTION, SoftwareUnitType.CLASS}
        ]
        files = [unit for unit in members if self._type(unit) is SoftwareUnitType.FILE]
        module_file_memories = [
            memory for memory in file_memories if self._module(memory) == module_name
        ]
        configs = self._unique(
            config
            for symbol in symbols
            for config in self._related(
                symbol.uid,
                SoftwareRelation.AFFECTS_BEHAVIOR_OF,
                "predecessors",
            )
        )
        tests = self._unique(
            test
            for symbol in symbols
            for test in self._related(
                symbol.uid, SoftwareRelation.TESTED_BY, "successors"
            )
        )
        interfaces = [
            unit.uid
            for unit in symbols
            if not str(unit.raw_data.get("name", "")).startswith("_")
            and not unit.metadata.get("is_test")
        ]
        local_components = {
            component
            for file_unit in files
            for component in str(file_unit.metadata.get("module_name", "")).split(".")[
                1:
            ]
            if component
        }
        dependencies = self._unique(
            self._dependency_component(imported)
            for file_unit in files
            for imported in file_unit.raw_data.get("imports", [])
            if imported
            and self._dependency_component(imported)
            and self._dependency_component(imported) != module_name
            and self._dependency_component(imported) not in local_components
            and self._dependency_component(imported) not in sys.stdlib_module_names
        )
        risks = self._unique(
            item
            for unit in symbols
            for item in (
                list(unit.raw_data.get("exceptions", []))
                + list(unit.raw_data.get("side_effects", []))
            )
        )
        member_paths = {
            unit.raw_data.get("path") or unit.raw_data.get("file_path")
            for unit in members
        }
        recent = [
            f"commit:{commit.commit_hash} {commit.message}"
            for commit in commits
            if member_paths.intersection(commit.changed_files)
        ][:10]
        responsibilities = self._unique(
            memory.raw_data.get("responsibility", "")
            or memory.raw_data.get("semantic_summary", "")
            for memory in module_file_memories
            if memory.raw_data.get("responsibility")
            or memory.raw_data.get("semantic_summary")
        )[:10]
        summary = f"Module {module_name} contains {len(files)} file(s) and {len(symbols)} symbol(s)."
        payload = {
            "module_name": module_name,
            "responsibilities": responsibilities or [summary],
            "key_entities": self._unique(
                uid
                for memory in module_file_memories
                for uid in memory.raw_data.get("key_symbols", [])
            )[:50]
            or [unit.uid for unit in symbols[:50]],
            "important_dependencies": dependencies[:30],
            "public_interfaces": interfaces[:30],
            "important_configs": [unit.uid for unit in configs[:30]],
            "important_tests": [unit.uid for unit in tests[:30]],
            "known_risks": risks[:30],
            "recent_important_changes": recent,
            "semantic_summary": summary,
            "evolution_summary": "; ".join(recent[:5]),
            "derived_from": self._unique(
                [memory.uid for memory in module_file_memories]
                # FileMemory is the file-grained aggregation layer and already
                # preserves File/Class/Function/CodeSegment provenance. Avoid
                # duplicating every low-level member as a module-level edge.
                + [unit.uid for unit in files]
                + [unit.uid for unit in tests]
            ),
            "metadata": {"repo_uid": self.repo_uid, "module_name": module_name},
        }
        return ModuleMemorySchema(**payload).to_memory_unit()

    def _architecture_memory(
        self,
        module_names: Sequence[str],
        repo_units: Sequence[MemoryUnit],
    ) -> MemoryUnit:
        module_uids = [f"module_memory:{name}" for name in module_names]
        functions = [
            unit for unit in repo_units if self._type(unit) is SoftwareUnitType.FUNCTION
        ]
        files = [
            unit for unit in repo_units if self._type(unit) is SoftwareUnitType.FILE
        ]
        configs = [
            unit.uid
            for unit in repo_units
            if self._type(unit) is SoftwareUnitType.CONFIG
        ]
        tests = [unit.uid for unit in functions if unit.metadata.get("is_test")]
        interfaces = [
            unit.uid
            for unit in functions
            if not unit.metadata.get("is_test")
            and not str(unit.raw_data.get("name", "")).startswith("_")
        ][:50]
        internal_components = {
            component
            for file_unit in files
            for component in str(file_unit.metadata.get("module_name", "")).split(".")
            if component
        }
        dependencies = self._unique(
            f"{self._module(file_unit)} -> {self._dependency_component(imported)}"
            for file_unit in files
            for imported in file_unit.raw_data.get("imports", [])
            if imported
            and self._module(file_unit)
            and self._dependency_component(imported)
            and self._dependency_component(imported) != self._module(file_unit)
            and self._dependency_component(imported) not in internal_components
            and self._dependency_component(imported) not in sys.stdlib_module_names
        )
        supporting_infrastructure = self._supporting_infrastructure(repo_units)
        critical_paths = [
            f"{unit.uid} calls {', '.join(unit.raw_data.get('calls', [])[:5])}"
            for unit in functions
            if unit.raw_data.get("calls")
        ][:20]
        scope = self.repo_uid.split(":", 1)[-1]
        summary = (
            f"Repository {scope} is organized into {len(module_names)} module(s): "
            f"{', '.join(module_names[:20])}."
        )
        payload = {
            "scope": scope,
            "components": list(module_names),
            "responsibilities": [
                f"{name}: aggregated in module_memory:{name}" for name in module_names
            ],
            "dependencies": dependencies[:50],
            "critical_paths": critical_paths,
            "public_interfaces": interfaces,
            "important_configurations": configs[:50],
            "important_tests": tests[:50],
            "supporting_infrastructure": supporting_infrastructure,
            "semantic_summary": summary,
            "derived_from": module_uids,
            "metadata": {"repo_uid": self.repo_uid},
        }
        return ArchitectureMemorySchema(**payload).to_memory_unit()

    def _repo_units(self) -> list[MemoryUnit]:
        return sorted(
            (
                unit
                for unit in self.graph.get_units_in_memory_space(
                    SoftwareSpace.SOFTWARE.value, recursive=True
                )
                if unit.metadata.get("repo_uid") == self.repo_uid
                and self._type(unit)
                not in {
                    SoftwareUnitType.SYMBOL_MEMORY,
                    SoftwareUnitType.REPAIR_MEMORY,
                    SoftwareUnitType.FILE_MEMORY,
                    SoftwareUnitType.MODULE_MEMORY,
                    SoftwareUnitType.ARCHITECTURE_MEMORY,
                }
            ),
            key=lambda unit: unit.uid,
        )

    def _prepare_history_index(self, repo_units: Sequence[MemoryUnit]) -> None:
        """Index evolution nodes once instead of rescanning the space per symbol."""
        by_subject: dict[str, list[MemoryUnit]] = defaultdict(list)
        for unit in repo_units:
            if self._type(unit) is not SoftwareUnitType.EVOLUTION:
                continue
            # Current-state aggregates must be reconstructable from repository
            # bytes plus committed Git history. Session mutation events remain
            # available through get_history(), without changing embeddings and
            # provenance of identical current code based on its editing path.
            if unit.raw_data.get("change_type") == "working_tree_change":
                continue
            subject_uid = str(unit.raw_data.get("subject_uid", ""))
            if subject_uid:
                by_subject[subject_uid].append(unit)
        self._history_by_subject = {
            uid: sorted(
                units,
                key=lambda unit: (
                    *self.semantic_code._timestamp_key(unit),
                    unit.uid,
                ),
            )
            for uid, units in by_subject.items()
        }

    def _history(self, uid: str) -> list[MemoryUnit]:
        return list(self._history_by_subject.get(uid, ()))

    def _related_commits(self, uid: str) -> list[MemoryUnit]:
        cached = self._commits_cache.get(uid)
        if cached is not None:
            return list(cached)
        commits = [
            unit
            for unit in self._related(uid, SoftwareRelation.MODIFIES, "all")
            if self._type(unit) is SoftwareUnitType.COMMIT
        ]
        for evolution in self._history(uid):
            commits.extend(
                unit
                for unit in self._related(
                    evolution.uid,
                    SoftwareRelation.CAUSED_BY_EVIDENCE,
                    "all",
                )
                if self._type(unit) is SoftwareUnitType.COMMIT
            )
        result = self._deduplicate_units(commits)
        self._commits_cache[uid] = result
        return list(result)

    def _related_issues(self, uid: str) -> list[MemoryUnit]:
        cached = self._issues_cache.get(uid)
        if cached is not None:
            return list(cached)
        issues: list[MemoryUnit] = []
        for relation in (SoftwareRelation.MENTIONS, SoftwareRelation.IMPACTS):
            issues.extend(
                unit
                for unit in self._related(uid, relation, "all")
                if self._type(unit) is SoftwareUnitType.ISSUE
            )
        for commit in self._related_commits(uid):
            issues.extend(
                unit
                for unit in self._related(
                    commit.uid, SoftwareRelation.FIXED_BY, "all"
                )
                if self._type(unit) is SoftwareUnitType.ISSUE
            )
        for evolution in self._history(uid):
            issues.extend(
                unit
                for unit in self._related(
                    evolution.uid,
                    SoftwareRelation.CAUSED_BY_EVIDENCE,
                    "all",
                )
                if self._type(unit) is SoftwareUnitType.ISSUE
            )
        result = self._deduplicate_units(issues)
        self._issues_cache[uid] = result
        return list(result)

    @staticmethod
    def _deduplicate_units(units: Iterable[MemoryUnit]) -> list[MemoryUnit]:
        by_uid = {unit.uid: unit for unit in units}
        return [by_uid[uid] for uid in sorted(by_uid)]

    def _related(
        self,
        uid: str,
        relation: SoftwareRelation,
        direction: str,
    ) -> list[MemoryUnit]:
        cache_key = (uid, relation, direction)
        cached = self._related_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        related = sorted(
            self.graph.traverse_explicit_nodes(
                uid,
                relationship_type=relation.value,
                direction=direction,
            ),
            key=lambda unit: unit.uid,
        )
        self._related_cache[cache_key] = related
        return list(related)

    def _provenance_edges(
        self,
        result: HighLevelBuildResult,
        memory: MemoryUnit,
        relation: SoftwareRelation = SoftwareRelation.DERIVED_FROM,
    ) -> None:
        for evidence_uid in memory.raw_data.get("derived_from", []):
            if self.graph.semantic_map.get_unit(evidence_uid) is not None or any(
                unit.uid == evidence_uid for unit in result.units
            ):
                result.edges.append(SoftwareEdge(memory.uid, evidence_uid, relation))

    @staticmethod
    def _append(
        result: HighLevelBuildResult,
        memory: MemoryUnit,
        space: SoftwareSpace,
    ) -> None:
        result.units.append(memory)
        result.spaces.append([space.value])

    def _enrich(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        if self.enricher is None:
            return "not_configured"
        enrichment_payload = dict(payload)
        enrichment_payload.update(extra or {})
        try:
            enriched = self.enricher(kind, enrichment_payload)
        except Exception:
            return "failed"
        if isinstance(enriched, str) and enriched.strip():
            payload["semantic_summary"] = enriched.strip()
        elif isinstance(enriched, dict):
            for key, value in enriched.items():
                if key in payload and key not in {"derived_from", "metadata", "uid"}:
                    payload[key] = value
        return "called"

    @staticmethod
    def _type(unit: MemoryUnit) -> SoftwareUnitType | None:
        value = unit.metadata.get("software_type") or unit.raw_data.get("software_type")
        if value == SoftwareUnitType.SEMANTIC_BLOCK.value:
            return SoftwareUnitType.CODE_SEGMENT
        try:
            return SoftwareUnitType(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hash(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _module(unit: MemoryUnit) -> str:
        if unit.metadata.get("is_test"):
            return ""
        if (
            unit.raw_data.get("is_config")
            or unit.metadata.get("source_kind") == "config"
        ):
            return ""
        module = str(unit.metadata.get("module_name", ""))
        parts = [part for part in module.split(".") if part]
        while parts and parts[0].lower() in {"src", "lib", "python"}:
            parts.pop(0)
        if not parts or parts[0].lower() in {
            "build",
            "dist",
            "doc",
            "docs",
            "example",
            "examples",
            "scripts",
            "setup",
            "t",
            "test",
            "tests",
            "tox",
            ".github",
            "github",
            "conftest",
            "makefile",
            "pyproject",
            "readthedocs",
            "pre-commit-config",
        }:
            return ""
        return parts[0]

    @staticmethod
    def _dependency_component(imported: str) -> str:
        parts = [part for part in imported.lstrip(".").split(".") if part]
        while parts and parts[0].lower() in {"src", "lib", "python"}:
            parts.pop(0)
        return parts[0] if parts else ""

    @staticmethod
    def _supporting_infrastructure(repo_units: Sequence[MemoryUnit]) -> list[str]:
        supporting: set[str] = set()
        support_names = {
            ".github",
            "build",
            "ci",
            "doc",
            "docs",
            "example",
            "examples",
            "scripts",
            "t",
            "test",
            "tests",
        }
        build_files = {
            "makefile",
            "noxfile.py",
            "pyproject.toml",
            "setup.cfg",
            "setup.py",
            "tox.ini",
        }
        for unit in repo_units:
            if HighLevelMemoryBuilder._type(unit) is not SoftwareUnitType.FILE:
                continue
            path = str(unit.raw_data.get("path", ""))
            first = path.split("/", 1)[0].lower()
            name = path.rsplit("/", 1)[-1].lower()
            if unit.metadata.get("is_test"):
                supporting.add("tests")
            elif first in support_names:
                supporting.add(first)
            elif unit.raw_data.get("is_config"):
                supporting.add(path)
            elif name in build_files:
                supporting.add(path)
        return sorted(supporting)

    @staticmethod
    def _is_repair(message: str) -> bool:
        lowered = message.lower()
        return any(
            token in lowered
            for token in (
                "fix",
                "bug",
                "regression",
                "crash",
                "error",
                "incorrect",
                "timeout",
                "race",
            )
        )

    @staticmethod
    def _unique(values: Iterable[Any]) -> list[Any]:
        result: list[Any] = []
        seen: set[Any] = set()
        for value in values:
            if value in (None, ""):
                continue
            try:
                if value in seen:
                    continue
                seen.add(value)
            except TypeError:
                if value in result:
                    continue
            result.append(value)
        return result


__all__ = ["HighLevelBuildResult", "HighLevelMemoryBuilder", "SemanticEnricher"]
