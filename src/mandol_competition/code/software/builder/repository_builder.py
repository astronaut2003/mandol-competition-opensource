"""End-to-end local Repository -> SemanticCode construction pipeline."""

from __future__ import annotations

import os
import time
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mandol_substrate.representation import MemoryUnit
from ..relations import SoftwareRelation
from ..schemas import (
    CommitSchema,
    EvolutionSchema,
    IssueSchema,
    RepositorySchema,
    SoftwareUnitType,
)
from ..spaces import SoftwareSpace
from .cache import BUILDER_VERSION, BuilderCache
from .git_analyzer import GitAnalyzer, GitCommitFacts
from .high_level_builder import HighLevelMemoryBuilder, SemanticEnricher
from .parser import (
    CParser,
    CppParser,
    GoParser,
    JavaScriptParser,
    JavaParser,
    LanguageParser,
    ParsedFile,
    PythonParser,
    RustParser,
    TypeScriptParser,
)
from .relation_builder import BaseBuildFacts, RelationBuilder, SoftwareEdge
from .scanner import ManifestEntry, RepositoryScanner

if TYPE_CHECKING:
    from ...core.semantic_graph import SemanticGraph
    from ..facade import SemanticCode


@dataclass
class RepositoryBuildReport:
    repository_path: str
    revision: str
    repository_files: int = 0
    source_files: int = 0
    source_loc: int = 0
    functions: int = 0
    classes: int = 0
    code_segments: int = 0
    # Deprecated report field retained for consumers of builder version 3.
    semantic_blocks: int = 0
    parse_failures: int = 0
    unsupported_files: int = 0
    language_statistics: dict[str, int] = field(default_factory=dict)
    memory_units: int = 0
    relationships: int = 0
    high_level_memories: int = 0
    file_memories: int = 0
    parsed_files: int = 0
    cache_hits: int = 0
    changed_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    rebuilt_symbols: list[str] = field(default_factory=list)
    rebuilt_modules: list[str] = field(default_factory=list)
    rebuilt_files: list[str] = field(default_factory=list)
    rebuilt_file_memories: list[str] = field(default_factory=list)
    semantic_enrichment_calls: int = 0
    semantic_enrichment_cache_hits: int = 0
    semantic_enrichment_failures: int = 0
    scan_time: float = 0.0
    parse_time: float = 0.0
    base_memory_build_time: float = 0.0
    git_evolution_time: float = 0.0
    embedding_index_time: float = 0.0
    dense_embedding_time: float = 0.0
    base_dense_embedding_time: float = 0.0
    git_dense_embedding_time: float = 0.0
    high_level_dense_embedding_time: float = 0.0
    dense_embeddings_generated: int = 0
    embedding_batches: int = 0
    embedding_batch_size: int = 0
    embedding_device: str = ""
    relation_fact_build_time: float = 0.0
    relationship_insert_time: float = 0.0
    relationships_removed: int = 0
    retired_evolutions: int = 0
    consistency_repair: bool = False
    relation_build_mode: str = "full"
    relation_files_processed: int = 0
    git_history_cache_hit: bool = False
    git_commits_processed: int = 0
    fast_noop: bool = False
    high_level_memory_build_time: float = 0.0
    file_memory_build_time: float = 0.0
    total_build_time: float = 0.0
    peak_ram_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RepositoryMemoryBuilder:
    """Orchestrate discovery, parsing, Git facts, graph edges, and agglomeration."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        semantic_code: SemanticCode | None = None,
        semantic_graph: SemanticGraph | None = None,
        revision: str = "HEAD",
        build_relations: bool = True,
        build_high_level_memory: bool = True,
        include_git_history: bool = True,
        incremental: bool = True,
        issues: Sequence[IssueSchema | MemoryUnit | dict[str, Any]] | None = None,
        cache_path: str | Path | None = None,
        max_workers: int | None = None,
        max_git_commits: int | None = 100,
        batch_size: int = 32,
        build_indexes: bool = True,
        build_code_segments: bool = True,
        build_semantic_blocks: bool | None = None,
        max_file_size: int = 1_000_000,
        parser: LanguageParser | None = None,
        parsers: Mapping[str, LanguageParser] | None = None,
        semantic_enricher: SemanticEnricher | None = None,
        max_file_context_chars: int = 40_000,
        semantic_enricher_version: str = "",
        enrich_symbols: bool = False,
        enrich_repairs: bool = False,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.revision = revision
        self.build_relations = build_relations
        self.build_high_level_memory = build_high_level_memory
        self.include_git_history = include_git_history
        self.incremental = incremental
        self.issues = list(issues or [])
        self.cache_path = (
            Path(cache_path)
            if cache_path
            else self.repo_path / ".mandol" / "semantic_code_cache.json"
        )
        self.max_workers = max_workers or min(8, max(1, os.cpu_count() or 1))
        self.max_git_commits = max_git_commits
        self.batch_size = max(1, batch_size)
        self.build_indexes = build_indexes
        self.max_file_size = max_file_size
        if build_semantic_blocks is not None:
            build_code_segments = build_semantic_blocks
        self.build_code_segments = build_code_segments
        default_parsers: dict[str, LanguageParser] = {
            "python": PythonParser(build_code_segments=build_code_segments),
            "c": CParser(build_code_segments=build_code_segments),
            "cpp": CppParser(build_code_segments=build_code_segments),
            "java": JavaParser(build_code_segments=build_code_segments),
            "javascript": JavaScriptParser(
                build_code_segments=build_code_segments
            ),
            "typescript": TypeScriptParser(
                build_code_segments=build_code_segments
            ),
            "go": GoParser(build_code_segments=build_code_segments),
            "rust": RustParser(build_code_segments=build_code_segments),
        }
        if parser is not None:
            default_parsers[parser.language] = parser
        if parsers:
            default_parsers.update(parsers)
        self.parsers = default_parsers
        self.parser = self.parsers["python"]
        self.semantic_enricher = semantic_enricher
        self.max_file_context_chars = max_file_context_chars
        self.semantic_enricher_version = semantic_enricher_version
        self.enrich_symbols = enrich_symbols
        self.enrich_repairs = enrich_repairs

        if semantic_code is None:
            from ..facade import SemanticCode

            semantic_code = SemanticCode(semantic_graph)
        elif (
            semantic_graph is not None
            and semantic_code.semantic_graph is not semantic_graph
        ):
            raise ValueError(
                "semantic_code and semantic_graph refer to different graphs"
            )
        self.semantic_code = semantic_code
        self.graph = semantic_code.semantic_graph

    def build(self) -> SemanticCode:
        started = time.perf_counter()
        self._dense_embedding_time = 0.0
        self._dense_embeddings_generated = 0
        self._embedding_batches = 0
        self._phase_embedding_time = {"base": 0.0, "git": 0.0, "high": 0.0}
        self._relationship_insert_time = 0.0
        self._relationships_removed = 0
        self._retired_evolutions = 0
        self._relation_changed_uids: set[str] = set()
        report = RepositoryBuildReport(
            repository_path=str(self.repo_path),
            revision=self.revision,
            embedding_batch_size=self.batch_size,
            embedding_device=str(
                getattr(self.graph.semantic_map, "embedding_device", "")
            ),
        )

        scan_started = time.perf_counter()
        manifest = RepositoryScanner(
            self.repo_path,
            max_file_size=self.max_file_size,
        ).scan()
        report.scan_time = time.perf_counter() - scan_started
        report.repository_files = manifest.file_count
        report.source_files = manifest.source_file_count
        report.source_loc = manifest.source_loc
        report.language_statistics = dict(manifest.language_statistics)

        cache = (
            BuilderCache(self.cache_path).load()
            if self.incremental
            else BuilderCache(self.cache_path)
        )
        entry_by_path = {entry.path: entry for entry in manifest.files}
        changed_paths = {
            path
            for path, entry in entry_by_path.items()
            if not self.incremental or cache.file_hash(path) != entry.content_hash
        }
        deleted_paths = cache.paths - set(entry_by_path)
        report.changed_files = sorted(changed_paths)
        report.deleted_files = sorted(deleted_paths)

        parse_started = time.perf_counter()
        parsed_files: dict[str, ParsedFile] = {}
        source_entries = [entry for entry in manifest.files if entry.kind == "source"]
        entries_to_parse: list[ManifestEntry] = []
        for entry in source_entries:
            if entry.language not in self.parsers:
                report.unsupported_files += 1
                continue
            cached = (
                cache.parsed_file(entry.path)
                if entry.path not in changed_paths
                else None
            )
            if cached is not None:
                parsed_files[entry.path] = cached
                report.cache_hits += 1
            else:
                entries_to_parse.append(entry)
        if entries_to_parse:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(entries_to_parse))
            ) as executor:
                futures = {
                    executor.submit(
                        self.parsers[entry.language].parse_file,
                        Path(entry.absolute_path),
                        entry.path,
                    ): entry
                    for entry in entries_to_parse
                }
                for future in as_completed(futures):
                    entry = futures[future]
                    try:
                        parsed_files[entry.path] = future.result()
                    except Exception as exc:
                        parsed_files[entry.path] = ParsedFile(
                            path=entry.path,
                            language=entry.language,
                            module_name=self._module_name(entry.path),
                            content_hash=entry.content_hash,
                            source_loc=entry.source_loc,
                            is_test=entry.is_test,
                            parse_error=str(exc),
                        )
                    report.parsed_files += 1
                    report.rebuilt_files.append(entry.path)
        report.parse_time = time.perf_counter() - parse_started
        # Parallel parse completion must not affect symbol resolution or caps.
        parsed_files = dict(sorted(parsed_files.items()))
        report.rebuilt_files.sort()
        report.functions = sum(
            len(parsed.functions) for parsed in parsed_files.values()
        )
        report.classes = sum(len(parsed.classes) for parsed in parsed_files.values())
        report.code_segments = sum(
            len(function.segments)
            for parsed in parsed_files.values()
            for function in parsed.functions
        )
        report.semantic_blocks = report.code_segments
        report.parse_failures = sum(
            bool(parsed.parse_error) for parsed in parsed_files.values()
        )

        git_analyzer = GitAnalyzer(
            self.repo_path,
            revision=self.revision,
            max_commits=self.max_git_commits,
            parsers=self.parsers,
        )
        default_branch = (
            git_analyzer.default_branch() if self.include_git_history else "main"
        )
        revision_hash = (
            git_analyzer.revision_hash() if self.include_git_history else self.revision
        )
        stored_repo = self.graph.semantic_map.get_unit(f"repo:{manifest.name}")
        if stored_repo is not None and not self.include_git_history:
            revision_hash = stored_repo.raw_data.get("revision", revision_hash)
            default_branch = stored_repo.raw_data.get("default_branch", default_branch)
        cached_revision = cache.data.get("last_processed_commit") or (
            stored_repo.raw_data.get("revision") if stored_repo else None
        )
        revision_changed = bool(cached_revision and cached_revision != revision_hash)
        if cached_revision and cached_revision != revision_hash:
            for unit in list(self.graph.semantic_map.get_all_units()):
                if (unit.metadata.get("repo_uid") == f"repo:{manifest.name}"
                        and unit.raw_data.get("change_type") == "working_tree_change"):
                    self.graph.delete_unit(unit.uid)
                    self._retired_evolutions += 1
            cache.data["working_evolutions"] = {}
        repo_uid = f"repo:{manifest.name}"
        repository_schema = RepositorySchema(
            name=manifest.name,
            path=str(self.repo_path),
            default_branch=default_branch,
            revision=revision_hash,
            language_statistics=manifest.language_statistics,
            file_count=manifest.file_count,
            source_file_count=manifest.source_file_count,
            test_file_count=manifest.test_file_count,
            description=f"Local software repository {manifest.name}.",
            uid=repo_uid,
            metadata={
                "repo_uid": repo_uid,
                "repo_path": str(self.repo_path),
                "builder_version": BUILDER_VERSION,
            },
        )

        old_symbol_hashes = {
            path: cache.symbol_hashes(path) for path in changed_paths | deleted_paths
        }
        existing_repo = self.graph.semantic_map.get_unit(repo_uid)
        fresh_graph = existing_repo is None
        repair_graph = not fresh_graph and cache.data.get("graph_consistency_version") != 1
        report.consistency_repair = repair_graph
        has_git_memory = any(
            unit.metadata.get("repo_uid") == repo_uid
            for unit in self.graph.get_units_in_memory_space(
                SoftwareSpace.COMMITS.value, recursive=True
            )
        )
        architecture_uid = f"architecture_memory:{manifest.name}"
        phase_state = getattr(self.semantic_code, "_phase_state", {})
        relations_ready = bool(
            not self.build_relations
            or phase_state.get("relations")
            or self._has_base_relations()
        )
        git_ready = bool(not self.include_git_history or has_git_memory)
        high_ready = bool(
            not self.build_high_level_memory
            or self.graph.semantic_map.get_unit(architecture_uid) is not None
        )
        index = getattr(self.graph.semantic_map, "faiss_index", None)
        indexes_ready = bool(
            not self.build_indexes
            or (index is not None and getattr(index, "ntotal", 0) > 0)
        )
        fast_noop = bool(
            self.incremental
            and not fresh_graph
            and not repair_graph
            and not changed_paths
            and not deleted_paths
            and cached_revision == revision_hash
            and not self.issues
            and relations_ready
            and git_ready
            and high_ready
            and indexes_ready
        )
        report.fast_noop = fast_noop

        base_started = time.perf_counter()
        stable_symbol_identity = False
        touched_base_uids = {
            uid
            for path in changed_paths | deleted_paths
            for uid in cache.unit_uids(path)
        }
        changed_payload_uids: set[str] = set()
        relation_source_paths: set[str] | None = None
        if fast_noop:
            report.relation_build_mode = "noop"
            base_facts = BaseBuildFacts()
            base_units: list[MemoryUnit] = []
            base_spaces: list[list[str]] = []
            issue_units: list[MemoryUnit] = []
            relationship_count = 0
        else:
            relation_started = time.perf_counter()
            relation_builder = RelationBuilder(manifest, repo_uid)
            can_try_local_build = bool(
                self.incremental
                and not fresh_graph
                and not repair_graph
                and changed_paths
                and not deleted_paths
                and changed_paths.issubset(cache.paths)
                and all(
                    entry.kind == "source"
                    for entry in manifest.files
                    if entry.path in changed_paths
                )
            )
            if can_try_local_build:
                relation_source_paths = changed_paths | self._dependent_source_paths(
                    touched_base_uids
                )
                candidate_facts = relation_builder.build(
                    parsed_files,
                    repository_schema=repository_schema,
                    unit_paths=changed_paths,
                    relation_paths=relation_source_paths,
                )
                stable_symbol_identity = all(
                    set(old_symbol_hashes.get(path, {}))
                    == set(candidate_facts.symbol_hashes_by_file.get(path, {}))
                    for path in changed_paths
                )
                base_facts = (
                    candidate_facts
                    if stable_symbol_identity
                    else relation_builder.build(
                        parsed_files,
                        repository_schema=repository_schema,
                    )
                )
                report.relation_build_mode = (
                    "incremental" if stable_symbol_identity else "full"
                )
                report.relation_files_processed = (
                    len(relation_source_paths)
                    if stable_symbol_identity
                    else len(parsed_files)
                )
            else:
                base_facts = relation_builder.build(
                    parsed_files,
                    repository_schema=repository_schema,
                )
                report.relation_build_mode = "full"
                report.relation_files_processed = len(parsed_files)
            report.relation_fact_build_time = time.perf_counter() - relation_started
            if report.relation_build_mode == "full":
                relation_source_paths = None
            changed_payload_uids = {
                unit.uid for unit in base_facts.units
                if self.graph.semantic_map.get_unit(unit.uid) != unit
            }
            # Capture old neighbours before a deleted symbol loses its edges.
            changed_payload_uids.update(
                self._high_level_relation_neighbors(touched_base_uids)
            )
            touched_base_uids.update(
                uid
                for path in changed_paths
                for uid in base_facts.unit_uids_by_file.get(path, [])
            )
            if not fresh_graph:
                self._remove_stale_base_units(
                    cache,
                    base_facts,
                    changed_paths,
                    deleted_paths,
                    full=report.relation_build_mode == "full",
                )
            base_units, base_spaces = self._select_base_units(
                base_facts,
                changed_paths,
                fresh_graph=fresh_graph,
            )
            self._batch_insert(base_units, base_spaces, phase="base")
            if self.build_relations:
                relationship_count = self._reconcile_edges(
                    base_facts.edges,
                    repo_uid=repo_uid,
                    relations={
                        SoftwareRelation.CONTAINS, SoftwareRelation.CONTAINED_IN,
                        SoftwareRelation.HAS_METHOD, SoftwareRelation.NEXT_SEGMENT,
                        SoftwareRelation.CALLS, SoftwareRelation.TESTED_BY,
                        SoftwareRelation.IMPORTS, SoftwareRelation.INHERITS_FROM,
                        SoftwareRelation.IMPLEMENTS, SoftwareRelation.AFFECTS_BEHAVIOR_OF,
                    },
                    source_paths=relation_source_paths,
                    structural_uids=touched_base_uids,
                )
            else:
                relationship_count = 0

            issue_units = self._issue_units(repo_uid)
            issue_units = [
                unit
                for unit in issue_units
                if self.graph.semantic_map.get_unit(unit.uid) != unit
            ]
            self._batch_insert(
                issue_units,
                [[SoftwareSpace.ISSUES.value] for _ in issue_units],
                phase="base",
            )
        report.base_memory_build_time = time.perf_counter() - base_started

        git_started = time.perf_counter()
        commit_facts: list[GitCommitFacts] = []
        if self.include_git_history and not fast_noop:
            cached_commit_facts = cache.cached_git_facts(
                revision_hash, self.max_git_commits
            )
            can_reuse_git_facts = bool(
                cached_commit_facts is not None
                and not deleted_paths
                and (not changed_paths or stable_symbol_identity)
            )
            if can_reuse_git_facts:
                commit_facts = cached_commit_facts or []
                report.git_history_cache_hit = True
            else:
                commit_facts = git_analyzer.analyze(parsed_files)
                cache.update_git_facts(
                    revision_hash, self.max_git_commits, commit_facts
                )
        report.git_commits_processed = len(commit_facts)
        git_units, git_spaces, git_edges = ([], [], []) if fast_noop else self._git_memory(
            commit_facts,
            repo_uid,
            base_facts,
            cache,
            changed_paths,
            old_symbol_hashes,
        )
        git_units_to_insert = [
            unit
            for unit in git_units
            if self.graph.semantic_map.get_unit(unit.uid) != unit
        ]
        git_space_by_uid = {
            unit.uid: space for unit, space in zip(git_units, git_spaces)
        }
        self._batch_insert(
            git_units_to_insert,
            [git_space_by_uid[unit.uid] for unit in git_units_to_insert],
            phase="git",
        )
        if not fast_noop:
            desired_git_uids = {unit.uid for unit in git_units}
            for unit in list(self.graph.semantic_map.get_all_units()):
                if unit.metadata.get("software_type") not in {"evolution", "commit"}:
                    continue
                if unit.metadata.get("repo_uid") == repo_uid and unit.uid not in desired_git_uids:
                    self.graph.delete_unit(unit.uid)
                    changed_payload_uids.add(unit.uid)
                    if unit.metadata.get("software_type") == "evolution":
                        self._retired_evolutions += 1
            relationship_count += self._reconcile_edges(
                git_edges, repo_uid=repo_uid,
                relations={SoftwareRelation.MODIFIES, SoftwareRelation.HAS_EVOLUTION,
                           SoftwareRelation.CAUSED_BY_EVIDENCE, SoftwareRelation.EVOLVES_FROM,
                           SoftwareRelation.FIXED_BY},
            )
        report.git_evolution_time = time.perf_counter() - git_started

        changed_symbol_uids = self._changed_symbol_uids(
            base_facts, changed_paths | deleted_paths, old_symbol_hashes
        )
        changed_symbol_uids.update(changed_payload_uids | self._relation_changed_uids)
        affected_modules = self._affected_modules(
            parsed_files, changed_paths, deleted_paths, cache
        )

        high_started = time.perf_counter()
        high_units_inserted: list[MemoryUnit] = []
        if self.build_high_level_memory:
            changed_symbol_uids.update(
                self._high_level_relation_neighbors(changed_symbol_uids)
            )
            architecture_uid = f"architecture_memory:{manifest.name}"
            force_high = (
                fresh_graph
                or not self.incremental
                or repair_graph
                or revision_changed
                or self.graph.semantic_map.get_unit(architecture_uid) is None
            )
            if (
                force_high
                or changed_paths
                or deleted_paths
                or changed_symbol_uids
                or affected_modules
            ):
                invalidated_symbol_subjects = self._invalidate_high_level(
                    changed_symbol_uids,
                    affected_modules,
                    changed_paths | deleted_paths,
                    repo_uid,
                    force_all=force_high,
                )
                changed_symbol_uids.update(invalidated_symbol_subjects)
                # Deleting a caller/test in another module can invalidate an
                # otherwise unchanged SymbolMemory. Deleting that memory also
                # removes its inbound ModuleMemory provenance edge. Rebuild
                # the owning module so its hierarchy and edge set are restored
                # from current facts rather than leaving a history-dependent
                # graph after repeated incremental updates.
                affected_modules.update(
                    module
                    for subject_uid in invalidated_symbol_subjects
                    if (subject := self.graph.semantic_map.get_unit(subject_uid))
                    is not None
                    and (module := self._module_for_unit(subject))
                )
            high_result = HighLevelMemoryBuilder(
                self.semantic_code,
                repo_uid,
                enricher=self.semantic_enricher,
                cache=cache,
                repo_path=self.repo_path,
                max_file_context_chars=self.max_file_context_chars,
                enrichment_config_key=self.semantic_enricher_version,
                enrich_symbols=self.enrich_symbols,
                enrich_repairs=self.enrich_repairs,
            ).build(
                commit_facts=commit_facts,
                affected_symbol_uids=changed_symbol_uids,
                affected_modules=affected_modules,
                changed_file_paths=changed_paths,
                force_all=force_high,
            )
            self._batch_insert(
                high_result.units, high_result.spaces, phase="high"
            )
            high_units_inserted = high_result.units
            relationship_count += self._reconcile_edges(
                high_result.edges, repo_uid=repo_uid,
                relations={SoftwareRelation.DERIVED_FROM, SoftwareRelation.SUPPORTED_BY},
                source_uids={unit.uid for unit in high_result.units},
            )
            report.rebuilt_symbols = high_result.rebuilt_symbol_uids
            report.rebuilt_modules = high_result.rebuilt_module_names
            report.rebuilt_file_memories = high_result.rebuilt_file_paths
            report.semantic_enrichment_calls = high_result.semantic_enrichment_calls
            report.semantic_enrichment_cache_hits = (
                high_result.semantic_enrichment_cache_hits
            )
            report.semantic_enrichment_failures = (
                high_result.semantic_enrichment_failures
            )
            report.file_memory_build_time = high_result.file_memory_build_time
        report.high_level_memory_build_time = time.perf_counter() - high_started

        index_started = time.perf_counter()
        if self.build_indexes and (
            base_units or issue_units or git_units_to_insert or high_units_inserted
        ):
            self.graph.semantic_map.build_index()
        report.embedding_index_time = time.perf_counter() - index_started

        if not fast_noop:
            paths_to_cache = (
                {entry.path for entry in manifest.files}
                if fresh_graph or not self.incremental or report.relation_build_mode == "full"
                else changed_paths
            )
            for entry in manifest.files:
                if entry.path not in paths_to_cache:
                    continue
                parsed = parsed_files.get(entry.path)
                cache.update_file(
                    entry.path,
                    entry.content_hash,
                    parsed,
                    base_facts.unit_uids_by_file.get(entry.path, []),
                    base_facts.symbol_hashes_by_file.get(entry.path, {}),
                )
        for path in deleted_paths:
            cache.remove_file(path)
        if not fast_noop:
            cache.data["last_processed_commit"] = revision_hash
            cache.data["graph_consistency_version"] = 1
        if self.incremental and not fast_noop:
            cache.save()

        repo_units = [
            unit
            for unit in self.graph.get_units_in_memory_space(
                SoftwareSpace.SOFTWARE.value, recursive=True
            )
            if unit.metadata.get("repo_uid") == repo_uid
        ]
        report.memory_units = len(repo_units)
        report.high_level_memories = sum(
            unit.metadata.get("software_type")
            in {
                SoftwareUnitType.SYMBOL_MEMORY.value,
                SoftwareUnitType.REPAIR_MEMORY.value,
                SoftwareUnitType.FILE_MEMORY.value,
                SoftwareUnitType.MODULE_MEMORY.value,
                SoftwareUnitType.ARCHITECTURE_MEMORY.value,
            }
            for unit in repo_units
        )
        report.file_memories = sum(
            unit.metadata.get("software_type") == SoftwareUnitType.FILE_MEMORY.value
            for unit in repo_units
        )
        report.relationships = self.graph.rx_graph.num_edges()
        report.dense_embedding_time = self._dense_embedding_time
        report.base_dense_embedding_time = self._phase_embedding_time["base"]
        report.git_dense_embedding_time = self._phase_embedding_time["git"]
        report.high_level_dense_embedding_time = self._phase_embedding_time["high"]
        report.dense_embeddings_generated = self._dense_embeddings_generated
        report.embedding_batches = self._embedding_batches
        report.relationship_insert_time = self._relationship_insert_time
        report.relationships_removed = self._relationships_removed
        report.retired_evolutions = self._retired_evolutions
        report.peak_ram_mb = self._peak_ram_mb()
        report.total_build_time = time.perf_counter() - started

        self.semantic_code.last_build_report = report
        self.semantic_code.repository_path = str(self.repo_path)
        self.semantic_code.repository_uid = repo_uid
        self.semantic_code._repository_builder_options = {
            "revision": self.revision,
            "build_relations": self.build_relations,
            "build_high_level_memory": self.build_high_level_memory,
            "include_git_history": self.include_git_history,
            "incremental": self.incremental,
            "cache_path": str(self.cache_path),
            "max_workers": self.max_workers,
            "max_git_commits": self.max_git_commits,
            "batch_size": self.batch_size,
            "build_indexes": self.build_indexes,
            "max_file_size": self.max_file_size,
            "build_code_segments": self.build_code_segments,
            "parser": self.parser,
            "parsers": self.parsers,
            "semantic_enricher": self.semantic_enricher,
            "max_file_context_chars": self.max_file_context_chars,
            "semantic_enricher_version": self.semantic_enricher_version,
            "enrich_symbols": self.enrich_symbols,
            "enrich_repairs": self.enrich_repairs,
            "issues": self.issues,
        }
        return self.semantic_code

    def _remove_stale_base_units(
        self,
        cache: BuilderCache,
        facts: BaseBuildFacts,
        changed_paths: set[str],
        deleted_paths: set[str],
        *,
        full: bool = False,
    ) -> None:
        if full:
            current = {unit.uid for unit in facts.units}
            base_types = {"repository", "directory", "file", "class", "function",
                          "code_segment", "semantic_block", "config"}
            repo_uid = f"repo:{self.repo_path.name}"
            for unit in list(self.graph.semantic_map.get_all_units()):
                if (unit.metadata.get("repo_uid") == repo_uid
                        and unit.metadata.get("software_type") in base_types
                        and unit.uid not in current):
                    self.graph.delete_unit(unit.uid)
            return
        for path in sorted(deleted_paths):
            for uid in cache.unit_uids(path):
                if self.graph.semantic_map.get_unit(uid) is not None:
                    self.graph.delete_unit(uid)

        for path in sorted(changed_paths):
            old_uids = set(cache.unit_uids(path))
            current_uids = set(facts.unit_uids_by_file.get(path, []))
            delete_uids = old_uids - current_uids
            for uid in delete_uids:
                if self.graph.semantic_map.get_unit(uid) is not None:
                    self.graph.delete_unit(uid)

    def _select_base_units(
        self,
        facts: BaseBuildFacts,
        changed_paths: set[str],
        *,
        fresh_graph: bool,
    ) -> tuple[list[MemoryUnit], list[list[str]]]:
        if fresh_graph:
            return facts.units, facts.spaces
        units: list[MemoryUnit] = []
        spaces: list[list[str]] = []
        for unit, unit_spaces in zip(facts.units, facts.spaces):
            path = (
                unit.raw_data.get("path")
                or unit.raw_data.get("file_path")
                or unit.raw_data.get("source_file")
            )
            existing = self.graph.semantic_map.get_unit(unit.uid)
            if path in changed_paths or existing is None or existing != unit:
                units.append(unit)
                spaces.append(unit_spaces)
        return units, spaces

    def _batch_insert(
        self,
        units: Sequence[MemoryUnit],
        spaces: Sequence[list[str]],
        *,
        phase: str,
    ) -> dict[str, Any]:
        if not units:
            return {}
        stats = self.graph.batch_add_units(
            list(units),
            batch_size=self.batch_size,
            per_unit_space_names=list(spaces),
            index_update_mode="none",
            generate_sparse_embedding=False,
            show_progress=False,
        )
        embedding_time = float(stats.get("dense_embedding_time", 0.0))
        self._dense_embedding_time += embedding_time
        self._phase_embedding_time[phase] += embedding_time
        self._dense_embeddings_generated += int(
            stats.get("embedding_generated", 0)
        )
        self._embedding_batches += int(stats.get("embedding_batches", 0))
        return stats

    def _add_edges(
        self,
        edges: Iterable[SoftwareEdge],
        *,
        assume_missing: bool = False,
    ) -> int:
        started = time.perf_counter()
        added = 0
        seen_in_batch: set[tuple[Any, ...]] = set()
        for edge in edges:
            call_site_identity = (
                edge.properties.get("call_site_id")
                if edge.relation is SoftwareRelation.CALLS
                else None
            )
            edge_identity = (
                edge.source_uid,
                edge.target_uid,
                edge.relation.value,
                call_site_identity,
            )
            if edge_identity in seen_in_batch:
                continue
            seen_in_batch.add(edge_identity)
            existing_edges = (
                []
                if assume_missing
                else self.graph.get_edge_data(edge.source_uid, edge.target_uid)
            )
            if edge.relation is SoftwareRelation.CALLS:
                call_site_id = edge.properties.get("call_site_id")
                duplicate = any(
                    data.get("type") == edge.relation.value
                    and (
                        data.get("call_site_id") == call_site_id
                        if call_site_id
                        else data.get("source_order")
                        == edge.properties.get("source_order")
                    )
                    for data in existing_edges
                )
            else:
                duplicate = any(
                    data.get("type") == edge.relation.value for data in existing_edges
                )
            if duplicate:
                continue
            if self.graph.add_relationship(
                edge.source_uid,
                edge.target_uid,
                edge.relation.value,
                **edge.properties,
            ):
                added += 1
        self._relationship_insert_time += time.perf_counter() - started
        return added

    def _reconcile_edges(
        self,
        edges: Iterable[SoftwareEdge],
        *,
        repo_uid: str,
        relations: set[SoftwareRelation],
        source_paths: set[str] | None = None,
        structural_uids: set[str] | None = None,
        source_uids: set[str] | None = None,
    ) -> int:
        """Replace changed relationship groups, retaining repeated call sites.

        Local semantic facts are owned by their emitting file, including
        reversed TESTED_BY/config edges. Full resolution passes reconcile all
        pairs: adding a symbol can invalidate a formerly unique resolution in
        an otherwise untouched file. Core's deletion API removes one multiedge
        per call, so remove exactly the observed count for a changed group.
        """
        relation_names = {relation.value for relation in relations}
        structural = {"CONTAINS", "CONTAINED_IN", "HAS_METHOD", "NEXT_SEGMENT"}
        reverse = {"TESTED_BY", "AFFECTS_BEHAVIOR_OF"}

        def in_scope(source: str, target: str, relation: str) -> bool:
            if relation not in relation_names:
                return False
            unit = self.graph.semantic_map.get_unit(source)
            if unit is None or unit.metadata.get("repo_uid") != repo_uid:
                return False
            if source_uids is not None:
                return source in source_uids
            if source_paths is None:
                return True
            if relation in structural:
                return source in (structural_uids or ()) or target in (structural_uids or ())
            owner = self.graph.semantic_map.get_unit(target if relation in reverse else source)
            if owner is None:
                return False
            path = owner.raw_data.get("file_path") or owner.raw_data.get("path")
            return path in source_paths

        expected: dict[tuple[str, str, str], list[SoftwareEdge]] = defaultdict(list)
        for edge in edges:
            expected[(edge.source_uid, edge.target_uid, edge.relation.value)].append(edge)
        actual: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for index in self.graph.rx_graph.edge_indices():
            a, b = self.graph.rx_graph.get_edge_endpoints_by_index(index)
            source, target = self.graph._index_to_uid[a], self.graph._index_to_uid[b]
            data = dict(self.graph.rx_graph.get_edge_data_by_index(index) or {})
            relation = data.pop("type", "")
            data.pop("created", None)
            if (source, target, relation) in expected or in_scope(source, target, relation):
                actual[(source, target, relation)].append(data)

        def signatures(properties):
            return Counter(json.dumps(p, sort_keys=True, default=str) for p in properties)

        pending: list[SoftwareEdge] = []
        for key in sorted(actual.keys() | expected.keys()):
            old = actual.get(key, [])
            new = expected.get(key, [])
            if signatures(old) == signatures(edge.properties for edge in new):
                continue
            self._relation_changed_uids.update(key[:2])
            for _ in old:
                self.graph.delete_relationship(*key)
                self._relationships_removed += 1
            pending.extend(new)
        return self._add_edges(pending, assume_missing=True)

    def _has_base_relations(self) -> bool:
        base_relations = {
            SoftwareRelation.CONTAINS.value,
            SoftwareRelation.CONTAINED_IN.value,
            SoftwareRelation.HAS_METHOD.value,
            SoftwareRelation.IMPORTS.value,
            SoftwareRelation.CALLS.value,
        }
        return any(
            data and data.get("type") in base_relations
            for edge_index in self.graph.rx_graph.edge_indices()
            for data in [self.graph.rx_graph.get_edge_data_by_index(edge_index)]
        )

    def _dependent_source_paths(self, touched_uids: Iterable[str]) -> set[str]:
        """Return source files whose relations may be lost with touched units.

        Deleting a changed Function/File also removes inbound multiedges.  A
        local rebuild therefore includes the existing one-hop dependants so
        their CALLS/TESTED_BY/IMPORTS facts are restored without rebuilding all
        repository facts.
        """
        paths: set[str] = set()
        for uid in touched_uids:
            neighbor_uids = self.graph.get_predecessors(uid)
            neighbor_uids.extend(self.graph.get_successors(uid))
            for neighbor_uid in neighbor_uids:
                unit = self.graph.semantic_map.get_unit(neighbor_uid)
                if unit is None:
                    continue
                path = unit.raw_data.get("file_path") or unit.raw_data.get("path")
                if isinstance(path, str) and path:
                    paths.add(path)
        return paths

    def _issue_units(self, repo_uid: str) -> list[MemoryUnit]:
        units: list[MemoryUnit] = []
        for issue in self.issues:
            if isinstance(issue, MemoryUnit):
                unit = issue
            elif isinstance(issue, IssueSchema):
                unit = issue.to_memory_unit()
            elif isinstance(issue, dict):
                unit = IssueSchema(**issue).to_memory_unit()
            else:
                raise TypeError(f"Unsupported issue input: {type(issue)}")
            if unit.metadata.get("software_type") != SoftwareUnitType.ISSUE.value:
                raise ValueError(f"Issue input {unit.uid} is not an Issue MemoryUnit")
            unit.metadata.update(
                {"repo_uid": repo_uid, "repo_path": str(self.repo_path)}
            )
            units.append(unit)
        return units

    def _git_memory(
        self,
        commits: Sequence[GitCommitFacts],
        repo_uid: str,
        base_facts: BaseBuildFacts,
        cache: BuilderCache,
        changed_paths: set[str],
        old_symbol_hashes: dict[str, dict[str, str]],
    ) -> tuple[list[MemoryUnit], list[list[str]], list[SoftwareEdge]]:
        units: list[MemoryUnit] = []
        spaces: list[list[str]] = []
        edges: list[SoftwareEdge] = []
        issues_by_id = {
            str(unit.raw_data.get("issue_id")): unit.uid
            for unit in self.graph.get_units_in_memory_space(
                SoftwareSpace.ISSUES.value, recursive=True
            )
            if unit.metadata.get("repo_uid") == repo_uid
        }
        prior_by_subject: dict[str, str] = {}

        if not self.include_git_history:
            # A staged relation-only build must preserve an already populated
            # committed history stage, even though it did not request Git I/O.
            history_units = [unit for unit in self.graph.semantic_map.get_all_units()
                             if unit.metadata.get("repo_uid") == repo_uid
                             and unit.metadata.get("software_type") in {"commit", "evolution"}
                             and unit.raw_data.get("change_type") != "working_tree_change"]
            for unit in history_units:
                units.append(unit)
                spaces.append([SoftwareSpace.COMMITS.value if unit.metadata["software_type"] == "commit"
                               else SoftwareSpace.EVOLUTION.value])
            for unit in sorted(history_units, key=self.semantic_code._timestamp_key):
                if subject := unit.raw_data.get("subject_uid"):
                    prior_by_subject[subject] = unit.uid
            for index in self.graph.rx_graph.edge_indices():
                a, b = self.graph.rx_graph.get_edge_endpoints_by_index(index)
                source, target = self.graph._index_to_uid[a], self.graph._index_to_uid[b]
                data = dict(self.graph.rx_graph.get_edge_data_by_index(index))
                relation = data.pop("type", "")
                if relation in {"MODIFIES", "HAS_EVOLUTION", "CAUSED_BY_EVIDENCE", "EVOLVES_FROM", "FIXED_BY"}:
                    left = self.graph.semantic_map.get_unit(source)
                    right = self.graph.semantic_map.get_unit(target)
                    if (left and right and left.metadata.get("repo_uid") == repo_uid
                            and all(u.raw_data.get("change_type") != "working_tree_change" for u in (left, right))):
                        data.pop("created", None)
                        edges.append(SoftwareEdge(source, target, SoftwareRelation(relation), data))

        for commit in sorted(
            commits, key=lambda item: (item.timestamp, item.commit_hash)
        ):
            commit_uid = f"commit:{commit.commit_hash}"
            commit_unit = CommitSchema(
                commit_hash=commit.commit_hash,
                message=commit.message,
                author=commit.author,
                timestamp=commit.timestamp,
                changed_files=commit.changed_files,
                metadata={
                    "repo_uid": repo_uid,
                    "repo_path": str(self.repo_path),
                    "parents": commit.parents,
                    "diff_summary": commit.diff_summary,
                },
            ).to_memory_unit()
            units.append(commit_unit)
            spaces.append([SoftwareSpace.COMMITS.value])
            linked_issues = [
                issues_by_id[issue_id]
                for issue_id in GitAnalyzer.issue_ids(commit.message)
                if issue_id in issues_by_id
            ]
            for issue_uid in linked_issues:
                edges.append(
                    SoftwareEdge(issue_uid, commit_uid, SoftwareRelation.FIXED_BY)
                )
            for path in commit.changed_files:
                file_uid = f"file:{path}"
                if self.graph.semantic_map.get_unit(file_uid) is not None:
                    edges.append(
                        SoftwareEdge(commit_uid, file_uid, SoftwareRelation.MODIFIES)
                    )
            for subject_uid in commit.modified_symbols:
                if self.graph.semantic_map.get_unit(subject_uid) is None:
                    continue
                edges.append(
                    SoftwareEdge(commit_uid, subject_uid, SoftwareRelation.MODIFIES)
                )
                prior_uid = prior_by_subject.get(subject_uid)
                evolution = EvolutionSchema(
                    subject_uid=subject_uid,
                    change_type=GitAnalyzer.classify_change(commit.message),
                    timestamp=commit.timestamp,
                    diff_summary=commit.diff_summary,
                    source_evidence=[commit_uid] + linked_issues,
                    prior_evolution_uid=prior_uid,
                    impacted_units=[subject_uid]
                    + [f"file:{path}" for path in commit.changed_files],
                    confidence=0.9,
                    evolution_id=commit.commit_hash,
                    metadata={"repo_uid": repo_uid, "repo_path": str(self.repo_path)},
                ).to_memory_unit()
                units.append(evolution)
                spaces.append([SoftwareSpace.EVOLUTION.value])
                edges.append(
                    SoftwareEdge(
                        subject_uid, evolution.uid, SoftwareRelation.HAS_EVOLUTION
                    )
                )
                edges.append(
                    SoftwareEdge(
                        evolution.uid, commit_uid, SoftwareRelation.CAUSED_BY_EVIDENCE
                    )
                )
                for issue_uid in linked_issues:
                    edges.append(
                        SoftwareEdge(
                            evolution.uid,
                            issue_uid,
                            SoftwareRelation.CAUSED_BY_EVIDENCE,
                        )
                    )
                if prior_uid:
                    edges.append(
                        SoftwareEdge(
                            evolution.uid, prior_uid, SoftwareRelation.EVOLVES_FROM
                        )
                    )
                prior_by_subject[subject_uid] = evolution.uid

        cached_working = cache.data.setdefault("working_evolutions", {})
        # The checkpoint is authoritative; BuilderCache is expendable. Recover
        # events when a loaded graph is rebound to a fresh/missing cache.
        evolution_fields = {item.name for item in fields(EvolutionSchema)}
        for unit in self.graph.get_units_in_memory_space(SoftwareSpace.EVOLUTION.value, recursive=True):
            if (unit.metadata.get("repo_uid") == repo_uid
                    and unit.raw_data.get("change_type") == "working_tree_change"
                    and unit.uid not in cached_working):
                cached_working[unit.uid] = {
                    **{key: value for key, value in unit.raw_data.items() if key in evolution_fields},
                    "uid": unit.uid, "metadata": dict(unit.metadata),
                }
        for evolution_uid, payload in sorted(
            list(cached_working.items()),
            key=lambda item: (item[1].get("timestamp", ""), item[0]),
        ):
            try:
                evolution = EvolutionSchema(**payload).to_memory_unit()
            except (TypeError, ValueError):
                continue
            if (
                self.graph.semantic_map.get_unit(evolution.raw_data["subject_uid"])
                is None
            ):
                # Deleted symbols must not leave orphan events in the graph
                # or cache (which would resurrect them on later updates).
                cached_working.pop(evolution_uid, None)
                continue
            units.append(evolution)
            spaces.append([SoftwareSpace.EVOLUTION.value])
            edges.append(
                SoftwareEdge(
                    evolution.raw_data["subject_uid"],
                    evolution.uid,
                    SoftwareRelation.HAS_EVOLUTION,
                )
            )
            prior_uid = prior_by_subject.get(evolution.raw_data["subject_uid"])
            if prior_uid != evolution.raw_data.get("prior_evolution_uid"):
                payload["prior_evolution_uid"] = prior_uid
                evolution = EvolutionSchema(**payload).to_memory_unit()
                units[-1] = evolution
            if prior_uid:
                edges.append(
                    SoftwareEdge(
                        evolution.uid, prior_uid, SoftwareRelation.EVOLVES_FROM
                    )
                )
            prior_by_subject[evolution.raw_data["subject_uid"]] = evolution.uid

        current_hashes = base_facts.symbol_hashes_by_file
        now = datetime.now(UTC).isoformat()
        for path in sorted(changed_paths):
            for subject_uid, current_hash in current_hashes.get(path, {}).items():
                old_hash = old_symbol_hashes.get(path, {}).get(subject_uid)
                if not old_hash or old_hash == current_hash:
                    continue
                # Returning to an earlier body is a new event, not an update
                # of an old hash-addressed node (which creates history cycles).
                event_hash = hashlib.sha256(json.dumps(
                    [subject_uid, prior_by_subject.get(subject_uid), current_hash],
                    separators=(",", ":"),
                ).encode()).hexdigest()[:20]
                evolution_id = f"working-{event_hash}"
                payload = {
                    "subject_uid": subject_uid,
                    "change_type": "working_tree_change",
                    "timestamp": now,
                    "diff_summary": f"Working tree changed {subject_uid} in {path}.",
                    "source_evidence": [],
                    "prior_evolution_uid": prior_by_subject.get(subject_uid),
                    "impacted_units": [subject_uid, f"file:{path}"],
                    "confidence": 0.7,
                    "evolution_id": evolution_id,
                    "metadata": {
                        "repo_uid": repo_uid,
                        "repo_path": str(self.repo_path),
                    },
                }
                evolution = EvolutionSchema(**payload).to_memory_unit()
                cached_working[evolution.uid] = payload
                units.append(evolution)
                spaces.append([SoftwareSpace.EVOLUTION.value])
                edges.append(
                    SoftwareEdge(
                        subject_uid, evolution.uid, SoftwareRelation.HAS_EVOLUTION
                    )
                )
                prior_uid = prior_by_subject.get(subject_uid)
                if prior_uid:
                    edges.append(
                        SoftwareEdge(
                            evolution.uid, prior_uid, SoftwareRelation.EVOLVES_FROM
                        )
                    )
                prior_by_subject[subject_uid] = evolution.uid
        return self._deduplicate_units(units, spaces, edges)

    @staticmethod
    def _deduplicate_units(
        units: list[MemoryUnit],
        spaces: list[list[str]],
        edges: list[SoftwareEdge],
    ) -> tuple[list[MemoryUnit], list[list[str]], list[SoftwareEdge]]:
        by_uid: dict[str, tuple[MemoryUnit, list[str]]] = {}
        for unit, unit_spaces in zip(units, spaces):
            by_uid[unit.uid] = (unit, unit_spaces)
        dedup_edges: dict[tuple[str, str, SoftwareRelation], SoftwareEdge] = {
            (edge.source_uid, edge.target_uid, edge.relation): edge for edge in edges
        }
        return (
            [item[0] for item in by_uid.values()],
            [item[1] for item in by_uid.values()],
            list(dedup_edges.values()),
        )

    @staticmethod
    def _changed_symbol_uids(
        facts: BaseBuildFacts,
        changed_paths: set[str],
        old_hashes: dict[str, dict[str, str]],
    ) -> set[str]:
        result: set[str] = set()
        for path in changed_paths:
            current = facts.symbol_hashes_by_file.get(path, {})
            previous = old_hashes.get(path, {})
            result.update(
                uid
                for uid, content_hash in current.items()
                if previous.get(uid) != content_hash
            )
            result.update(set(previous) - set(current))
        return result

    @staticmethod
    def _affected_modules(
        parsed_files: dict[str, ParsedFile],
        changed_paths: set[str],
        deleted_paths: set[str],
        cache: BuilderCache,
    ) -> set[str]:
        modules = {
            parsed_files[path].module_name.split(".", 1)[0]
            for path in changed_paths
            if path in parsed_files and parsed_files[path].module_name
        }
        for path in deleted_paths:
            parsed = cache.parsed_file(path)
            if parsed and parsed.module_name:
                modules.add(parsed.module_name.split(".", 1)[0])
        for path in changed_paths:
            if path not in parsed_files:
                modules.add(RepositoryMemoryBuilder._module_name(path).split(".", 1)[0])
        return {module for module in modules if module}

    def _invalidate_high_level(
        self,
        changed_symbols: Iterable[str],
        affected_modules: Iterable[str],
        affected_paths: Iterable[str],
        repo_uid: str,
        *,
        force_all: bool = False,
    ) -> set[str]:
        changed_symbols = set(changed_symbols)
        affected_paths = set(affected_paths)
        candidates = [
            f"symbol_memory:{uid.split(':', 1)[-1]}" for uid in changed_symbols
        ]
        candidates.extend(f"module_memory:{module}" for module in affected_modules)
        candidates.extend(f"file_memory:{path}" for path in affected_paths)
        candidates.append(f"architecture_memory:{repo_uid.split(':', 1)[-1]}")

        # A deleted or changed symbol can also be evidence for a RepairMemory
        # (or another high-level aggregate outside its owning module). Remove
        # any aggregate whose provenance intersects the invalidated current
        # facts; the high-level builder will deterministically reconstruct the
        # still-valid aggregate from the current graph later in this build.
        invalidated_evidence = changed_symbols | {
            f"file:{path}" for path in affected_paths
        }
        high_units = [unit for unit in self.graph.get_units_in_memory_space(
            SoftwareSpace.HIGH_LEVEL.value,
            recursive=True,
        ) if unit.metadata.get("repo_uid") == repo_uid]
        # Follow provenance to a fixed point, so deleting/rebuilding a
        # FileMemory or SymbolMemory also refreshes its module/architecture.
        dependents: dict[str, set[str]] = defaultdict(set)
        for unit in high_units:
            for uid in unit.raw_data.get("derived_from", []):
                dependents[uid].add(unit.uid)
            if force_all:
                candidates.append(unit.uid)
        # Some module -> symbol edges are supplemental hierarchy provenance.
        for unit in high_units:
            for uid in self.graph.get_successors(unit.uid):
                dependents[uid].add(unit.uid)
        pending = list(invalidated_evidence | set(candidates))
        invalidated = set(pending)
        while pending:
            for uid in dependents.get(pending.pop(), ()):
                if uid not in invalidated:
                    invalidated.add(uid)
                    pending.append(uid)
        candidates = list(invalidated)
        invalidated_symbol_subjects: set[str] = set()
        for uid in set(candidates):
            unit = self.graph.semantic_map.get_unit(uid)
            if unit is not None and unit.metadata.get("repo_uid") == repo_uid:
                if self.semantic_code._software_type(unit) not in {
                    SoftwareUnitType.SYMBOL_MEMORY, SoftwareUnitType.FILE_MEMORY,
                    SoftwareUnitType.MODULE_MEMORY, SoftwareUnitType.ARCHITECTURE_MEMORY,
                    SoftwareUnitType.REPAIR_MEMORY,
                }:
                    continue
                module = HighLevelMemoryBuilder._module(unit)
                if module and isinstance(affected_modules, set):
                    affected_modules.add(module)
                if (
                    unit.metadata.get("software_type")
                    == SoftwareUnitType.SYMBOL_MEMORY.value
                ):
                    subject_uid = unit.raw_data.get("subject_uid")
                    if subject_uid:
                        invalidated_symbol_subjects.add(str(subject_uid))
                self.graph.delete_unit(uid)
        return invalidated_symbol_subjects

    def _high_level_relation_neighbors(
        self, changed_symbol_uids: Iterable[str]
    ) -> set[str]:
        """Return current symbols whose aggregate context depends on changes."""
        neighbors: set[str] = set()
        for uid in changed_symbol_uids:
            if self.graph.semantic_map.get_unit(uid) is None:
                continue
            for relation in (
                SoftwareRelation.CALLS,
                SoftwareRelation.TESTED_BY,
                SoftwareRelation.AFFECTS_BEHAVIOR_OF,
            ):
                for direction in ("predecessors", "successors"):
                    for unit in self.graph.traverse_explicit_nodes(
                        uid,
                        relationship_type=relation.value,
                        direction=direction,
                    ):
                        if unit.metadata.get("software_type") in {
                            SoftwareUnitType.FUNCTION.value,
                            SoftwareUnitType.CLASS.value,
                        }:
                            neighbors.add(unit.uid)
        return neighbors

    @staticmethod
    def _module_name(path: str) -> str:
        pure = Path(path)
        parts = list(pure.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _module_for_unit(unit: MemoryUnit) -> str:
        return HighLevelMemoryBuilder._module(unit)

    @staticmethod
    def _peak_ram_mb() -> float | None:
        try:
            import resource

            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return float(value) / (1024.0 if os.name != "darwin" else 1024.0 * 1024.0)
        except (ImportError, OSError):
            return None


__all__ = ["RepositoryBuildReport", "RepositoryMemoryBuilder"]
