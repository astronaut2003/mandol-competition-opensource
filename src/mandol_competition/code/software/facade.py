"""Thin software-domain facade over Mandol's existing graph and map."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mandol_substrate.representation import MemoryUnit
from .relations import SoftwareRelation
from .schemas import SoftwareUnitType
from .spaces import (
    SoftwareSpace,
    initialize_software_spaces,
    software_space_for_type,
)

if TYPE_CHECKING:
    from ..core.semantic_graph import SemanticGraph


class SemanticCode:
    """Software-oriented convenience API backed by a standard SemanticGraph."""

    def __init__(
        self,
        semantic_graph: SemanticGraph | None = None,
        *,
        initialize_spaces: bool = True,
    ) -> None:
        if semantic_graph is None:
            from ..ingestion.repository import RepositoryPreparation

            semantic_graph = RepositoryPreparation()
        self.semantic_graph = semantic_graph
        if initialize_spaces:
            initialize_software_spaces(self.semantic_graph)

    @classmethod
    def from_repository(
        cls,
        repo_path: str,
        *,
        revision: str = "HEAD",
        build_high_level_memory: bool = True,
        include_git_history: bool = True,
        incremental: bool = True,
        semantic_graph: SemanticGraph | None = None,
        **builder_options: Any,
    ) -> SemanticCode:
        """Automatically convert a local repository into Mandol memory."""
        from .builder import RepositoryMemoryBuilder

        semantic_code = cls(semantic_graph)
        return RepositoryMemoryBuilder(
            repo_path,
            semantic_code=semantic_code,
            revision=revision,
            build_high_level_memory=build_high_level_memory,
            include_git_history=include_git_history,
            incremental=incremental,
            **builder_options,
        ).build()

    def update_repository(
        self, repo_path: str | None = None, **overrides: Any
    ) -> SemanticCode:
        """Incrementally rebuild a previously attached local repository."""
        from .builder import RepositoryMemoryBuilder

        resolved_path = repo_path or getattr(self, "repository_path", None)
        if not resolved_path:
            raise ValueError("repo_path is required before the first repository build")
        options = dict(getattr(self, "_repository_builder_options", {}))
        options.update(overrides)
        return RepositoryMemoryBuilder(
            resolved_path,
            semantic_code=self,
            **options,
        ).build()

    @staticmethod
    def _software_type(unit: MemoryUnit) -> SoftwareUnitType | None:
        value = unit.metadata.get("software_type") or unit.raw_data.get("software_type")
        if value == SoftwareUnitType.SEMANTIC_BLOCK.value:
            return SoftwareUnitType.CODE_SEGMENT
        try:
            return SoftwareUnitType(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _deduplicate(units: Iterable[MemoryUnit]) -> list[MemoryUnit]:
        by_uid = {unit.uid: unit for unit in units}
        return [by_uid[uid] for uid in sorted(by_uid)]

    def add_software_unit(
        self,
        unit_or_schema: Any,
        *,
        space_names: Sequence[SoftwareSpace | str] | None = None,
        index_update_mode: str = "incremental",
        generate_sparse_embedding: bool = False,
    ) -> MemoryUnit:
        """Convert a schema if needed, then insert a standard MemoryUnit."""
        if isinstance(unit_or_schema, MemoryUnit):
            unit = unit_or_schema
        else:
            converter = getattr(unit_or_schema, "to_memory_unit", None)
            if not callable(converter):
                raise TypeError(
                    "add_software_unit() expects a MemoryUnit or software schema"
                )
            unit = converter()

        software_type = self._software_type(unit)
        if software_type is None:
            raise ValueError(
                "Software MemoryUnit must define a supported metadata['software_type']"
            )
        unit.metadata["software_type"] = software_type.value
        unit.raw_data["software_type"] = software_type.value

        if space_names is None:
            resolved_spaces = [software_space_for_type(software_type).value]
        else:
            if isinstance(space_names, (str, SoftwareSpace)):
                space_names = [space_names]
            resolved_spaces = [
                space.value if isinstance(space, SoftwareSpace) else str(space)
                for space in space_names
            ]

        self.semantic_graph.add_unit(
            unit,
            space_names=resolved_spaces,
            index_update_mode=index_update_mode,
            generate_sparse_embedding=generate_sparse_embedding,
        )
        return unit

    def add_relation(
        self,
        source_uid: str,
        target_uid: str,
        relation: SoftwareRelation,
        **properties: Any,
    ) -> bool:
        """Add a software relationship as an ordinary SemanticGraph edge."""
        relation_value = SoftwareRelation(relation).value
        return bool(
            self.semantic_graph.add_relationship(
                source_uid,
                target_uid,
                relation_value,
                **properties,
            )
        )

    def search(
        self,
        query: str,
        software_types: Sequence[SoftwareUnitType | str] | None = None,
        top_k: int = 5,
        *,
        return_score: bool = False,
        repo_uid: str | None = None,
        current_state_only: bool = False,
    ):
        """Search software units through Mandol's existing similarity API.

        ``current_state_only`` excludes session working-tree Evolution before
        ranking. Committed history stays eligible. Exact score ties in this
        view are ordered by UID, independent of incremental insertion order.
        The default remains the complete memory, including working history.
        """
        candidates = self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.SOFTWARE.value,
            recursive=True,
        )
        if repo_uid:
            candidates = [
                unit for unit in candidates if unit.metadata.get("repo_uid") == repo_uid
            ]
        if current_state_only:
            candidates = [unit for unit in candidates
                          if unit.raw_data.get("change_type") != "working_tree_change"]
        if software_types:
            if isinstance(software_types, (str, SoftwareUnitType)):
                software_types = [software_types]
            allowed_types = {SoftwareUnitType(value) for value in software_types}
            candidates = [
                unit
                for unit in candidates
                if self._software_type(unit) in allowed_types
            ]

        if not candidates or top_k <= 0:
            return []
        if current_state_only:
            # Fetch beyond the cutoff so UID tie-breaking also covers ties at
            # position k. Usually one query suffices; all-equal scores expand
            # to the candidate count without changing the Core index semantics.
            limit = min(len(candidates), max(32, top_k * 2))
            while True:
                matches = self.semantic_graph.search_similarity_in_graph(
                    query_text=query, top_k=limit,
                    ms_names=[SoftwareSpace.SOFTWARE.value],
                    candidate_units=candidates, return_score=True,
                )
                matches.sort(key=lambda pair: (-float(pair[1]), pair[0].uid))
                if (len(matches) < limit or limit == len(candidates)
                        or len(matches) <= top_k
                        or matches[top_k - 1][1] != matches[-1][1]):
                    selected = matches[:top_k]
                    return selected if return_score else [unit for unit, _ in selected]
                limit = min(len(candidates), limit * 2)
        return self.semantic_graph.search_similarity_in_graph(
            query_text=query,
            top_k=top_k,
            ms_names=[SoftwareSpace.SOFTWARE.value],
            candidate_units=candidates,
            return_score=return_score,
        )

    def get_repository(self, uid_or_name: str | None = None) -> MemoryUnit | None:
        """Return an ingested repository memory."""
        candidate = uid_or_name or getattr(self, "repository_uid", None)
        if candidate:
            uid = candidate if candidate.startswith("repo:") else f"repo:{candidate}"
            unit = self.semantic_graph.semantic_map.get_unit(uid)
            if (
                unit is not None
                and self._software_type(unit) is SoftwareUnitType.REPOSITORY
            ):
                return unit
        for unit in self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.CODE.value,
            recursive=True,
        ):
            if self._software_type(unit) is SoftwareUnitType.REPOSITORY:
                return unit
        return None

    def get_file(self, uid_or_path: str) -> MemoryUnit | None:
        """Return a source/configuration file by UID or repository-relative path."""
        uid = uid_or_path if uid_or_path.startswith("file:") else f"file:{uid_or_path}"
        unit = self.semantic_graph.semantic_map.get_unit(uid)
        return (
            unit
            if unit is not None and self._software_type(unit) is SoftwareUnitType.FILE
            else None
        )

    def get_function(self, uid_or_qualified_name: str) -> MemoryUnit | None:
        """Look up a function by UID or qualified name."""
        if uid_or_qualified_name.startswith("function:"):
            direct_candidates = [uid_or_qualified_name]
        else:
            direct_candidates = [
                f"function:{uid_or_qualified_name}",
                uid_or_qualified_name,
            ]

        for uid in direct_candidates:
            unit = self.semantic_graph.get_unit(uid)
            if (
                unit is not None
                and self._software_type(unit) is SoftwareUnitType.FUNCTION
            ):
                return unit

        for unit in self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.CODE.value,
            recursive=True,
        ):
            if (
                self._software_type(unit) is SoftwareUnitType.FUNCTION
                and unit.raw_data.get("qualified_name") == uid_or_qualified_name
            ):
                return unit
        return None

    def get_code_segment(self, uid_or_segment_id: str) -> MemoryUnit | None:
        """Return a CodeSegment by UID or an unambiguous segment ID."""
        if uid_or_segment_id.startswith(("segment:", "block:")):
            unit = self.semantic_graph.semantic_map.get_unit(uid_or_segment_id)
            if unit is not None and self._software_type(unit) is SoftwareUnitType.CODE_SEGMENT:
                return unit
        matches = [
            unit
            for unit in self.get_code_segments()
            if unit.raw_data.get("segment_id", unit.raw_data.get("block_id"))
            == uid_or_segment_id
        ]
        return matches[0] if len(matches) == 1 else None

    def get_code_segments(self, parent_uid: str | None = None) -> list[MemoryUnit]:
        """Return CodeSegments, optionally restricted to one owner UID."""
        units = [
            unit
            for unit in self.semantic_graph.get_units_in_memory_space(
                SoftwareSpace.CODE.value,
                recursive=True,
            )
            if self._software_type(unit) is SoftwareUnitType.CODE_SEGMENT
        ]
        if parent_uid is not None:
            units = [
                unit
                for unit in units
                if unit.raw_data.get("parent_uid") == parent_uid
                or unit.raw_data.get("function_uid") == parent_uid
            ]
        return sorted(
            units,
            key=lambda unit: (
                str(unit.raw_data.get("file_path", "")),
                int(unit.metadata.get("source_order", 0) or 0),
                int(unit.raw_data.get("start_line", 0) or 0),
                unit.uid,
            ),
        )

    def get_file_segments(self, uid_or_path: str) -> list[MemoryUnit]:
        """Return all CodeSegments located in a repository-relative file."""
        path = uid_or_path.split(":", 1)[1] if uid_or_path.startswith("file:") else uid_or_path
        return [
            unit
            for unit in self.get_code_segments()
            if unit.raw_data.get("file_path") == path
            or unit.metadata.get("file_path") == path
        ]

    def get_function_segments(self, function_uid: str) -> list[MemoryUnit]:
        """Return lexical CodeSegments owned by a function."""
        function = self.get_function(function_uid)
        return self.get_code_segments(function.uid if function else function_uid)

    def traverse(
        self,
        uid: str,
        relations: SoftwareRelation
        | str
        | Sequence[SoftwareRelation | str]
        | None = None,
        direction: str = "all",
    ) -> list[MemoryUnit]:
        """Traverse immediate explicit graph neighbors, optionally by edge type."""
        direction = {
            "outgoing": "successors",
            "incoming": "predecessors",
            "both": "all",
        }.get(direction, direction)

        if relations is None:
            return self._deduplicate(
                self.semantic_graph.traverse_explicit_nodes(
                    uid,
                    direction=direction,
                )
            )
        if isinstance(relations, (str, SoftwareRelation)):
            relations = [relations]

        results: list[MemoryUnit] = []
        for relation in relations:
            relation_value = SoftwareRelation(relation).value
            results.extend(
                self.semantic_graph.traverse_explicit_nodes(
                    uid,
                    relationship_type=relation_value,
                    direction=direction,
                )
            )
        return self._deduplicate(results)

    def get_related_commits(self, function_uid: str) -> list[MemoryUnit]:
        """Return commits directly modifying a function or cited by its history."""
        commits = [
            unit
            for unit in self.traverse(
                function_uid,
                SoftwareRelation.MODIFIES,
                direction="all",
            )
            if self._software_type(unit) is SoftwareUnitType.COMMIT
        ]
        for evolution in self.get_history(function_uid):
            commits.extend(
                unit
                for unit in self.traverse(
                    evolution.uid,
                    SoftwareRelation.CAUSED_BY_EVIDENCE,
                    direction="all",
                )
                if self._software_type(unit) is SoftwareUnitType.COMMIT
            )
        return self._deduplicate(commits)

    def get_related_issues(self, function_uid: str) -> list[MemoryUnit]:
        """Return issues linked directly, through commits, or through evolution."""
        issues: list[MemoryUnit] = []
        for relation in (SoftwareRelation.MENTIONS, SoftwareRelation.IMPACTS):
            issues.extend(
                unit
                for unit in self.traverse(function_uid, relation, direction="all")
                if self._software_type(unit) is SoftwareUnitType.ISSUE
            )

        for commit in self.get_related_commits(function_uid):
            issues.extend(
                unit
                for unit in self.traverse(
                    commit.uid,
                    SoftwareRelation.FIXED_BY,
                    direction="all",
                )
                if self._software_type(unit) is SoftwareUnitType.ISSUE
            )

        for evolution in self.get_history(function_uid):
            issues.extend(
                unit
                for unit in self.traverse(
                    evolution.uid,
                    SoftwareRelation.CAUSED_BY_EVIDENCE,
                    direction="all",
                )
                if self._software_type(unit) is SoftwareUnitType.ISSUE
            )
        return self._deduplicate(issues)

    def get_history(self, unit_uid: str) -> list[MemoryUnit]:
        """Return evolution nodes for a subject, including linked evolution chains."""
        evolution_units = [
            unit
            for unit in self.traverse(
                unit_uid,
                SoftwareRelation.HAS_EVOLUTION,
                direction="all",
            )
            if self._software_type(unit) is SoftwareUnitType.EVOLUTION
        ]
        evolution_units.extend(
            unit
            for unit in self.semantic_graph.get_units_in_memory_space(
                SoftwareSpace.EVOLUTION.value,
                recursive=True,
            )
            if unit.raw_data.get("subject_uid") == unit_uid
        )

        pending = list(self._deduplicate(evolution_units))
        by_uid = {unit.uid: unit for unit in pending}
        while pending:
            current = pending.pop()
            for linked in self.traverse(
                current.uid,
                SoftwareRelation.EVOLVES_FROM,
                direction="all",
            ):
                if (
                    self._software_type(linked) is SoftwareUnitType.EVOLUTION
                    and linked.uid not in by_uid
                ):
                    by_uid[linked.uid] = linked
                    pending.append(linked)

        return sorted(
            by_uid.values(),
            key=lambda unit: (*self._timestamp_key(unit), unit.uid),
        )

    @staticmethod
    def _timestamp_key(unit: MemoryUnit) -> tuple[float, str]:
        raw_timestamp = str(unit.raw_data.get("timestamp", ""))
        try:
            parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            return parsed.timestamp(), raw_timestamp
        except (TypeError, ValueError, OverflowError):
            return 0.0, raw_timestamp

    def get_module_memory(self, module_name: str) -> MemoryUnit | None:
        """Return a module-level high-order software memory."""
        direct = self.semantic_graph.get_unit(f"module_memory:{module_name}")
        if (
            direct is not None
            and self._software_type(direct) is SoftwareUnitType.MODULE_MEMORY
        ):
            return direct
        for unit in self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.HIGH_LEVEL.value,
            recursive=True,
        ):
            if (
                self._software_type(unit) is SoftwareUnitType.MODULE_MEMORY
                and unit.raw_data.get("module_name") == module_name
            ):
                return unit
        return None

    def get_file_memory(self, uid_or_path: str) -> MemoryUnit | None:
        """Return file-level high-order memory by UID or repository path."""
        uid = (
            uid_or_path
            if uid_or_path.startswith("file_memory:")
            else f"file_memory:{uid_or_path.removeprefix('file:')}"
        )
        unit = self.semantic_graph.semantic_map.get_unit(uid)
        if unit is not None and self._software_type(unit) is SoftwareUnitType.FILE_MEMORY:
            return unit
        return None

    def get_symbol_memory(self, uid_or_qualified_name: str) -> MemoryUnit | None:
        """Return the agglomerated memory for a Function or Class."""
        if uid_or_qualified_name.startswith(("function:", "class:")):
            subject_uid = uid_or_qualified_name
        else:
            function = self.get_function(uid_or_qualified_name)
            subject_uid = function.uid if function else uid_or_qualified_name
        key = subject_uid.split(":", 1)[-1]
        direct = self.semantic_graph.semantic_map.get_unit(f"symbol_memory:{key}")
        if (
            direct is not None
            and self._software_type(direct) is SoftwareUnitType.SYMBOL_MEMORY
        ):
            return direct
        for unit in self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.HIGH_LEVEL_SYMBOL.value,
            recursive=True,
        ):
            if unit.raw_data.get("subject_uid") == subject_uid:
                return unit
        return None

    def get_architecture_memory(self, scope: str | None = None) -> MemoryUnit | None:
        """Return the repository-level architecture memory."""
        resolved_scope = scope
        if resolved_scope is None:
            repository_uid = getattr(self, "repository_uid", "")
            resolved_scope = repository_uid.split(":", 1)[-1] if repository_uid else ""
        if resolved_scope:
            uid = (
                resolved_scope
                if resolved_scope.startswith("architecture_memory:")
                else f"architecture_memory:{resolved_scope}"
            )
            unit = self.semantic_graph.semantic_map.get_unit(uid)
            if (
                unit is not None
                and self._software_type(unit) is SoftwareUnitType.ARCHITECTURE_MEMORY
            ):
                return unit
        memories = self.semantic_graph.get_units_in_memory_space(
            SoftwareSpace.HIGH_LEVEL_ARCHITECTURE.value,
            recursive=True,
        )
        return memories[0] if memories else None

    def get_repairs_for_symbol(self, symbol_uid: str) -> list[MemoryUnit]:
        """Return repair memories whose provenance includes a symbol."""
        if not symbol_uid.startswith(("function:", "class:")):
            function = self.get_function(symbol_uid)
            if function:
                symbol_uid = function.uid
        return self._deduplicate(
            unit
            for unit in self.semantic_graph.get_units_in_memory_space(
                SoftwareSpace.HIGH_LEVEL_REPAIR.value,
                recursive=True,
            )
            if symbol_uid in unit.raw_data.get("affected_symbols", [])
            or symbol_uid in unit.raw_data.get("changed_symbols", [])
            or symbol_uid in unit.raw_data.get("derived_from", [])
        )

    def get_callers(self, function_uid: str) -> list[MemoryUnit]:
        return self.traverse(function_uid, SoftwareRelation.CALLS, direction="incoming")

    def get_callees(self, function_uid: str) -> list[MemoryUnit]:
        return self.traverse(function_uid, SoftwareRelation.CALLS, direction="outgoing")

    def get_call_sites(self, function_uid: str) -> list[dict[str, Any]]:
        """Return statically resolved call sites ordered by lexical source position.

        ``source_order`` describes source order only; it is not a claim about
        dynamic runtime execution order.
        """
        graph = self.semantic_graph
        node_index = graph._uid_to_index.get(function_uid)
        if node_index is None:
            return []
        call_sites: list[dict[str, Any]] = []
        for _, target_index, edge_data in graph.rx_graph.out_edges(node_index):
            if not edge_data or edge_data.get("type") != SoftwareRelation.CALLS.value:
                continue
            callee_uid = graph._index_to_uid.get(target_index, "")
            if not callee_uid:
                continue
            properties = {
                key: value
                for key, value in edge_data.items()
                if key not in {"type", "created", "updated"}
            }
            call_sites.append({"callee_uid": callee_uid, **properties})
        return sorted(
            call_sites,
            key=lambda item: (
                int(item.get("source_order", 0)),
                int(item.get("start_line", 0)),
                -1
                if item.get("start_column") is None
                else int(item["start_column"]),
                str(item.get("call_site_id", "")),
            ),
        )

    get_calls = get_call_sites

    def get_tests(self, symbol_uid: str) -> list[MemoryUnit]:
        return self.traverse(
            symbol_uid, SoftwareRelation.TESTED_BY, direction="outgoing"
        )

    def get_configs(self, symbol_uid: str) -> list[MemoryUnit]:
        return [
            unit
            for unit in self.traverse(
                symbol_uid,
                SoftwareRelation.AFFECTS_BEHAVIOR_OF,
                direction="incoming",
            )
            if self._software_type(unit) is SoftwareUnitType.CONFIG
        ]


SoftwareMemory = SemanticCode


__all__ = ["SemanticCode", "SoftwareMemory"]
