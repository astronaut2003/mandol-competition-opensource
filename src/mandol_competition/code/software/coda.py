"""Staged code-domain entry point built on SemanticCode and Mandol."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .facade import SemanticCode
from .relations import SoftwareRelation
from .schemas import SoftwareUnitType

if TYPE_CHECKING:
    from ..core.semantic_graph import SemanticGraph
    from .builder.repository_builder import RepositoryBuildReport


_PHASES = (
    "basic_memory",
    "relations",
    "git_history",
    "high_level_memory",
    "indexes",
)


class Coda(SemanticCode):
    """Unified, staged software-memory facade over the existing Mandol graph.

    The default repository import performs deterministic scanning, parsing and
    base MemoryUnit insertion only. More expensive or optional stages are
    explicit methods and reuse the repository builder's file cache.
    """

    def __init__(
        self,
        semantic_graph: SemanticGraph | None = None,
        *,
        initialize_spaces: bool = True,
    ) -> None:
        super().__init__(semantic_graph, initialize_spaces=initialize_spaces)
        self._phase_state = {phase: False for phase in _PHASES}
        self._coda_builder_options: dict[str, Any] = {}

    @classmethod
    def from_repository(
        cls,
        repo_path: str,
        *,
        revision: str = "HEAD",
        build_basic_memory: bool = True,
        build_relations: bool = False,
        include_git_history: bool = False,
        build_high_level_memory: bool = False,
        build_indexes: bool = False,
        incremental: bool = True,
        semantic_graph: SemanticGraph | None = None,
        **builder_options: Any,
    ) -> Self:
        """Create Coda with independently selectable repository build stages."""
        coda = cls(semantic_graph)
        coda.repository_path = str(Path(repo_path).expanduser().resolve())
        coda.repository_uid = f"repo:{Path(coda.repository_path).name}"
        coda._coda_builder_options = {
            "revision": revision,
            "incremental": incremental,
            **builder_options,
        }

        requested_build = any(
            (
                build_basic_memory,
                build_relations,
                include_git_history,
                build_high_level_memory,
                build_indexes,
            )
        )
        if requested_build:
            effective_relations = build_relations or build_high_level_memory
            coda._run_builder(
                build_relations=effective_relations,
                include_git_history=include_git_history,
                build_high_level_memory=build_high_level_memory,
                build_indexes=build_indexes,
            )
            coda._phase_state.update(
                {
                    "basic_memory": True,
                    "relations": effective_relations,
                    "git_history": include_git_history,
                    "high_level_memory": build_high_level_memory,
                    "indexes": build_indexes,
                }
            )
            coda._persist_phase_state()
        return coda

    def status(self) -> dict[str, bool]:
        """Return completed repository build phases."""
        return dict(self._phase_state)

    def build_relations(self) -> Self:
        """Build static relations from cached parsed facts."""
        self._run_builder(
            build_relations=True,
            include_git_history=False,
            build_high_level_memory=False,
            build_indexes=False,
        )
        self._phase_state.update({"basic_memory": True, "relations": True})
        self._persist_phase_state()
        return self

    def build_git_history(self) -> Self:
        """Add Commit/Evolution units and their evidence relationships."""
        self._run_builder(
            build_relations=self._phase_state["relations"],
            include_git_history=True,
            build_high_level_memory=False,
            build_indexes=False,
        )
        self._phase_state.update({"basic_memory": True, "git_history": True})
        self._persist_phase_state()
        return self

    def build_high_level_memory(self) -> Self:
        """Agglomerate high-level memories after ensuring static relations exist."""
        self._run_builder(
            build_relations=True,
            include_git_history=self._phase_state["git_history"],
            build_high_level_memory=True,
            build_indexes=False,
        )
        self._phase_state.update(
            {
                "basic_memory": True,
                "relations": True,
                "high_level_memory": True,
            }
        )
        self._persist_phase_state()
        return self

    def build_indexes(self) -> Self:
        """Build Mandol's existing retrieval indexes for stored units."""
        self.semantic_graph.semantic_map.build_index()
        self._phase_state["indexes"] = True
        self._persist_phase_state()
        return self

    def sync_repository_changes(self, **overrides: Any) -> RepositoryBuildReport:
        """Incrementally synchronize a mutable worktree with the current graph.

        This is a small staged-facade wrapper around the existing repository
        builder. It preserves the phases already enabled on this Coda instance,
        updates their persisted state, and returns the detailed build report so
        an agent integration can audit exactly what changed after a shell step.
        """
        options = {
            "build_relations": self._phase_state["relations"],
            "include_git_history": self._phase_state["git_history"],
            "build_high_level_memory": self._phase_state["high_level_memory"],
            "build_indexes": self._phase_state["indexes"],
            **overrides,
        }
        super().update_repository(**options)
        self._persist_phase_state()
        return self.last_build_report

    def save(self, path: str | Path, **kwargs: Any) -> None:
        """Persist through ``SemanticGraph.save_graph`` without a Coda database."""
        from ..retrieval.retrieval_interface import RetrievalMethod

        self._persist_phase_state()
        kwargs.setdefault("build_sparse_vectors", False)
        # SemanticGraph's omitted/empty retriever selection means "build the
        # default BM25/SPLADE set". Keep Coda.save lightweight unless callers
        # explicitly request those indexes; the dense SemanticMap state is
        # still persisted by save_graph itself.
        kwargs.setdefault(
            "retriever_methods_to_build",
            [RetrievalMethod.COSINE_SIMILARITY],
        )
        self.semantic_graph.save_graph(str(path), **kwargs)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> Self:
        """Restore graph data, repository identity and staged build status."""
        from ..core.semantic_graph import SemanticGraph

        graph = SemanticGraph.load_graph(str(path), **kwargs)
        coda = cls(graph)
        coda._restore_repository_state()
        return coda

    def _run_builder(self, **stage_options: Any) -> None:
        from .builder import RepositoryMemoryBuilder

        repo_path = getattr(self, "repository_path", "")
        if not repo_path:
            raise ValueError("No repository path is attached to this Coda instance")
        options = dict(self._coda_builder_options)
        options.update(stage_options)
        RepositoryMemoryBuilder(
            repo_path,
            semantic_code=self,
            **options,
        ).build()
        runtime_options = dict(getattr(self, "_repository_builder_options", {}))
        for stage_name in (
            "build_relations",
            "include_git_history",
            "build_high_level_memory",
            "build_indexes",
        ):
            runtime_options.pop(stage_name, None)
        self._coda_builder_options = runtime_options

    def _persist_phase_state(self) -> None:
        repository = self.get_repository(getattr(self, "repository_uid", None))
        if repository is None:
            return
        repository.metadata["coda"] = {
            "version": 2,
            "phases": self.status(),
            "repository_uid": repository.uid,
            "repository_path": getattr(
                self, "repository_path", repository.raw_data.get("path", "")
            ),
            "revision": repository.raw_data.get("revision", "HEAD"),
            "incremental": bool(self._coda_builder_options.get("incremental", True)),
            "builder_options": {
                key: value
                for key, value in getattr(self, "_repository_builder_options", self._coda_builder_options).items()
                if key in {
                    "revision", "incremental", "cache_path", "max_workers",
                    "max_git_commits", "batch_size", "max_file_size",
                    "build_code_segments", "max_file_context_chars",
                    "semantic_enricher_version", "enrich_symbols", "enrich_repairs",
                }
            },
        }

    def _restore_repository_state(self) -> None:
        repository = self.get_repository()
        if repository is None:
            return
        state = repository.metadata.get("coda", {})
        stored_phases = state.get("phases", {})
        inferred = self._infer_phase_state(repository.uid)
        self._phase_state = {
            phase: bool(stored_phases.get(phase, inferred[phase])) for phase in _PHASES
        }
        self.repository_uid = str(state.get("repository_uid") or repository.uid)
        self.repository_path = str(
            state.get("repository_path") or repository.raw_data.get("path", "")
        )
        self._coda_builder_options = {
            "revision": str(
                state.get("revision") or repository.raw_data.get("revision", "HEAD")
            ),
            "incremental": bool(state.get("incremental", True)),
            **state.get("builder_options", {}),
        }
        self._repository_builder_options = dict(self._coda_builder_options)

    def _infer_phase_state(self, repo_uid: str) -> dict[str, bool]:
        repo_units = [
            unit
            for unit in self.semantic_graph.semantic_map.get_all_units()
            if unit.metadata.get("repo_uid") == repo_uid
        ]
        unit_types = {self._software_type(unit) for unit in repo_units}
        base_relation_types = {
            SoftwareRelation.CONTAINS.value,
            SoftwareRelation.CONTAINED_IN.value,
            SoftwareRelation.HAS_METHOD.value,
            SoftwareRelation.IMPORTS.value,
            SoftwareRelation.CALLS.value,
        }
        has_relations = any(
            data and data.get("type") in base_relation_types
            for edge_index in self.semantic_graph.rx_graph.edge_indices()
            for data in [
                self.semantic_graph.rx_graph.get_edge_data_by_index(edge_index)
            ]
        )
        index = getattr(self.semantic_graph.semantic_map, "faiss_index", None)
        return {
            "basic_memory": SoftwareUnitType.REPOSITORY in unit_types,
            "relations": has_relations,
            "git_history": bool(
                {SoftwareUnitType.COMMIT, SoftwareUnitType.EVOLUTION} & unit_types
            ),
            "high_level_memory": bool(
                {
                    SoftwareUnitType.FILE_MEMORY,
                    SoftwareUnitType.SYMBOL_MEMORY,
                    SoftwareUnitType.REPAIR_MEMORY,
                    SoftwareUnitType.MODULE_MEMORY,
                    SoftwareUnitType.ARCHITECTURE_MEMORY,
                }
                & unit_types
            ),
            "indexes": bool(index is not None and getattr(index, "ntotal", 0)),
        }


__all__ = ["Coda"]
