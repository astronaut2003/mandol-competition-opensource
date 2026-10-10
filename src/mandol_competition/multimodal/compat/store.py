"""Experimental graph adapter, available only by explicit development injection.

Formal Add/Search persistence and recovery are implemented by backend.py.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from mandol_substrate.representation import MemoryUnit

from ..config import MultimodalConfig
from .persistence import CanonicalStore

SPACE_NAME = "aml"


def _add_unit(graph: Any, unit: MemoryUnit, *, mode: str, use_splade: bool) -> None:
    graph.add_unit(
        unit,
        explicit_content_for_embedding=unit.raw_data.get("text_content", ""),
        content_type_for_embedding="text",
        space_names=[SPACE_NAME],
        index_update_mode=mode,
        generate_sparse_embedding=use_splade,
    )


def _restore_graph(
    graph: Any, store: CanonicalStore, user_id: str, config: MultimodalConfig
) -> Any:
    units = store.succeeded_units(user_id)
    for unit in units:
        _add_unit(graph, unit, mode="none", use_splade="splade" in config.retrieval_methods)
    if units:
        graph.rebuild_all_indexes()
    return graph


def build_mandol_graph(
    config: MultimodalConfig, store: CanonicalStore, user_id: str
) -> Any:
    """Build an index from succeeded units, never from a graph snapshot."""
    from mandol.core.semantic_graph import SemanticGraph

    if config.embedding_model_name:
        from mandol.core.semantic_map import SemanticMap

        graph = SemanticGraph(SemanticMap(
            embedding_model_name=config.embedding_model_name,
            embedding_dim=config.embedding_dim,
            use_flash_attention=False,
        ))
    else:
        graph = SemanticGraph()
    try:
        return _restore_graph(graph, store, user_id, config)
    except BaseException:
        graph.close()
        raise


class _Entry:
    def __init__(self, user_id: str, graph: Any) -> None:
        self.user_id = user_id
        self.graph = graph
        self.lock = threading.RLock()
        self.pins = 0
        self.last_used = time.monotonic()
        self.invalid = False


class GraphSession:
    """A graph lease; use transaction() across indexing and SQLite success."""

    def __init__(self, entry: _Entry, registry: UserGraphRegistry) -> None:
        self._entry = entry
        self._registry = registry
        self._released = False

    def _ensure_active(self) -> None:
        if self._released or self._entry.invalid:
            raise RuntimeError("Graph lease is released or invalidated")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Hold the per-user lock through add_units and mark_succeeded.

        If either step fails, retire the graph without snapshotting it. Other
        leases already waiting on this graph fail rather than see prepared units.
        """
        with self._entry.lock:
            self._ensure_active()
            try:
                yield
            except BaseException:
                self._registry._invalidate(self._entry)
                raise

    def add_units(self, units: list[MemoryUnit]) -> None:
        with self.transaction():
            for unit in units:
                if unit.metadata.get("user_id") != self._entry.user_id:
                    raise ValueError("MemoryUnit user_id does not match graph owner")
                _add_unit(
                    self._entry.graph, unit, mode="incremental",
                    use_splade="splade" in self._registry.config.retrieval_methods,
                )

    def search(
        self, query: str, *, top_k: int, rerank_method: str | None = None
    ) -> list[tuple[MemoryUnit, float]]:
        with self._entry.lock:
            self._ensure_active()
            results = self._entry.graph.get_multi_retriever().smart_search(
                query,
                methods=list(self._registry.config.retrieval_methods),
                top_k=top_k,
                fusion_method="rrf",
                rerank_method=rerank_method,
            )
            return list(results)

class UserGraphRegistry:
    """Lazily load user graphs, pin active leases, and evict unpinned LRU graphs."""

    def __init__(
        self,
        config: MultimodalConfig,
        store: CanonicalStore,
        *,
        graph_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._graph_factory = graph_factory
        self._entries: dict[str, _Entry] = {}
        self._loading: set[str] = set()
        self._retiring: set[str] = set()
        self._condition = threading.Condition()
        self._user_locks: dict[str, threading.RLock] = {}
        self._closed = False

    @contextmanager
    def user_operation(self, user_id: str) -> Iterator[None]:
        """Serialize one user's API work with purge, including media and VLM."""
        with self._condition:
            lock = self._user_locks.setdefault(user_id, threading.RLock())
        with lock:
            yield

    def _build(self, user_id: str) -> Any:
        if self._graph_factory is None:
            return build_mandol_graph(self.config, self.store, user_id)
        graph = self._graph_factory(user_id)
        try:
            return _restore_graph(graph, self.store, user_id, self.config)
        except BaseException:
            close = getattr(graph, "close", None)
            if close is not None:
                close()
            raise

    def _lease(self, user_id: str) -> _Entry:
        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError("Graph registry is closed")
                entry = self._entries.get(user_id)
                if entry is not None:
                    entry.pins += 1
                    entry.last_used = time.monotonic()
                    return entry
                if user_id not in self._loading and user_id not in self._retiring:
                    self._loading.add(user_id)
                    break
                self._condition.wait()

        try:
            entry = _Entry(user_id, self._build(user_id))
        except BaseException:
            with self._condition:
                self._loading.remove(user_id)
                self._condition.notify_all()
            raise

        with self._condition:
            self._loading.remove(user_id)
            closed = self._closed
            if closed:
                self._retiring.add(user_id)
            else:
                entry.pins = 1
                self._entries[user_id] = entry
            self._condition.notify_all()

        if closed:
            self._retire(entry)
            raise RuntimeError("Graph registry is closed")
        self._trim()
        return entry

    @contextmanager
    def acquire(self, user_id: str) -> Iterator[GraphSession]:
        entry = self._lease(user_id)
        session = GraphSession(entry, self)
        try:
            yield session
        finally:
            session._released = True
            with self._condition:
                entry.pins -= 1
                entry.last_used = time.monotonic()
                retire = entry.invalid and entry.pins == 0
                self._condition.notify_all()
            if retire:
                self._retire(entry)
            self._trim()

    def _invalidate(self, entry: _Entry) -> None:
        with self._condition:
            if entry.invalid:
                return
            entry.invalid = True
            if self._entries.get(entry.user_id) is entry:
                del self._entries[entry.user_id]
            self._retiring.add(entry.user_id)
            self._condition.notify_all()

    def _retire(self, entry: _Entry) -> None:
        try:
            with entry.lock:
                close = getattr(entry.graph, "close", None)
                if close is not None:
                    close()
        finally:
            with self._condition:
                self._retiring.remove(entry.user_id)
                self._condition.notify_all()

    def _trim(self) -> None:
        while True:
            with self._condition:
                if self._closed or len(self._entries) <= self.config.max_graphs:
                    return
                candidates = (entry for entry in self._entries.values() if entry.pins == 0)
                victim = min(candidates, key=lambda entry: entry.last_used, default=None)
                if victim is None:
                    return
                del self._entries[victim.user_id]
                self._retiring.add(victim.user_id)
            self._retire(victim)

    def evict(self, user_id: str) -> bool:
        with self._condition:
            entry = self._entries.get(user_id)
            if entry is None:
                return False
            if entry.pins:
                raise RuntimeError("Cannot purge a user with active graph leases")
            del self._entries[user_id]
            self._retiring.add(user_id)
        self._retire(entry)
        return True

    def loaded_count(self) -> int:
        with self._condition:
            return len(self._entries)

    def close(self) -> None:
        """Stop new leases, wait for active work, then close rebuildable indexes."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._loading or self._retiring or any(
                entry.pins for entry in self._entries.values()
            ):
                self._condition.wait()
            entries = list(self._entries.values())
            self._entries.clear()
            self._retiring.update(entry.user_id for entry in entries)
        for entry in entries:
            self._retire(entry)
