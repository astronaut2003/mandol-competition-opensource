"""Remove a user's canonical units and per-user derived media."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

from ..config import MultimodalConfig
from .persistence import CanonicalStore
from .store import UserGraphRegistry


@dataclass(frozen=True)
class PurgeReport:
    user_id: str
    graph_evicted: bool
    units_removed: int
    media_dir_removed: bool


def purge_user(
    *, user_id: str, config: MultimodalConfig, store: CanonicalStore, registry: UserGraphRegistry
) -> PurgeReport:
    with registry.user_operation(user_id):
        units_removed = len(store.succeeded_units(user_id))
        graph_evicted = registry.evict(user_id)
        store.delete_user(user_id)
        root = config.user_root(user_id)
        media_dir_removed = root.is_dir()
        if media_dir_removed:
            shutil.rmtree(root)
        return PurgeReport(user_id, graph_evicted, units_removed, media_dir_removed)
