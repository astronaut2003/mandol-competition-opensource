"""Textual composition over external projection and canonical SQLite.

No model or Qdrant process is created here. Frozen adapters still own graph
views, navigation, context rendering and result policy.
"""

from contextlib import contextmanager
from pathlib import Path

from mandol_substrate.contracts import PublicationPolicy, ResourceIdentity, SpacePolicy
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.spaces import MemorySpaceRegistry, TowerSpace
from mandol_substrate.state import SQLiteStateStore, Scope

from .core.graph import GraphViewManager
from .core.profile import load_profile
from ..runtime.storage import claim_runtime_root


def seed_frozen_capsule(root, family="source_b1_public_v1"):
    if family not in {"source_b1_public_v1", "aml_official_v4_v1"}:
        raise ValueError("unadmitted Textual quality family")
    root = claim_runtime_root(root, "textual")
    source = Path(__file__).parent / "profiles" / family
    for name in ("profile.json", "quality-profile.json", "quality-policies.json"):
        target = root / name
        expected = (source / name).read_bytes()
        if target.exists():
            if target.is_symlink() or target.read_bytes() != expected:
                raise ValueError("runtime quality capsule differs from frozen Textual profile")
        else:
            with target.open("xb") as stream:
                stream.write(expected)
    return root


class ExternalCompetitionRun:
    def __init__(self, settings):
        self.root = Path(settings.competition_run_root)
        self.profile = load_profile(self.root / "profile.json")
        identity = ResourceIdentity("textual", settings.evaluation_namespace, settings.qdrant_collection)
        spaces = SpacePolicy(tuple(s.value for s in TowerSpace), tuple(
            (parent.value, child.value)
            for parent, children in ((TowerSpace.HIERARCHICAL_ROOT, MemorySpaceRegistry.HIERARCHICAL_CHILDREN),
                                     (TowerSpace.GRAPH_ROOT, MemorySpaceRegistry.GRAPH_CHILDREN))
            for child in children))
        self.state = SQLiteStateStore(self.root / "state.sqlite3", self.profile,
                                     identity=identity, space_policy=spaces)
        try:
            self.vector = QdrantProjectionBackend(settings.qdrant_endpoint, self.profile, state=self.state,
                storage_id=self.state.db_uuid, identity=identity, collection_name=settings.qdrant_collection,
                api_key=settings.qdrant_api_key)
            self.graphs = GraphViewManager(self.state)
        except BaseException:
            self.state.close()
            raise
        self.coordinator = None

    def finish_startup(self, encoders):
        self.coordinator = PublicationCoordinator(self.state, self.vector, self.vector.codec, encoders,
            self.graphs, policy=PublicationPolicy("textual-strict-vector-v1"))
        from mandol_substrate.maintenance import ResourcePurger
        purger = ResourcePurger(self.state, self.vector, cleanup=self.graphs.evict)
        for scope in self.state.iter_scopes():
            purger.complete(scope)
        self.coordinator.recover()
        self.vector.ensure_schema(Scope(self.state.identity.namespace, "__startup_probe__"), create=True)

    def ensure_scope(self, scope):
        self.state.ensure_scope(scope)
        self.vector.ensure_schema(scope, create=self.state.scope_state(scope)["canonical_seq"] == 0)

    @contextmanager
    def maintenance_view(self, scope):
        with self.coordinator.gates.lease(scope, write=True):
            yield

    def health(self):
        return self.state.health() and self.vector.health()

    def close(self):
        self.vector.close()
        self.state.close()
