"""Self-describing, exclusively owned dataset/run storage and portable checkpoints.

The run barrier freezes *admission*, not the semantic publication protocol. Its
shared leases permit independent users concurrently. Only an explicit checkpoint
or lifecycle operation takes the exclusive lease. SQLite remains authoritative;
manifest frontiers are observations and are never used to serve a query.
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path

from qdrant_client import models

from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.qdrant.service import QdrantServiceManager, atomic_json
from mandol_substrate.state import SQLiteStateStore
from mandol_substrate.state.models import RetrievalProfile, Scope, StateUnavailable, digest
from mandol_substrate.state.schema import SCHEMA_VERSION
from mandol_substrate.vector.verification import verify_projection
from .gates import ScopeGates
from .graph import GraphViewManager
from .publication import PublicationCoordinator
from .run_quality import CAPSULE_NAME, capture_capsule, load_capsule, validate_capsule
from .text import MandolTextProcessor

LAYOUT_VERSION = 1


def checksum(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checked_path(path, *, allow_scratch=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("run/checkpoint path must not traverse symlinks")
    if path == Path("/") or (
        str(path).startswith(("/mnt/", "/tmp/")) and not allow_scratch
    ):
        raise ValueError("persistent Linux run filesystem required")
    return path


def slug(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value) or value in {
        ".",
        "..",
    }:
        raise ValueError("unsafe dataset/run/checkpoint name")
    return value


def fsync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fork_database_identity(connection):
    """An independent branch cannot reuse another branch's graph cache key.

    Only called on an offline, newly created canonical copy. Semantic rows,
    logical identities, projection IDs and publication frontiers stay intact.
    """
    source_uuid = connection.execute(
        "SELECT value FROM manifest WHERE key='db_uuid'"
    ).fetchone()[0]
    new_uuid = uuid.uuid4().hex
    connection.execute("UPDATE manifest SET value=? WHERE key='db_uuid'", (new_uuid,))
    return source_uuid, new_uuid


def git_identity():
    def read(*args):
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)

    try:
        root = Path(read("rev-parse", "--show-toplevel").decode().strip())
        code = {
            str(path.relative_to(root)): checksum(path)
            for directory in ("src", "scripts")
            for path in (root / directory).rglob("*.py")
            if "__pycache__" not in path.parts
        }
        code.update(
            {name: checksum(root / name) for name in ("pyproject.toml", "uv.lock")}
        )
        return {
            "head": read("rev-parse", "HEAD").decode().strip(),
            "tracked_diff_sha256": hashlib.sha256(read("diff", "HEAD")).hexdigest(),
            "status_sha256": hashlib.sha256(
                read("status", "--porcelain=v1", "-uall")
            ).hexdigest(),
            "code_files": code,
            "code_fingerprint": digest(code),
        }
    except subprocess.CalledProcessError:
        return {"head": "UNAVAILABLE"}


class RunPublicationCoordinator(PublicationCoordinator):
    def __init__(self, *args, admission, admission_scope, **kwargs):
        super().__init__(*args, **kwargs)
        self.admission, self.admission_scope = admission, admission_scope

    def apply(self, *args, **kwargs):
        with self.admission.lease(self.admission_scope):
            return super().apply(*args, **kwargs)

    def recover(self):
        with self.admission.lease(self.admission_scope):
            return super().recover()

    @contextmanager
    def search_view(self, scope):
        with (
            self.admission.lease(self.admission_scope),
            super().search_view(scope) as view,
        ):
            yield view


class CompetitionRun:
    """Opened run. Mutations and searches use coordinator admission leases.

    Direct state/vector attributes support inspection and explicit maintenance;
    callers must not bypass the coordinator to mutate a serving run.
    """

    def __init__(
        self,
        root,
        *,
        encoders,
        binary=None,
        allow_scratch=False,
        fault=None,
        reproject=False,
        defer_recovery=False,
    ):
        self.root = checked_path(root, allow_scratch=allow_scratch)
        for name in (
            "manifest.json",
            "profile.json",
            "quality-profile.json",
            "projection-policy.json",
            "run.owner",
        ):
            checked_path(self.root / name, allow_scratch=allow_scratch)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("storage_layout_version") != LAYOUT_VERSION:
            raise StateUnavailable("unknown run layout")
        if self.manifest.get("status") == "REPROJECTING" and not reproject:
            raise StateUnavailable("explicit reproject admission required")
        if self.manifest.get("status") not in {
            "CREATED",
            "RESTORING",
            "RESTORED",
            "OPEN",
            "REPROJECTING",
        }:
            raise StateUnavailable("run creation/clone is incomplete")
        self.storage_id = self.manifest["storage_id"]
        self._owner = (self.root / "run.owner").open("a+b")
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._owner.close()
            raise
        self.state = self.vector = self.service = None
        self._closed = False
        self.ready = False
        self.fault = fault or (lambda point: None)
        self.admission = ScopeGates(timeout=300, max_inflight_scopes=1)
        self.admission_scope = Scope("run-checkpoint", self.storage_id)
        try:
            profile_data = json.loads((self.root / "profile.json").read_text())
            self.profile = RetrievalProfile(**profile_data)
            self.policy = QdrantProjectionPolicy(
                **json.loads((self.root / "projection-policy.json").read_text())
            )
            self.quality_profile = json.loads(
                (self.root / "quality-profile.json").read_text()
            )
            if (
                self.profile.hash != self.manifest["semantic_profile_hash"]
                or self.policy.hash != self.manifest["projection_policy_hash"]
                or digest(self.quality_profile) != self.manifest["quality_profile_hash"]
            ):
                raise StateUnavailable("run profile fingerprint mismatch")
            if self.manifest.get("quality_capsule_hash"):
                capsule = load_capsule(self.root, self.quality_profile, self.profile)
                if digest(capsule) != self.manifest["quality_capsule_hash"]:
                    raise StateUnavailable("run quality capsule fingerprint mismatch")
            for relative in (
                "sqlite",
                "sqlite/canonical.sqlite",
                "qdrant",
                "checkpoints",
                "logs",
            ):
                checked_path(self.root / relative, allow_scratch=allow_scratch)
            processor = MandolTextProcessor(
                mode=self.profile.options.get("text_mode", "regex"),
                use_jieba=self.profile.options.get("jieba", False),
            )
            self.state = SQLiteStateStore(
                self.root / "sqlite/canonical.sqlite",
                self.profile,
                processor=processor,
                fault=self.fault,
            )
            deployment = self.manifest["deployment"]
            self.service = QdrantServiceManager(
                self.root,
                self.storage_id,
                binary=binary,
                allow_scratch=allow_scratch,
                **deployment,
            )
            self.startup = self.service.start()
            self.vector = QdrantProjectionBackend(
                self.service.uri,
                self.profile,
                state=self.state,
                storage_id=self.storage_id,
                policy=self.policy,
            )
            graph_options = {
                key: self.profile.options["graph_" + key]
                for key in ("max_users", "max_nodes", "max_edges", "max_bytes")
                if "graph_" + key in self.profile.options
            }
            self.graphs = GraphViewManager(self.state, **graph_options)
            self.coordinator = RunPublicationCoordinator(
                self.state,
                self.vector,
                self.vector.codec,
                encoders,
                self.graphs,
                admission=self.admission,
                admission_scope=self.admission_scope,
                fault=self.fault,
            )
            # Empty runs still own a real collection, including before first Add.
            if self.manifest.get("status") == "RESTORING":
                self._restore_projection()
            creating_projection = self.manifest.get("status") == "REPROJECTING"
            # Explicit canonical reprojection is distinct from a new user's
            # first Add; only the former may recreate a published collection.
            self.vector.explicit_rebuild = creating_projection
            try:
                self.vector.ensure_schema(
                    None,
                    create=creating_projection
                    or (
                        self.state.pending_count() == 0
                        and not any(
                            self.state.scope_state(s)["canonical_seq"]
                            for s in self.state.iter_scopes()
                        )
                    ),
                )
            finally:
                self.vector.explicit_rebuild = False
            if not defer_recovery:
                self.finish_startup()
        except BaseException:
            self.close()
            raise

    def finish_startup(self, encoders=None):
        if encoders is not None:
            self.coordinator.encoders = encoders
        if self.coordinator.encoders is None:
            raise StateUnavailable(
                "qualified document encoders required before recovery"
            )
        self.coordinator.recover()
        for scope in self.state.iter_scopes():
            self.coordinator.audit_scope(scope)
        report = verify_projection(self.state, self.vector)
        if not report["servable"]:
            raise StateUnavailable("run projection cannot reconcile")
        self.ready = True
        self.observe(status="OPEN", verification=report)

    def ensure_scope(self, scope):
        with (
            self.admission.lease(self.admission_scope),
            self.coordinator.gates.lease(scope, write=True),
        ):
            self.state.ensure_scope(scope)
            self.vector.ensure_schema(scope, create=False)

    @contextmanager
    def maintenance_view(self, scope):
        with (
            self.admission.lease(self.admission_scope),
            self.coordinator.gates.lease(scope),
        ):
            yield

    def frontiers(self):
        result = []
        for scope in self.state.iter_scopes():
            row = self.state.scope_state(scope)
            result.append(
                {
                    "namespace": scope.namespace,
                    "user_id": scope.user_id,
                    "scope_key": scope.key,
                    **{
                        k: row[k]
                        for k in ("canonical_seq", "lexical_seq", "published_seq")
                    },
                }
            )
        return result

    def observe(self, **updates):
        self.manifest.update(
            updates,
            observed_at=time.time(),
            frontiers=self.frontiers(),
            canonical_db_uuid=self.state.db_uuid,
        )
        atomic_json(self.root / "manifest.json", self.manifest)

    def health(self):
        if not self.ready:
            raise StateUnavailable("run not ready")
        self.state.health()
        self.service.health()
        self.vector.ensure_schema(None, create=False)
        if self.state.pending_count():
            raise StateUnavailable("unpublished run mutation")
        return True

    def checkpoint(self, name):
        slug(name)
        if not self.ready:
            raise StateUnavailable("cannot checkpoint an unready run")
        target = self.root / "checkpoints" / name
        stage = target.with_name(name + ".tmp")
        if target.exists() or stage.exists():
            raise FileExistsError("checkpoint name already exists; evidence preserved")
        with (
            self.admission.lease(self.admission_scope, write=True),
            self.state.quiescent_access(),
        ):
            report = verify_projection(self.state, self.vector)
            if not report["servable"]:
                raise StateUnavailable("checkpoint projection is inconsistent")
            frontiers = self.frontiers()
            stage.mkdir()
            self.fault("checkpoint_before_sqlite_backup")
            self.state.backup(stage / "canonical.sqlite")
            fsync_file(stage / "canonical.sqlite")
            self.fault("checkpoint_after_sqlite_backup")
            snap = self.vector.client.create_snapshot(
                self.vector.collection_name, wait=True
            )
            source = (
                self.service.qroot
                / "snapshots"
                / self.vector.collection_name
                / snap.name
            )
            checked_path(source, allow_scratch=True)
            # Same run filesystem: rename the completed snapshot, no second copy.
            os.replace(source, stage / "qdrant.snapshot")
            fsync_file(stage / "qdrant.snapshot")
            self.fault("checkpoint_after_qdrant_snapshot")
            policy_files = [
                "profile.json",
                "quality-profile.json",
                "projection-policy.json",
            ]
            if self.manifest.get("quality_capsule_hash"):
                capsule = load_capsule(self.root, self.quality_profile, self.profile)
                if digest(capsule) != self.manifest["quality_capsule_hash"]:
                    raise StateUnavailable("checkpoint quality capsule mismatch")
                policy_files.append(CAPSULE_NAME)
            for name_ in policy_files:
                shutil.copyfile(self.root / name_, stage / name_)
                fsync_file(stage / name_)
            data = {
                "checkpoint_format": 2 if CAPSULE_NAME in policy_files else 1,
                "created_at": time.time(),
                "source_storage_id": self.storage_id,
                "db_uuid": self.state.db_uuid,
                "frontiers": frontiers,
                "semantic_profile_hash": self.profile.hash,
                "quality_profile_hash": digest(self.quality_profile),
                "projection_policy_hash": self.policy.hash,
                "server_version": self.policy.server_version,
                "client_version": self.policy.client_version,
                "dataset": self.manifest["dataset"],
                "dataset_fingerprint": self.manifest["dataset_fingerprint"],
                "collection": self.vector.collection_name,
                "verification": report,
                "files": {
                    p: checksum(stage / p)
                    for p in ["canonical.sqlite", "qdrant.snapshot", *policy_files]
                },
            }
            atomic_json(stage / "checkpoint.json", data)
            self.fault("checkpoint_before_publish")
            os.rename(stage, target)
            fsync_directory(target.parent)
            self.observe(
                snapshot_state={
                    "path": str(target.relative_to(self.root)),
                    "manifest_sha256": checksum(target / "checkpoint.json"),
                }
            )
            return target

    def _restore_projection(self):
        data = json.loads((self.root / "artifacts/restore-checkpoint.json").read_text())
        source = self.root / "qdrant/snapshots/restore.snapshot"
        if checksum(source) != data["files"]["qdrant.snapshot"]:
            raise StateUnavailable("restore snapshot checksum mismatch")
        # The SQLite copy can never be advanced before successful restore/admission.
        if (
            self.frontiers() != data["frontiers"]
            or self.state.db_uuid != self.manifest["canonical_db_uuid"]
            or self.manifest["lineage"]["source_db_uuid"] != data["db_uuid"]
        ):
            raise StateUnavailable("restore canonical frontier mismatch")
        location = (
            source.as_uri()
            if self.service.mode == "native"
            else "file:///run/qdrant/snapshots/restore.snapshot"
        )
        expected = {
            **self.vector.collection_metadata,
            "mandol_storage_id": data["source_storage_id"],
        }
        metadata = (
            self.vector.client.get_collection(
                self.vector.collection_name
            ).config.metadata
            if self.vector.client.collection_exists(self.vector.collection_name)
            else None
        )
        if metadata not in (None, expected, self.vector.collection_metadata):
            raise StateUnavailable(
                "restore destination belongs to a different projection contract"
            )
        # A crash after reownership, or a storage-only CLI verification, must
        # not replay a snapshot over an already restored destination. Admission
        # still requires the exact checkpoint frontier and complete verification.
        if metadata != self.vector.collection_metadata:
            self.vector.client.recover_snapshot(
                self.vector.collection_name,
                location=location,
                priority=models.SnapshotPriority.SNAPSHOT,
                wait=True,
            )
            self.fault("restore_after_snapshot")
            metadata = self.vector.client.get_collection(
                self.vector.collection_name
            ).config.metadata
            if metadata != expected:
                raise StateUnavailable(
                    "snapshot belongs to a different projection contract"
                )
        # Explicit clone changes only physical run ownership, never semantic/point IDs.
        self.vector.client.update_collection(
            self.vector.collection_name, metadata=self.vector.collection_metadata
        )
        self.fault("restore_after_reownership")
        self.vector.ensure_schema(None, create=False)
        report = verify_projection(self.state, self.vector)
        if not report["servable"]:
            raise StateUnavailable(
                "restored projection differs from canonical checkpoint"
            )
        self.fault("restore_before_receipt")
        self.observe(status="RESTORED", restore_verification=report)

    def close(self):
        if self._closed:
            return
        with self.admission.lease(self.admission_scope, write=True):
            self._close_resources()

    def _close_resources(self):
        self.ready = False
        try:
            if self.vector:
                self.vector.close()
        finally:
            try:
                if self.service:
                    self.service.close()
            finally:
                try:
                    if self.state:
                        self.state.close()
                finally:
                    self._owner.close()
                    self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class RunStorageManager:
    def __init__(self, root, *, allow_scratch=False):
        self.root = checked_path(root, allow_scratch=allow_scratch)
        self.allow_scratch = allow_scratch

    def path(self, dataset, run_id):
        return checked_path(
            self.root / slug(dataset) / slug(run_id), allow_scratch=self.allow_scratch
        )

    def create(
        self,
        dataset,
        run_id,
        *,
        dataset_fingerprint,
        profile,
        quality_profile,
        deployment,
        policy=None,
        initial_status="CREATED",
        quality_capsule=None,
    ):
        if not re.fullmatch("[a-f0-9]{64}", dataset_fingerprint):
            raise ValueError("explicit dataset SHA256/fingerprint required")
        if initial_status not in {"CREATED", "COPYING_CANONICAL", "CLONING"}:
            raise ValueError("invalid initial run admission state")
        policy = policy or QdrantProjectionPolicy()
        if quality_capsule is not None:
            validate_capsule(quality_capsule, quality_profile, profile)
        root = self.path(dataset, run_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()  # exclusive; never overwrite a previous run
        for relative in (
            "sqlite",
            "qdrant/storage",
            "qdrant/snapshots",
            "checkpoints",
            "artifacts",
            "logs",
            "qualification",
        ):
            (root / relative).mkdir(parents=True)
        atomic_json(root / "profile.json", asdict(profile))
        atomic_json(root / "quality-profile.json", quality_profile)
        atomic_json(root / "projection-policy.json", asdict(policy))
        if quality_capsule is not None:
            atomic_json(root / CAPSULE_NAME, quality_capsule)
        atomic_json(
            root / "manifest.json",
            {
                "storage_layout_version": LAYOUT_VERSION,
                "semantic_schema_version": SCHEMA_VERSION,
                "storage_id": uuid.uuid4().hex,
                "dataset": dataset,
                "dataset_fingerprint": dataset_fingerprint,
                "run_id": run_id,
                "created_at": time.time(),
                "git": git_identity(),
                "paths": {
                    "sqlite": "sqlite/canonical.sqlite",
                    "vector": "qdrant/storage",
                    "checkpoints": "checkpoints",
                    "artifacts": "artifacts",
                },
                "vector_backend": "qdrant",
                "collections": ["memories"],
                "deployment": deployment,
                "semantic_profile_hash": profile.hash,
                "quality_profile_hash": digest(quality_profile),
                "quality_capsule_hash": digest(quality_capsule)
                if quality_capsule is not None
                else None,
                "projection_policy_hash": policy.hash,
                "status": initial_status,
                "frontiers": [],
                "snapshot_state": None,
                "qualification_state": "NOT_QUALIFIED",
            },
        )
        return root

    def package_quality(self, dataset, run_id):
        """Add portable policy bytes to a stopped run without changing its profile."""
        root = self.path(dataset, run_id)
        for name in (
            "run.owner",
            "manifest.json",
            "quality-profile.json",
            "profile.json",
            CAPSULE_NAME,
        ):
            checked_path(root / name, allow_scratch=self.allow_scratch)
        with (root / "run.owner").open("a+b") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = json.loads((root / "manifest.json").read_text())
            quality = json.loads((root / "quality-profile.json").read_text())
            profile = RetrievalProfile(
                **json.loads((root / "profile.json").read_text())
            )
            if (
                digest(quality) != manifest["quality_profile_hash"]
                or profile.hash != manifest["semantic_profile_hash"]
            ):
                raise StateUnavailable("run profile mismatch before packaging")
            capsule = (
                load_capsule(root, quality, profile)
                if (root / CAPSULE_NAME).exists()
                else capture_capsule(quality, profile)
            )
            if manifest.get("quality_capsule_hash") not in {None, digest(capsule)}:
                raise StateUnavailable("refusing to replace existing quality capsule")
            atomic_json(root / CAPSULE_NAME, capsule)
            manifest["quality_capsule_hash"] = digest(capsule)
            atomic_json(root / "manifest.json", manifest)
            return root / CAPSULE_NAME

    def open(self, dataset, run_id, **kwargs):
        return CompetitionRun(
            self.path(dataset, run_id), allow_scratch=self.allow_scratch, **kwargs
        )

    def clone(self, checkpoint, run_id, *, deployment):
        source = checked_path(checkpoint, allow_scratch=self.allow_scratch)
        if source.name.endswith(".tmp"):
            raise StateUnavailable("incomplete checkpoint is not restorable")
        data = json.loads((source / "checkpoint.json").read_text())
        expected_files = {
            "canonical.sqlite",
            "qdrant.snapshot",
            "profile.json",
            "quality-profile.json",
            "projection-policy.json",
        }
        if data.get("checkpoint_format") == 2:
            expected_files.add(CAPSULE_NAME)
        if (
            data.get("checkpoint_format") not in {1, 2}
            or set(data["files"]) != expected_files
        ):
            raise StateUnavailable("checkpoint format/file set mismatch")
        for name, expected in data["files"].items():
            checked_path(source / name, allow_scratch=self.allow_scratch)
            if checksum(source / name) != expected:
                raise StateUnavailable("checkpoint checksum mismatch")
        profile = RetrievalProfile(**json.loads((source / "profile.json").read_text()))
        policy = QdrantProjectionPolicy(
            **json.loads((source / "projection-policy.json").read_text())
        )
        quality = json.loads((source / "quality-profile.json").read_text())
        capsule = (
            load_capsule(source, quality, profile)
            if data["checkpoint_format"] == 2
            else None
        )
        if (
            profile.hash != data["semantic_profile_hash"]
            or policy.hash != data["projection_policy_hash"]
            or digest(quality) != data["quality_profile_hash"]
            or policy.server_version != data["server_version"]
        ):
            raise StateUnavailable("checkpoint profile mismatch")
        target = self.create(
            data["dataset"],
            run_id,
            dataset_fingerprint=data["dataset_fingerprint"],
            profile=profile,
            quality_profile=quality,
            deployment=deployment,
            policy=policy,
            initial_status="CLONING",
            quality_capsule=capsule,
        )
        manifest = json.loads((target / "manifest.json").read_text())
        manifest.update(status="CLONING")
        atomic_json(target / "manifest.json", manifest)
        shutil.copyfile(source / "canonical.sqlite", target / "sqlite/canonical.sqlite")
        with (
            closing(sqlite3.connect(target / "sqlite/canonical.sqlite")) as connection,
            connection,
        ):
            source_uuid, new_uuid = fork_database_identity(connection)
            if source_uuid != data["db_uuid"]:
                raise StateUnavailable("checkpoint canonical identity differs")
        shutil.copyfile(
            source / "qdrant.snapshot", target / "qdrant/snapshots/restore.snapshot"
        )
        for relative in (
            "sqlite/canonical.sqlite",
            "qdrant/snapshots/restore.snapshot",
        ):
            fsync_file(target / relative)
        atomic_json(target / "artifacts/restore-checkpoint.json", data)
        manifest.update(
            status="RESTORING",
            canonical_db_uuid=new_uuid,
            lineage={
                "source_storage_id": data["source_storage_id"],
                "source_db_uuid": source_uuid,
                "checkpoint_sha256": checksum(source / "checkpoint.json"),
            },
        )
        atomic_json(target / "manifest.json", manifest)
        return target

    def from_canonical(self, source_sqlite, dataset, run_id, **kwargs):
        """Explicit fresh projection. SQLite online backup, never vector export.

        The backup is a consistent SQLite snapshot even if an unrelated source
        process exists. If that snapshot contains an unpublished mutation it is
        refused. A caller should use source run checkpoint admission when live.
        """
        source = checked_path(source_sqlite, allow_scratch=self.allow_scratch)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.create(
            dataset, run_id, initial_status="COPYING_CANONICAL", **kwargs
        )
        manifest = json.loads((target / "manifest.json").read_text())
        manifest.update(status="COPYING_CANONICAL")
        atomic_json(target / "manifest.json", manifest)
        destination = target / "sqlite/canonical.sqlite"
        with (
            closing(
                sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
            ) as original,
            closing(sqlite3.connect(destination)) as copy,
            copy,
        ):
            original.backup(copy)
            if copy.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise StateUnavailable("imported canonical integrity failed")
            if copy.execute("PRAGMA foreign_key_check").fetchall():
                raise StateUnavailable("imported canonical relationships invalid")
            if copy.execute(
                "SELECT count(*) FROM scopes s JOIN lexical_corpus_stats l USING(scope) "
                "WHERE s.canonical_seq<>s.published_seq OR s.canonical_seq<>l.lexical_seq"
            ).fetchone()[0]:
                raise StateUnavailable("canonical import has an unpublished frontier")
            hashes = copy.execute("SELECT DISTINCT profile_hash FROM scopes").fetchall()
            if any(h[0] != kwargs["profile"].hash for h in hashes):
                raise StateUnavailable("canonical import profile differs")
            source_uuid, new_uuid = fork_database_identity(copy)
        fsync_file(destination)
        manifest.update(
            status="REPROJECTING",
            canonical_db_uuid=new_uuid,
            lineage={
                "canonical_snapshot_sha256": checksum(destination),
                "source_db_uuid": source_uuid,
                "source_kind": "SQLite online backup; no vector bytes",
            },
        )
        atomic_json(target / "manifest.json", manifest)
        return target

    def retire(self, dataset, run_id, *, storage_id):
        """Safe cleanup is archival rename, never recursive deletion.

        Requires a stopped service and explicit manifest identity. Unknown paths,
        active owners and live receipts are refused; bytes remain recoverable.
        """
        root = self.path(dataset, run_id)
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("storage_id") != storage_id or manifest.get("run_id") != run_id:
            raise StateUnavailable("retire identity mismatch")
        with (root / "run.owner").open("a+b") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from mandol_substrate.qdrant.service import process_identity

            receipt = root / "qdrant/service.json"
            if receipt.exists():
                record = json.loads(receipt.read_text())
                if record["storage_id"] != storage_id:
                    raise StateUnavailable("retire service identity mismatch")
                if record["mode"] == "native":
                    live = process_identity(record["pid"]) == record["identity"]
                else:
                    info = subprocess.run(
                        ["docker", "inspect", record["container_id"]],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    # Unknown Docker failure is not proof a container is stopped.
                    if info.returncode and "No such object" not in info.stderr:
                        raise StateUnavailable("cannot verify stopped Docker service")
                    live = (
                        info.returncode == 0
                        and json.loads(info.stdout)[0]["State"]["Running"]
                    )
                if live:
                    raise StateUnavailable("stop owned service before retiring run")
            destination = checked_path(
                self.root / ".retired" / dataset / (run_id + "--" + storage_id),
                allow_scratch=self.allow_scratch,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            os.rename(root, destination)
            fsync_directory(root.parent)
            fsync_directory(destination.parent)
            return destination

    def from_canonicals(self, sources, dataset, run_id, *, resume=False, **kwargs):
        """Converge disjoint per-history SQLite files into one dataset/run DB."""
        from .canonical_import import import_disjoint_canonical_sources

        target = self.path(dataset, run_id)
        if resume:
            manifest = json.loads((target / "manifest.json").read_text())
            if (
                manifest["status"] != "COPYING_CANONICAL"
                or manifest["semantic_profile_hash"] != kwargs["profile"].hash
                or manifest["quality_profile_hash"] != digest(kwargs["quality_profile"])
                or manifest["dataset_fingerprint"] != kwargs["dataset_fingerprint"]
            ):
                raise StateUnavailable("canonical import resume contract differs")
        else:
            target = self.create(
                dataset, run_id, initial_status="COPYING_CANONICAL", **kwargs
            )
            manifest = json.loads((target / "manifest.json").read_text())
            manifest.update(status="COPYING_CANONICAL")
            atomic_json(target / "manifest.json", manifest)
        with (target / "run.owner").open("a+b") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            destination = target / "sqlite/canonical.sqlite"
            profile = kwargs["profile"]
            processor = MandolTextProcessor(
                mode=profile.options.get("text_mode", "regex"),
                use_jieba=profile.options.get("jieba", False),
            )
            # The ordinary canonical schema/manifest is the only schema creator.
            state = SQLiteStateStore(destination, profile, processor=processor)
            state.close()
            receipts = import_disjoint_canonical_sources(
                destination,
                sources,
                profile.hash,
                scratch=target / "artifacts/import-staging",
                allow_scratch=self.allow_scratch,
            )
            atomic_json(
                target / "artifacts/canonical-import.json",
                {
                    "receipts": receipts,
                    "destination_sha256": checksum(destination),
                    "source_kind": "disjoint SQLite snapshots",
                },
            )
            manifest.update(
                status="REPROJECTING",
                lineage={"import_receipts": "artifacts/canonical-import.json"},
            )
            atomic_json(target / "manifest.json", manifest)
        return target
