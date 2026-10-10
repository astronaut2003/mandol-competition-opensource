"""Thin deployment entrypoints over the qualified run and AML lifecycle."""

import fcntl
import json
import logging
import os
import signal
import time
from dataclasses import replace
from pathlib import Path

from mandol_substrate.qdrant.service import process_identity, signal_owned_process
from mandol_substrate.state.models import StateUnavailable

from .run_storage import atomic_json, checksum


def settings_for(args):
    from mandol_competition.textual.api.config import Settings

    defaults = Settings.from_env(bind_run=False)
    settings = replace(defaults,
        competition_run_root=str(args.run_root),
        qdrant_binary=str(args.binary) if args.binary else os.getenv("QDRANT_BINARY"),
        evaluation_namespace=args.namespace
        or os.getenv("AML_EVALUATION_NAMESPACE", ""),
        embedding_url=args.embedding_url
        or os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:18101"),
        reranker_url=args.reranker_url
        or os.getenv("RERANK_ENDPOINT", "http://127.0.0.1:18106"),
        inference_contract=getattr(args, "inference_contract", None) or defaults.inference_contract,
        splade_url=getattr(args, "splade_url", None) or defaults.splade_url,
        model_cache=args.model_cache or os.getenv("AML_MODEL_CACHE"),
        memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY", ""),
        allow_unauthenticated_dev=args.allow_unauthenticated_dev or defaults.allow_unauthenticated_dev,
        host=args.host or defaults.host,
        port=args.port if args.port is not None else defaults.port,
    )
    return settings.bind_run()


def preflight(settings):
    from mandol_substrate.vector.verification import verify_projection
    from mandol_competition.textual.api.competition import CompetitionMemoryService

    service = CompetitionMemoryService(settings)
    began = time.monotonic()
    try:
        service.start()
        service.health()
        verification = verify_projection(service.state, service.projection)
        if not verification["clean"]:
            raise StateUnavailable("Preflight projection verification failed")
        root = Path(settings.competition_run_root)
        return {
            "status": "PASS",
            "elapsed_seconds": time.monotonic() - began,
            "backend": "qdrant",
            "quality_profile_sha256": checksum(root / "quality-profile.json"),
            "profile_hash": service.profile.hash,
            "projection_policy_hash": service.projection.policy.hash,
            "frontiers": service.run.frontiers(),
            "verification": verification,
            "models": (service.inference_attestation()
                       if settings.inference_contract == "vpc-vllm-029-v1"
                       else "REAL_ENDPOINTS_AND_PINNED_SPLADE_VERIFIED"),
            "formal_submission_ready": False,
        }
    finally:
        service.shutdown()


def serve(settings):
    import torch
    import uvicorn

    from mandol_competition.textual.api.app import create_app

    torch.set_num_threads(4)
    root = Path(settings.competition_run_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    # Separate launcher ownership prevents another API from replacing its receipt.
    with (root / "api.owner").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StateUnavailable("This run already has an API owner") from exc
        receipt = {
            "pid": os.getpid(),
            "identity": process_identity(os.getpid()),
            "storage_id": manifest["storage_id"],
            "host": settings.host,
            "port": settings.port,
            "status": "STARTING",
        }
        atomic_json(root / "api-process.json", receipt)
        logging.basicConfig(level=settings.log_level)
        try:
            uvicorn.run(
                create_app(settings),
                host=settings.host,
                port=settings.port,
                log_level=settings.log_level.lower(),
                access_log=False,
            )
        finally:
            atomic_json(root / "api-process.json", dict(receipt, status="STOPPED"))


def stop(root, timeout=180):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    receipt = json.loads((root / "api-process.json").read_text())
    if receipt["storage_id"] != manifest["storage_id"]:
        raise StateUnavailable("API receipt belongs to another run")
    if process_identity(receipt["pid"]) != receipt["identity"]:
        return {"status": "ALREADY_STOPPED", "foreign_process_signalled": False}
    if not _holds_api_lock(root, receipt["pid"]):
        raise StateUnavailable("Recorded process does not own this run's API lock")
    signal_owned_process(receipt["pid"], receipt["identity"], signal.SIGTERM)
    until = time.monotonic() + timeout
    while process_identity(receipt["pid"]) == receipt["identity"]:
        if time.monotonic() >= until:
            raise StateUnavailable(
                "API still draining; no broad kill or data deletion performed"
            )
        time.sleep(0.1)
    return {"status": "STOPPED", "foreign_process_signalled": False}


def _holds_api_lock(root, pid):
    """A copied receipt cannot authorize signalling an unrelated live process."""
    expected = (root / "api.owner").stat()
    try:
        for fd in Path(f"/proc/{pid}/fd").iterdir():
            try:
                actual = fd.stat()
                if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
                    continue
                info = Path(f"/proc/{pid}/fdinfo/{fd.name}").read_text()
                if any(
                    "FLOCK" in line and "WRITE" in line and f" {pid} " in line
                    for line in info.splitlines() if line.startswith("lock:")
                ):
                    return True
            except FileNotFoundError:
                continue
    except FileNotFoundError:
        return False
    return False
