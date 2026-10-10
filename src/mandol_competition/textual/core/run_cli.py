"""Administrative dataset/run lifecycle; never an alternative semantic writer.

Use ``python -m mandol_competition.textual.core.run_cli --help``. Standalone administration
requires an exclusive run owner; an active API can call its run.checkpoint()
directly. No command deletes a run recursively or changes the default backend.
"""

import argparse
import json
import os
import signal
import threading
from pathlib import Path

from mandol_substrate.state.models import RetrievalProfile, Scope, StateUnavailable
from mandol_substrate.vector.verification import verify_projection
from .projection_repair import repair_current_projections
from .run_quality import capture_capsule
from .run_storage import CompetitionRun, RunStorageManager, checksum


class NoInference:
    """Maintenance may verify existing vectors, but cannot fabricate missing ones."""

    def encode_documents(self, _):
        raise StateUnavailable(
            "missing projection requires explicit repair with qualified encoders"
        )


def deployment(args):
    if args.mode == "docker":
        if not args.image_id or not args.image_id.startswith("sha256:"):
            raise ValueError("immutable Docker image ID required")
        return {"mode": "docker", "image_id": args.image_id}
    if not args.binary:
        raise ValueError("pinned native binary required")
    return {"mode": "native", "binary_sha256": checksum(args.binary)}


def real_encoders(run, args):
    contract = getattr(args, "inference_contract", None) or os.getenv("AML_INFERENCE_CONTRACT", "reference-v1")
    if contract == "vpc-vllm-029-v1":
        from .remote_inference import remote_encoders

        encoders = remote_encoders(run.profile, args.embedding_url,
                                   getattr(args, "splade_url", None) or os.getenv("SPLADE_ENDPOINT"))
        return encoders, encoders
    if contract != "reference-v1":
        raise ValueError("Unknown inference contract")
    from transformers import AutoTokenizer

    from .models import RetrievalEncoders, ServingProfile, VllmModelClient
    from .splade import MandolSpladeEncoder

    if not args.embedding_url:
        raise ValueError("explicit qualified embedding endpoint required")
    profile = run.profile
    dense = VllmModelClient(
        args.embedding_url, ServingProfile(**profile.options["embedding_serving"])
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            profile.embedding_model,
            revision=profile.embedding_revision,
            cache_dir=args.model_cache,
            local_files_only=True,
        )
        dense.validate_profile(tokenizer)
        sparse = MandolSpladeEncoder(
            profile.splade_model,
            profile.splade_revision,
            cache_dir=args.model_cache,
            device="cpu",
        )
        return RetrievalEncoders(dense, sparse), dense
    except BaseException:
        dense.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "clone", "retire", "import-canonical", "package-quality"):
        sub = commands.add_parser(name)
        sub.add_argument("--runs-root", type=Path, required=True)
        sub.add_argument("--run-id", required=True)
        if name != "clone":
            sub.add_argument("--dataset", required=True)
        if name in {"create", "clone", "import-canonical"}:
            sub.add_argument("--mode", choices=("native", "docker"), default="native")
            sub.add_argument("--binary", type=Path)
            sub.add_argument("--image-id")
        if name in {"create", "import-canonical"}:
            sub.add_argument("--dataset-hash", required=True)
            sub.add_argument("--profile", type=Path, required=True)
            sub.add_argument("--quality-profile", type=Path, required=True)
            sub.add_argument("--quality-capsule", type=Path)
            if name == "import-canonical":
                sub.add_argument(
                    "--source-sqlite", type=Path, action="append", required=True
                )
        elif name == "clone":
            sub.add_argument("--checkpoint", type=Path, required=True)
        elif name == "retire":
            sub.add_argument("--storage-id", required=True)
    for name in (
        "verify",
        "checkpoint",
        "serve-storage",
        "reproject",
        "repair-current",
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        sub.add_argument("--binary", type=Path)
        if name == "checkpoint":
            sub.add_argument("--name", required=True)
        if name in {"reproject", "repair-current"}:
            sub.add_argument("--embedding-url", required=True)
            sub.add_argument("--splade-url")
            sub.add_argument("--inference-contract")
            sub.add_argument("--model-cache", type=Path)
        if name == "repair-current":
            sub.add_argument("--namespace", required=True)
            sub.add_argument("--user-id", required=True)
            sub.add_argument("--uid", action="append", required=True)
    for name in ("serve", "preflight", "stop"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-root", type=Path, required=True)
        if name != "stop":
            sub.add_argument("--binary", type=Path)
            sub.add_argument("--namespace")
            sub.add_argument("--embedding-url")
            sub.add_argument("--reranker-url")
            sub.add_argument("--splade-url")
            sub.add_argument("--inference-contract")
            sub.add_argument("--model-cache")
            sub.add_argument("--host")
            sub.add_argument("--port", type=int)
            sub.add_argument("--allow-unauthenticated-dev", action="store_true")
    args = parser.parse_args()
    if args.command in {"serve", "preflight", "stop"}:
        from .deployment import preflight, serve, settings_for, stop

        if args.command == "stop":
            print(json.dumps(stop(args.run_root)))
        elif args.command == "preflight":
            print(json.dumps(preflight(settings_for(args))))
        else:
            serve(settings_for(args))
        return
    if args.command in {
        "create",
        "clone",
        "retire",
        "import-canonical",
        "package-quality",
    }:
        manager = RunStorageManager(args.runs_root)
        if args.command in {"create", "import-canonical"}:
            options = {
                "dataset_fingerprint": args.dataset_hash,
                "profile": RetrievalProfile(**json.loads(args.profile.read_text())),
                "quality_profile": json.loads(args.quality_profile.read_text()),
                "deployment": deployment(args),
            }
            if options["quality_profile"].get("components"):
                options["quality_capsule"] = (
                    json.loads(args.quality_capsule.read_text())
                    if args.quality_capsule
                    else capture_capsule(options["quality_profile"], options["profile"])
                )
            if args.command == "import-canonical":
                root = manager.from_canonicals(
                    args.source_sqlite, args.dataset, args.run_id, **options
                )
            else:
                root = manager.create(args.dataset, args.run_id, **options)
        elif args.command == "clone":
            root = manager.clone(
                args.checkpoint, args.run_id, deployment=deployment(args)
            )
        elif args.command == "package-quality":
            root = manager.package_quality(args.dataset, args.run_id)
        else:
            root = manager.retire(args.dataset, args.run_id, storage_id=args.storage_id)
        print(json.dumps({"root": str(root), "operation": args.command}))
        return
    with CompetitionRun(
        args.run_root,
        encoders=NoInference(),
        binary=args.binary,
        defer_recovery=True,
        reproject=args.command == "reproject",
    ) as run:
        if args.command == "verify":
            report = verify_projection(run.state, run.vector)
            print(
                json.dumps(
                    {
                        "verification": report,
                        "frontiers": run.frontiers(),
                        "MIGRATION_CONSISTENT": "YES" if report["clean"] else "NO",
                        "MODEL_SERVING_QUALIFIED_BY_THIS_COMMAND": False,
                    }
                )
            )
            if not report["clean"]:
                raise SystemExit(2)
        elif args.command in {"repair-current", "reproject"}:
            encoders, dense = real_encoders(run, args)
            try:
                run.coordinator.encoders = encoders
                if args.command == "repair-current":
                    report = repair_current_projections(
                        run, Scope(args.namespace, args.user_id), args.uid
                    )
                else:
                    run.finish_startup()
                    report = verify_projection(run.state, run.vector)
                print(json.dumps(report))
            finally:
                dense.close()
        else:
            run.finish_startup()
            if args.command == "checkpoint":
                print(json.dumps({"checkpoint": str(run.checkpoint(args.name))}))
            else:
                stop = threading.Event()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    signal.signal(sig, lambda *_: stop.set())
                print(
                    json.dumps(
                        {
                            "uri": run.service.uri,
                            "storage_id": run.storage_id,
                            "storage_only": True,
                        }
                    ),
                    flush=True,
                )
                while not stop.wait(10):
                    run.health()


if __name__ == "__main__":
    main()
