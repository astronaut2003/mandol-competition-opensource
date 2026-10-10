"""Offline maintenance; the same owner lock excludes a running API process.

Usage: python -m mandol_competition.multimodal.admin verify|reconcile|gc|rebuild|import-legacy
Configuration uses the same MANDOL_MM_* variables as serving.
"""

import argparse
import json

from .backend import MultimodalBackend
from .config import get_config
from .migration import import_legacy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "reconcile", "gc", "rebuild", "import-legacy"))
    parser.add_argument("--source", help="legacy canonical.sqlite3, required for import-legacy")
    parser.add_argument("--media-root", help="legacy data directory containing users/<hash>/media")
    args = parser.parse_args()
    if args.command == "import-legacy" and not args.source:
        parser.error("import-legacy requires --source")
    backend = MultimodalBackend(get_config(), maintenance=True)
    try:
        if args.command == "verify":
            report = backend.verify()
        elif args.command == "import-legacy":
            report = import_legacy(backend, args.source, media_root=args.media_root)
        else:
            report = backend.reconcile(rebuild=args.command == "rebuild", gc=args.command == "gc")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report.get("servable", True) else 1
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
