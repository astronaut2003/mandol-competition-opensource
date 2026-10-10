"""Fetch and locate pinned AML evaluation sources outside the source distribution.

Downloads are explicit. Importing this module or constructing an evaluation
protocol never contacts the network. Retrieved sources retain upstream terms.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

SOURCE_PATHS = {
    "locomo": "data/locomo-refined/pipeline.py",
    "longmemeval": "data/longmemeval-s/pipeline.py",
}
SOURCE_DIRECTORY_ENV = "MANDOL_AML_PROTOCOL_SOURCE_DIR"


def source_directory(revision):
    """Use an explicit source directory or the user's versioned cache."""
    configured = os.getenv(SOURCE_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    cache = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return (cache / "mandol-competition" / "aml" / revision).expanduser().resolve()


def read_protocol_source(dataset, revision):
    if dataset not in SOURCE_PATHS:
        raise ValueError("Unsupported public QA dataset")
    try:
        return (source_directory(revision) / (dataset + ".py")).read_bytes()
    except FileNotFoundError:
        raise FileNotFoundError(
            "Pinned AML evaluation sources are required for local QA. "
            "Run python3 scripts/fetch_aml_protocol_sources.py, or set "
            "MANDOL_AML_PROTOCOL_SOURCE_DIR to your downloaded source directory. "
            "See docs/Reproduction.md."
        ) from None


def fetch_sources(output=None, *, verify_only=False, checkout=None):
    """Download exact public sources, verify hashes, and keep them outside Git."""
    from .evaluation_protocol import OFFICIAL_REVISION, SOURCE_HASHES

    destination = Path(output).expanduser().resolve() if output else source_directory(OFFICIAL_REVISION)
    if checkout is not None and destination.is_relative_to(Path(checkout).resolve()):
        raise ValueError("Store AML evaluation sources outside the project checkout")
    if not verify_only:
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    verified = []
    for dataset, upstream_path in SOURCE_PATHS.items():
        target = destination / (dataset + ".py")
        temporary = None
        if target.exists():
            raw = target.read_bytes()
        elif verify_only:
            raise FileNotFoundError("Pinned AML source is missing for " + dataset)
        else:
            url = (
                "https://raw.githubusercontent.com/AML-memory/agent-memory-leaderboard/"
                + OFFICIAL_REVISION + "/" + upstream_path
            )
            with urlopen(url, timeout=30) as response:
                raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("AML evaluation source exceeds download size limit")
        if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[dataset]:
            raise ValueError("Pinned AML source hash mismatch for " + dataset)
        if not target.exists():
            try:
                with tempfile.NamedTemporaryFile(dir=destination, prefix="." + dataset, delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(raw)
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        verified.append(dataset)
    return {"revision": OFFICIAL_REVISION, "datasets": verified, "source_directory": str(destination)}


def main(argv=None, *, checkout=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory outside this checkout; defaults to the versioned user cache")
    parser.add_argument("--verify-only", action="store_true", help="Check cached sources without network requests")
    args = parser.parse_args(argv)
    try:
        result = fetch_sources(args.output, verify_only=args.verify_only, checkout=checkout)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
