"""Claim fresh runtime roots; never adopt an old deployment implicitly."""

import json
import os
from pathlib import Path


def claim_runtime_root(root, track):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "unified-runtime.json"
    expected = {"format": "mandol-unified-fresh-v1", "track": track}
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != expected:
            raise ValueError("runtime root belongs to a different deployment or track")
    else:
        if any(root.iterdir()):
            raise ValueError("unowned nonempty runtime root; old runtime data is not migrated")
        with marker.open("x") as stream:
            json.dump(expected, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return root

