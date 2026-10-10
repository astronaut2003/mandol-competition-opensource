"""Backend-neutral resource/HTTP soak helpers; no projection comparator runtime."""

import json
import os
import subprocess
from pathlib import Path

import psutil


def resources(pid=None, db_path=None):
    p = psutil.Process(pid or os.getpid())
    result = {
        "rss_bytes": p.memory_info().rss,
        "threads": p.num_threads(),
        "fds": p.num_fds(),
    }
    if db_path:
        path = Path(db_path)
        result.update(
            {
                "db_bytes": path.stat().st_size if path.exists() else 0,
                "wal_bytes": Path(str(path) + "-wal").stat().st_size
                if Path(str(path) + "-wal").exists()
                else 0,
            }
        )
    return result


def container_resources():
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return [
        json.loads(line)
        for line in result.stdout.splitlines()
        if "mandol-qualification-final-" in line
    ]


def soak_verdict(elapsed, searches, failures, required_searches):
    """Elapsed time alone cannot qualify an unknown Full Evaluation workload."""
    if failures:
        return "FAIL"
    if elapsed < 24 * 3600:
        return "NOT_COMPLETED"
    if required_searches is None or searches < required_searches:
        return "NEEDS_VALIDATION"
    return "PASS"
