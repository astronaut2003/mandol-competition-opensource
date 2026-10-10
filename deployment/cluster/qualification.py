"""Timestamp and atomic report writers for local deployment measurements."""
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone


def utc():
    return datetime.now(timezone.utc).isoformat()

def atomic_text(path, content):
    """Keep readers and concurrent evidence writers isolated during replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + '\n')
