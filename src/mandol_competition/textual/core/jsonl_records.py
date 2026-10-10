"""Physical LF-delimited JSON records, preserving Unicode within JSON strings."""

import json
from pathlib import Path


def iter_jsonl_records(path):
    # splitlines() also splits U+0085/U+2028/U+2029, which are legal unescaped
    # characters within a JSON string. A torn/invalid record still raises.
    with Path(path).open(encoding="utf-8", newline="\n") as stream:
        for line in stream:
            yield json.loads(line)
