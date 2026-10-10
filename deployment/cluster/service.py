"""Read-only Linux process identity checks for safe lifecycle management."""
import hashlib
import os
from pathlib import Path

BOOT = Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def identity(pid):
    try:
        proc = Path('/proc') / str(pid)
        if proc.stat().st_uid != os.getuid():
            return None
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = (proc / 'cmdline').read_bytes()
        if not argv:
            return None  # /proc can expose an empty argv during exec.
        return {'pid': pid, 'start_ticks': fields[19], 'boot_id': BOOT,
                'executable': os.readlink(proc / 'exe'), 'argv_sha256': hashlib.sha256(argv).hexdigest()}
    except (OSError, ValueError):
        return None


def owned(record):
    actual = identity(record['pid'])
    # vLLM may change its process title after startup. Guard the immutable
    # PID/boot/start-time/executable tuple; argv hash is diagnostic provenance.
    return actual is not None and all(actual[k] == record.get(k)
                                     for k in ('pid', 'boot_id', 'start_ticks', 'executable'))

