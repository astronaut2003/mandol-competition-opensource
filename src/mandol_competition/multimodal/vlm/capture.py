"""Temporary bounded wire capture for diagnosing actual visual extraction calls.

Disabled unless a private directory and an absolute expiry are explicitly set.
Headers, credentials, correlation IDs and database identifiers never enter it.
The artifacts contain source images/text and must stay in the private directory.
Capture failures must never change the original inference or publication result.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time


class VisionWireCapture:
    FILE_BYTES = 20 * 1024 * 1024
    SLOTS = {'output_failure': 2, 'success': 1, 'transport_failure': 1}

    def __init__(self, directory, until):
        self.directory, self.until = Path(directory), until

    @classmethod
    def from_environment(cls):
        directory = os.getenv('MANDOL_MM_VISION_CAPTURE_DIR')
        expiry = os.getenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX')
        if not directory or not expiry:
            return None
        try:
            until = float(expiry)
            if (not Path(directory).is_absolute() or not math.isfinite(until)
                    or not time.time() < until <= time.time() + 1800):
                return None
            return cls(directory, until)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def classify(payload, response, error):
        if error is not None:
            return 'transport_failure'
        choices = response.get('choices') if isinstance(response, dict) else None
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return 'output_failure'
        choice = choices[0]
        if choice.get('finish_reason') != 'stop':
            return 'output_failure'
        usage = response.get('usage')
        completion = usage.get('completion_tokens') if isinstance(usage, dict) else None
        budget = payload.get('max_completion_tokens', payload.get('max_tokens'))
        if type(completion) is int and type(budget) is int and completion >= budget:
            return 'output_failure'
        message = choice.get('message')
        if not isinstance(message, dict):
            return 'output_failure'
        content = message.get('content')
        try:
            from .openai_compat import _extract_json_object
            _extract_json_object(content)
        except Exception:
            return 'output_failure'
        return 'success'

    def observe(self, payload, response, error=None):
        """Observe the exact HTTP JSON payload and response, without another call."""
        directory_fd = None
        try:
            if time.time() >= self.until:
                return
            kind = self.classify(payload, response, error)
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(directory_fd)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o077:
                return
            # A fixed set of exclusive filenames bounds samples across all
            # threads, client instances and process restarts, not just one client.
            slots = [f'{kind}-{index}.json' for index in range(1, self.SLOTS[kind] + 1)]
            if all((self.directory / name).exists() for name in slots):
                return
            request = {key: payload[key] for key in ('model', 'messages', 'temperature',
                'max_tokens', 'max_completion_tokens', 'response_format', 'stream', 'stream_options')
                if key in payload}
            # The callback never receives HTTP headers. The response whitelist
            # also excludes provider headers and their echoed diagnostics.
            reply = ({key: response[key] for key in ('model', 'choices', 'usage') if key in response}
                     if isinstance(response, dict) else None)
            request_bytes = json.dumps(request, ensure_ascii=False, allow_nan=False,
                separators=(',', ':')).encode('utf-8')
            artifact = dict(schema='mandol-private-vision-wire-capture-v1',
                captured_at_utc=datetime.now(timezone.utc).isoformat(), kind=kind,
                request_sha256=hashlib.sha256(request_bytes).hexdigest(),
                request=request, response=reply,
                error_type=type(error).__name__ if error is not None else None)
            data = json.dumps(artifact, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if len(data) > self.FILE_BYTES:
                return
            for name in slots:
                try:
                    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600, dir_fd=directory_fd)
                except FileExistsError:
                    continue
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(data)
                return
        except Exception:
            # Diagnostics cannot turn a successful inference into a failed Add,
            # or hide the existing error/cost/unknown-outcome classification.
            return
        finally:
            if directory_fd is not None:
                try:
                    os.close(directory_fd)
                except OSError:
                    pass
