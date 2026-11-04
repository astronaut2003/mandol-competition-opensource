"""Optional serving identity, distinct from frozen retrieval semantics."""
import hashlib
import json
import os
import tomllib
from pathlib import Path


def bindings():
    config = os.getenv('MANDOL_CONFIG')
    runtime = tomllib.loads(Path(config).read_text()).get('runtime', {}) if config else {}
    mode = runtime.get('mode')
    if mode in ('local', 'cluster'):
        from mandol_competition.config.topology import validate_bindings
        path = os.getenv('MANDOL_' + mode.upper() + '_INFERENCE_BINDINGS')
        if path:
            value = json.loads(Path(path).read_text())
            if value.get('format') != 'mandol-serving-bindings-v1':
                raise ValueError('unsupported serving identity manifest')
            return validate_bindings(value['models'])
        return validate_bindings(runtime.get('bindings'))
    path = os.getenv('MANDOL_INFERENCE_BINDINGS')
    if not path:
        return {}
    value = json.loads(Path(path).read_text())
    if value.get('format') != 'mandol-serving-bindings-v1':
        raise ValueError('unsupported serving identity manifest')
    return value['models']


def claim_serving_identity(root):
    """Never reuse vectors across local quantization and production precision."""
    models = bindings()
    marker = Path(root) / 'serving-identity.json'
    if not models:
        if marker.exists():
            raise ValueError('bound runtime requires its inference identity manifest')
        return
    semantic = {name: {k: v for k, v in spec.items() if k not in ('endpoint', 'root')}
                for name, spec in models.items()}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    expected = {'format': 'mandol-serving-identity-v1', 'sha256': digest}
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text()) != expected:
            raise ValueError('runtime belongs to a different inference precision/profile')
    else:
        if any(Path(root).rglob('*.sqlite*')):
            raise ValueError('cannot implicitly adopt an existing unbound database')
        with marker.open('x') as stream:
            json.dump(expected, stream)
            stream.flush()
            os.fsync(stream.fileno())
