from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from mandol_competition.inference.http import endpoint_origin

TRACKS = ('textual', 'code', 'multimodal')
DEFAULT_ENDPOINTS = {
 'textual': {'embedding': 'http://127.0.0.1:18101', 'sparse': 'http://127.0.0.1:18102', 'reranker': 'http://127.0.0.1:18106'},
 'code': {'embedding': 'http://127.0.0.1:18101'},
 'multimodal': {'embedding': 'http://127.0.0.1:18201', 'sparse': 'http://127.0.0.1:18201', 'reranker': 'http://127.0.0.1:18201'},
}

@dataclass(frozen=True)
class TrackSettings:
    track: str
    root: Path
    namespace: str
    collection: str
    qdrant_endpoint: str
    inference: dict[str, str]
    memory_system_key: str = field(repr=False)
    qdrant_api_key: str | None = field(repr=False)
    allow_unauthenticated_dev: bool
    inference_contract: str = 'vllm-http-v1'

@dataclass(frozen=True)
class UnifiedSettings:
    tracks: dict[str, TrackSettings]
    log_level: str = 'INFO'
    def track(self, name):
        return self.tracks[name]


def secret_from_environment(name):
    direct, mount = os.getenv(name), os.getenv(name + '_FILE')
    if direct and mount:
        raise ValueError('configure a secret environment value or mount, not both')
    return Path(mount).read_text().rstrip('\r\n') if mount else direct or ''


def load_config(path=None):
    path = path or os.getenv('MANDOL_CONFIG')
    document = tomllib.loads(Path(path).read_text()) if path else {}
    if set(document) - {'global', *TRACKS}:
        raise ValueError('unknown unified configuration section')
    def reject_secrets(obj):
        for key, value in obj.items():
            if any(part in key.lower() for part in ('secret', 'api_key', 'auth_token', 'memory_system_key', 'password')):
                raise ValueError('secrets belong in environment variables or secret mounts')
            if isinstance(value, dict): reject_secrets(value)
    reject_secrets(document)
    global_settings = document.get('global', {})
    if set(global_settings) - {'runtime_root', 'qdrant_endpoint', 'inference', 'log_level'}:
        raise ValueError('unknown global setting')
    root = Path(os.getenv('MANDOL_RUNTIME_ROOT', global_settings.get('runtime_root', 'data'))).resolve()
    endpoint = os.getenv('MANDOL_QDRANT_ENDPOINT', global_settings.get('qdrant_endpoint', 'http://127.0.0.1:6333'))
    result = {}
    for name in TRACKS:
        cfg = document.get(name, {})
        if set(cfg) - {'storage_root', 'namespace', 'qdrant', 'inference', 'inference_contract'}:
            raise ValueError('unknown track deployment setting')
        q = cfg.get('qdrant', {})
        if set(q) - {'endpoint', 'collection'}: raise ValueError('unknown Qdrant setting')
        prefix = 'MANDOL_' + name.upper() + '_'
        storage = Path(os.getenv(prefix + 'STORAGE_ROOT', cfg.get('storage_root', root / name))).resolve()
        contract = os.getenv(prefix + 'INFERENCE_CONTRACT', cfg.get('inference_contract', 'vllm-http-v1'))
        if contract not in {'vllm-http-v1','aml-official-eligible-v1'}:
            raise ValueError('unknown unified inference contract')
        expected_collection = name + ('_aml_v4_v1' if contract=='aml-official-eligible-v1' else '_v1')
        collection = os.getenv(prefix + 'QDRANT_COLLECTION', q.get('collection', expected_collection))
        if collection != expected_collection: raise ValueError('wrong track collection binding')
        recipes = DEFAULT_ENDPOINTS[name] | global_settings.get('inference', {}) | cfg.get('inference', {})
        if set(recipes) - {'embedding', 'sparse', 'reranker', 'vlm'}: raise ValueError('unknown inference channel')
        recipes = {channel: endpoint_origin(os.getenv(prefix + channel.upper() + '_ENDPOINT', value)) for channel, value in recipes.items() if value}
        result[name] = TrackSettings(name, storage, os.getenv(prefix + 'NAMESPACE', cfg.get('namespace', name)), collection,
            endpoint_origin(os.getenv(prefix + 'QDRANT_ENDPOINT', q.get('endpoint', endpoint))), recipes,
            secret_from_environment(prefix + 'MEMORY_SYSTEM_KEY') or secret_from_environment('MANDOL_MEMORY_SYSTEM_KEY'),
            secret_from_environment('MANDOL_QDRANT_API_KEY') or None,
            os.getenv('MANDOL_ALLOW_UNAUTHENTICATED_DEV', '0') == '1', contract)
    paths = [t.root for t in result.values()]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError('track runtime roots must be physically isolated and non-nested')
    return UnifiedSettings(result, os.getenv('MANDOL_LOG_LEVEL', global_settings.get('log_level', 'INFO')))
