from dataclasses import dataclass, field
import os
from pathlib import Path
import tomllib
from mandol_competition.inference.http import endpoint_origin
from .topology import RuntimeSettings, runtime_settings

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
    reranker_pool: dict = field(default_factory=dict)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    quality_candidate: str | None = None
    evidence_profile: str = 'default'
    retrieval_policy: str = 'baseline'
    reranker_backend: str = 'profile'
    reranker_credential_env: str = 'DASHSCOPE_API_KEY'

@dataclass(frozen=True)
class UnifiedSettings:
    tracks: dict[str, TrackSettings]
    log_level: str = 'INFO'
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
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
    if set(document) - {'global', 'runtime', *TRACKS}:
        raise ValueError('unknown unified configuration section')
    def reject_secrets(obj):
        for key, value in obj.items():
            if any(part in key.lower() for part in ('secret', 'api_key', 'auth_token', 'memory_system_key', 'password')):
                raise ValueError('secrets belong in environment variables or secret mounts')
            if isinstance(value, dict): reject_secrets(value)
    reject_secrets(document)
    runtime = runtime_settings(document.get('runtime'))
    # New explicit modes use mode-scoped overrides, never ambient production
    # paths/endpoints/secret mounts. Existing unscoped deployments stay compatible.
    mode_prefix = 'MANDOL_' if runtime.mode == 'legacy' else 'MANDOL_' + runtime.mode.upper() + '_'
    if runtime.mode == 'cluster' and os.getenv(mode_prefix + 'ALLOW_UNAUTHENTICATED_DEV') == '1':
        raise ValueError('Cluster mode cannot disable authentication')
    global_settings = document.get('global', {})
    if set(global_settings) - {'runtime_root', 'qdrant_endpoint', 'inference', 'log_level'}:
        raise ValueError('unknown global setting')
    root = Path(os.getenv(mode_prefix + 'RUNTIME_ROOT', global_settings.get('runtime_root', 'data'))).resolve()
    endpoint = os.getenv(mode_prefix + 'QDRANT_ENDPOINT', global_settings.get('qdrant_endpoint', 'http://127.0.0.1:6333'))
    result = {}
    for name in TRACKS:
        cfg = document.get(name, {})
        if set(cfg) - {'storage_root', 'namespace', 'qdrant', 'inference', 'inference_contract', 'reranker_pool', 'quality_candidate', 'evidence_profile', 'retrieval_policy', 'reranker_backend', 'reranker_credential_env'}:
            raise ValueError('unknown track deployment setting')
        q = cfg.get('qdrant', {})
        if set(q) - {'endpoint', 'collection'}: raise ValueError('unknown Qdrant setting')
        prefix = mode_prefix + name.upper() + '_'
        evidence = os.getenv(prefix + 'EVIDENCE_PROFILE', cfg.get('evidence_profile', 'default'))
        if name != 'textual' and ('evidence_profile' in cfg or prefix + 'EVIDENCE_PROFILE' in os.environ):
            raise ValueError('Versioned evidence profile is supported only for Textual')
        if name == 'textual':
            from mandol_competition.textual.core.evidence_profiles import validate_evidence_profile
            validate_evidence_profile(evidence)
        candidate_env = os.getenv(prefix + 'QUALITY_CANDIDATE')
        candidate = candidate_env if candidate_env is not None else cfg.get('quality_candidate')
        if candidate not in (None, ''):
            if name != 'textual' or not isinstance(candidate, str):
                raise ValueError('Explicit quality-candidate path is supported only for Textual')
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute() and candidate_env is None and path:
                candidate_path = Path(path).resolve().parent / candidate_path
            candidate = str(candidate_path.resolve())
        else:
            candidate = None
        storage = Path(os.getenv(prefix + 'STORAGE_ROOT', cfg.get('storage_root', root / name))).resolve()
        contract = os.getenv(prefix + 'INFERENCE_CONTRACT', cfg.get('inference_contract', 'vllm-http-v1'))
        if contract not in {'vllm-http-v1','aml-official-eligible-v1'}:
            raise ValueError('unknown unified inference contract')
        policy = os.getenv(prefix + 'RETRIEVAL_POLICY', cfg.get('retrieval_policy', 'baseline'))
        from mandol_competition.inference.dashscope_rerank import BACKEND, ENDPOINT, PATH
        # Explicit historical MM policies keep their frozen model. New v4
        # Compositions default to the requested external reranker; explicit
        # historical Code/MM recipes retain their original ranking.
        explicit_endpoint = os.getenv(prefix + 'RERANKER_ENDPOINT',
            cfg.get('inference', {}).get('reranker', global_settings.get('inference', {}).get('reranker')))
        historical_endpoint = (explicit_endpoint is not None and not explicit_endpoint.endswith(PATH))
        default_backend = (BACKEND if contract == 'aml-official-eligible-v1'
                           and not historical_endpoint and not cfg.get('reranker_pool', {}).get('endpoints')
                           and policy not in {'c0-original-v1', 'm0-original-v1', 'mm-querit-same-pool-rank-only-v1'} else 'profile')
        reranker_backend = os.getenv(prefix + 'RERANKER_BACKEND', cfg.get('reranker_backend', default_backend))
        if reranker_backend not in {'profile', BACKEND}:
            raise ValueError('unsupported track reranker backend')
        if name == 'code' and reranker_backend == BACKEND and contract != 'aml-official-eligible-v1':
            raise ValueError('Code Qwen reranker requires the official v4 inference contract')
        credential_env = cfg.get('reranker_credential_env', 'DASHSCOPE_API_KEY')
        import re
        if not isinstance(credential_env, str) or not re.fullmatch('[A-Z][A-Z0-9_]*', credential_env) or credential_env.endswith('_FILE'):
            raise ValueError('reranker credential source must be an environment variable name')
        if name == 'multimodal' and reranker_backend == BACKEND:
            if policy not in {'baseline', 'mm-qwen37-same-pool-rank-only-v1'}:
                raise ValueError('explicit MM retrieval policy and reranker disagree')
            policy = 'mm-qwen37-same-pool-rank-only-v1'
        allowed = {'textual': {'baseline'},
                   'code': {'baseline', 'c0-original-v1', 'code-native-sparse-rrf100-v1'},
                   'multimodal': {'baseline', 'm0-original-v1', 'mm-querit-same-pool-rank-only-v1', 'mm-qwen37-same-pool-rank-only-v1'}}
        if policy not in allowed[name] or (policy not in {'baseline','c0-original-v1','m0-original-v1'}
                                           and contract != 'aml-official-eligible-v1'):
            raise ValueError('unsupported track retrieval policy or model contract')
        if runtime.mode != 'legacy' and contract != 'aml-official-eligible-v1':
            raise ValueError('Local/Cluster profiles must use frozen T0/C0/M0')
        expected_collection = name + ('_aml_v4_v1' if contract=='aml-official-eligible-v1' else '_v1')
        collection = os.getenv(prefix + 'QDRANT_COLLECTION', q.get('collection', expected_collection))
        if collection != expected_collection: raise ValueError('wrong track collection binding')
        recipes = DEFAULT_ENDPOINTS[name] | global_settings.get('inference', {}) | cfg.get('inference', {})
        if reranker_backend == BACKEND:
            recipes['reranker'] = explicit_endpoint or ENDPOINT
            from urllib.parse import urlsplit
            provider_url = urlsplit(endpoint_origin(recipes['reranker']))
            if provider_url.scheme != 'https' or provider_url.path.rstrip('/') != PATH:
                raise ValueError('explicit HTTPS DashScope reranker endpoint required')
        if set(recipes) - {'embedding', 'sparse', 'reranker', 'vlm'}: raise ValueError('unknown inference channel')
        recipes = {channel: endpoint_origin(os.getenv(prefix + channel.upper() + '_ENDPOINT', value)) for channel, value in recipes.items() if value}
        from mandol_competition.inference.replicas import pool_configuration, replica_endpoints
        pool = pool_configuration(cfg.get('reranker_pool'))
        if reranker_backend == BACKEND and name != 'code':
            # Legacy private replica addresses must never receive cloud payloads
            # or provider credentials. Configure a single explicit cloud origin.
            pool['endpoints'] = [recipes['reranker']]
        if name == 'code' and pool:
            raise ValueError('Code reranking uses one explicit endpoint, not a replica pool')
        if name != 'code':
            configured = pool.get('endpoints')
            if runtime.mode != 'legacy' and reranker_backend != BACKEND:
                override = os.getenv(prefix + 'RERANKER_REPLICAS')
                configured = override.split(',') if override else configured or [recipes['reranker']]
            pool['endpoints'] = replica_endpoints(name, recipes['reranker'], configured)
            recipes['reranker'] = pool['endpoints'][0]
        if name == 'multimodal' and policy == 'mm-qwen37-same-pool-rank-only-v1' and reranker_backend != BACKEND:
            raise ValueError('MM Qwen search policy requires its explicit reranker backend')
        if runtime.mode != 'legacy':
            if name != 'code' and recipes['embedding'] != recipes['sparse']:
                raise ValueError('frozen Dense/native Sparse must share their paired provider endpoint')
            if name == 'code' and set(recipes) != ({'embedding', 'reranker'} if reranker_backend == BACKEND else {'embedding'}):
                raise ValueError('Code supports optional Dense/native Sparse and an explicit optional reranker')
            if name == 'multimodal' and 'vlm' not in recipes:
                raise ValueError('explicit VLM endpoint required')
        credential = (secret_from_environment(prefix + 'MEMORY_SYSTEM_KEY') or secret_from_environment('MANDOL_MEMORY_SYSTEM_KEY')
                      if runtime.mode == 'legacy' else secret_from_environment(runtime.credential_env))
        result[name] = TrackSettings(name, storage, os.getenv(prefix + 'NAMESPACE', cfg.get('namespace', name)), collection,
            endpoint_origin(os.getenv(prefix + 'QDRANT_ENDPOINT', q.get('endpoint', endpoint))), recipes,
            credential,
            secret_from_environment(runtime.qdrant_credential_env) or None,
            os.getenv(mode_prefix + 'ALLOW_UNAUTHENTICATED_DEV', '0') == '1', contract, pool, runtime, candidate, evidence, policy,
            reranker_backend, credential_env)
    paths = [t.root for t in result.values()]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError('track runtime roots must be physically isolated and non-nested')
    return UnifiedSettings(result, os.getenv('MANDOL_LOG_LEVEL', global_settings.get('log_level', 'INFO')), runtime)
