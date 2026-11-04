"""Explicit deployment topology. Frozen retrieval profiles remain elsewhere."""
from dataclasses import dataclass, field
import ipaddress
import math
import re


FROZEN_MODELS = {
    'querit-4b': dict(model='Querit/Querit-4B', revision='e557bb5ce6555f16b33dbb7d2ca62e8d27c14c13',
                      dtype='bfloat16', vllm_version='0.29.0', served_alias='querit-4b'),
    'BAAI/bge-reranker-v2-m3': dict(model='BAAI/bge-reranker-v2-m3',
                      revision='953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e',
                      dtype='bfloat16', vllm_version='0.25.1', served_alias='BAAI/bge-reranker-v2-m3'),
}


@dataclass(frozen=True)
class RuntimeSettings:
    mode: str = 'legacy'
    host: str = '127.0.0.1'
    ports: tuple[int, ...] = (18080, 18081, 18082)
    startup_timeout: float = 240.
    credential_env: str = 'MANDOL_MEMORY_SYSTEM_KEY'
    vlm_credential_env: str = 'OPENAI_API_KEY'
    qdrant_credential_env: str = 'MANDOL_QDRANT_API_KEY'
    bindings: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def environment(self):
        """Only non-secret operational limits; credentials stay with the caller."""
        mapping = dict(worker_capacity='MANDOL_API_WORKER_CAPACITY', max_queue='MANDOL_API_MAX_QUEUE',
                       queue_timeout='MANDOL_API_QUEUE_TIMEOUT_SECONDS',
                       add_timeout='AML_ADD_TIMEOUT_SECONDS', search_timeout='AML_SEARCH_TIMEOUT_SECONDS')
        result = {mapping[k]: str(v) for k, v in self.limits.items()}
        for operation in ('add', 'search'):
            if operation + '_timeout' in self.limits:
                result['MANDOL_MM_' + operation.upper() + '_TIMEOUT_SECONDS'] = str(self.limits[operation + '_timeout'])
        return result


def validate_bindings(models):
    if not isinstance(models, dict) or set(models) != set(FROZEN_MODELS):
        raise ValueError('explicit topology requires frozen Querit and BGE serving bindings')
    for alias, expected in FROZEN_MODELS.items():
        value = models[alias]
        if (not isinstance(value, dict) or set(value) != set(expected) | {'root'}
                or any(value.get(k) != v for k, v in expected.items())
                or not isinstance(value.get('root'), str) or not value['root'].strip()):
            raise ValueError('serving binding differs from frozen model/revision/dtype/version')
    return models


def runtime_settings(value):
    if value is None:
        return RuntimeSettings()
    if not isinstance(value, dict) or set(value) - {'mode', 'api', 'authentication', 'vlm', 'qdrant', 'bindings', 'limits'}:
        raise ValueError('unknown runtime topology setting')
    mode = value.get('mode')
    if mode not in ('local', 'cluster'):
        raise ValueError('runtime mode must explicitly be local or cluster')
    api = value.get('api', {})
    if set(api) - {'host', 'textual_port', 'code_port', 'multimodal_port', 'startup_timeout'}:
        raise ValueError('unknown API binding setting')
    host = api.get('host', '127.0.0.1')
    address = ipaddress.ip_address(host)
    if mode == 'local' and not address.is_loopback:
        raise ValueError('Local APIs must bind loopback')
    ports = tuple(api.get(t + '_port', p) for t, p in zip(('textual', 'code', 'multimodal'), (18080, 18081, 18082)))
    if len(set(ports)) != 3 or any(type(p) is not int or not 1 <= p <= 65535 for p in ports):
        raise ValueError('three distinct valid API ports required')
    startup = api.get('startup_timeout', 240.)
    if type(startup) not in (int, float) or not math.isfinite(startup) or not 0 < startup <= 1800:
        raise ValueError('bounded startup timeout required')
    sources = []
    for key, default in [('authentication', 'MANDOL_' + mode.upper() + '_API_KEY'),
                         ('vlm', 'MANDOL_' + mode.upper() + '_VLM_KEY'),
                         ('qdrant', 'MANDOL_' + mode.upper() + '_QDRANT_KEY')]:
        source = value.get(key, {})
        if set(source) - {'credential_env'}:
            raise ValueError('only explicit credential environment sources are supported')
        name = source.get('credential_env', default)
        if not isinstance(name, str) or not re.fullmatch('[A-Z][A-Z0-9_]*', name) or name.endswith('_FILE'):
            raise ValueError('credential source must be an environment variable name')
        sources.append(name)
    limits = value.get('limits', {})
    if set(limits) - {'worker_capacity', 'max_queue', 'queue_timeout', 'add_timeout', 'search_timeout'}:
        raise ValueError('unknown API resource limit')
    for key, v in limits.items():
        if key in ('worker_capacity', 'max_queue'):
            valid = type(v) is int and (key == 'max_queue' and v == 0 or 1 <= v <= 1024)
        else:
            valid = type(v) in (int, float) and math.isfinite(v) and 0 < v <= 1800
        if not valid:
            raise ValueError('finite bounded API resource limits required')
    return RuntimeSettings(mode, host, ports, startup, *sources, validate_bindings(value.get('bindings')), limits)
