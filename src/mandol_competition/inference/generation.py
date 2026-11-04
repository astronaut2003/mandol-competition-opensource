"""Explicit generation-only routing; retrieval models never use this policy.

The historical logical mini alias remains a storage/recipe identity. The actual
provider, wire model, credential source, and scoped proxy are operational facts.
"""
from dataclasses import dataclass
import os
from urllib.parse import urlsplit

OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1'
OPENROUTER_MODEL = 'openai/gpt-4o-mini'
CLOSEAI_ENDPOINT = 'https://api.openai-proxy.org/v1'
DEFAULT_PROXY = 'http://127.0.0.1:17897'
MINI_RESPONSE_MODELS = frozenset({'gpt-4o-mini', 'gpt-4o-mini-2024-07-18',
                                'openai/gpt-4o-mini', 'openai/gpt-4o-mini-2024-07-18'})
MINI_REQUEST_MODELS = MINI_RESPONSE_MODELS | {'gpt-4o-mini-closeai'}


def is_mini_response(model):
    return isinstance(model, str) and model in MINI_RESPONSE_MODELS


def is_openrouter_endpoint(endpoint):
    try:
        parsed = urlsplit(endpoint or '')
        return (parsed.scheme == 'https' and parsed.hostname == 'openrouter.ai'
                and parsed.port in (None, 443) and parsed.path.rstrip('/') == '/api/v1'
                and not any((parsed.username, parsed.password, parsed.query, parsed.fragment)))
    except (TypeError, ValueError):
        return False


def validate_generation_proxy(endpoint, proxy):
    if not is_openrouter_endpoint(endpoint):
        raise ValueError('generation proxy is restricted to the OpenRouter origin')
    try:
        parsed = urlsplit(proxy)
        valid = (parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', '::1', 'localhost'}
                 and parsed.port is not None and 1 <= parsed.port <= 65535
                 and parsed.path in ('', '/')
                 and not any((parsed.username, parsed.password, parsed.query, parsed.fragment)))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('credential-free loopback HTTP generation proxy required')
    return proxy.rstrip('/')


@dataclass(frozen=True)
class GenerationRoute:
    endpoint: str
    model: str
    credential_env: str
    proxy: str | None
    provider: str
    environment_override: bool = False

    @property
    def openrouter(self):
        return self.provider == 'openrouter'

    @property
    def closeai(self):
        return self.provider == 'closeai'

    @property
    def routed(self):
        return self.openrouter or self.closeai

    def key(self, fallback=None):
        from mandol_competition.config.unified import secret_from_environment
        if self.closeai:
            # A stale OpenRouter argument/config must never supply this key.
            value = secret_from_environment('CLOSEAI_API_KEY')
        elif self.openrouter:
            # A CloseAI key must never be forwarded to the new destination.
            value = secret_from_environment('OPENROUTER_API_KEY')
            if not value and not self.environment_override:
                value = fallback
        else:
            value = fallback or secret_from_environment(self.credential_env)
        if not value:
            raise ValueError('configured generation credential is required')
        return value

    def options(self, options):
        result = dict(options)
        if not self.routed:
            return result
        maximum = result.get('max_completion_tokens')
        if maximum is not None:
            if 'max_tokens' in result and (type(result['max_tokens']) is not int or result['max_tokens'] != maximum):
                raise ValueError('conflicting generation output budgets')
        budget = maximum if maximum is not None else result.get('max_tokens')
        if type(budget) is not int or not 1 <= budget <= 16384:
            raise ValueError('explicit bounded mini output budget required')
        if any(key in result for key in ('model', 'models', 'route')) or result.get('stream'):
            raise ValueError('fixed-model non-streaming generation required')
        if self.closeai:
            if any(key in result for key in ('provider', 'transforms')):
                raise ValueError('OpenRouter-only routing options cannot reach CloseAI')
            # CloseAI keeps the caller's explicit OpenAI-compatible token field.
            return result
        if maximum is not None:
            result.pop('max_completion_tokens')
            result['max_tokens'] = maximum
        provider = result.get('provider', {})
        if not isinstance(provider, dict):
            raise ValueError('generation provider preferences must be an object')
        result['provider'] = dict(provider, require_parameters=True)
        # No automatic middle-out compression of source evidence.
        result['transforms'] = []
        return result


def generation_route(endpoint, model, *, credential_env='OPENAI_API_KEY'):
    policy = os.getenv('MANDOL_GENERATION_PROVIDER', 'configured')
    if policy not in {'configured', 'openrouter', 'closeai'}:
        raise ValueError('unknown generation provider policy')
    try:
        parsed = urlsplit(endpoint or '')
    except (TypeError, ValueError):
        raise ValueError('invalid generation endpoint') from None
    if policy == 'closeai':
        if model not in MINI_REQUEST_MODELS:
            raise ValueError('CloseAI selection preserves the existing mini model only')
        target = os.getenv('MANDOL_CLOSEAI_BASE_URL', CLOSEAI_ENDPOINT)
        try:
            configured = urlsplit(target)
            valid = (configured.scheme == 'https' and configured.hostname
                and configured.port in (None, 443) and configured.path.rstrip('/') == '/v1'
                and configured.hostname != 'openrouter.ai'
                and not any((configured.username, configured.password, configured.query, configured.fragment)))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('credential-free HTTPS CloseAI endpoint required')
        wire = model.removeprefix('openai/')
        if wire == 'gpt-4o-mini-closeai':
            wire = 'gpt-4o-mini'
        return GenerationRoute(target.rstrip('/'), wire, 'CLOSEAI_API_KEY', None, 'closeai', True)
    selected = policy == 'openrouter' or parsed.hostname == 'openrouter.ai'
    if not selected:
        return GenerationRoute(endpoint, model, credential_env, None, 'openai-proxy')
    if parsed.hostname == 'openrouter.ai' and not is_openrouter_endpoint(endpoint):
        raise ValueError('canonical HTTPS OpenRouter endpoint required')
    if model not in MINI_REQUEST_MODELS:
        raise ValueError('OpenRouter switch preserves the existing mini model only')
    proxy = validate_generation_proxy(OPENROUTER_ENDPOINT,
        os.getenv('MANDOL_OPENROUTER_PROXY', DEFAULT_PROXY))
    wire = ('openai/gpt-4o-mini-2024-07-18' if model.endswith('-2024-07-18') else OPENROUTER_MODEL)
    return GenerationRoute(OPENROUTER_ENDPOINT, wire, 'OPENROUTER_API_KEY', proxy,
                           'openrouter', policy == 'openrouter')
