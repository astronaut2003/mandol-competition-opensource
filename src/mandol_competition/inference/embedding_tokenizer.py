"""Pinned local Qwen3-Embedding tokenizer used to budget v4 query text.

The provider does not attest a tokenizer revision. Its usage was verified on
owned compatibility text to equal this tokenizer's count plus one framing token.
No model weights, chat template, network access or content cache is used here.
"""
import hashlib
from importlib.resources import files
import threading

from .errors import InferenceSchemaError, InferenceUnavailable

MODEL = 'Qwen/Qwen3-Embedding-0.6B'
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
ASSET_SHA256 = 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a'
_instance = None
_load_lock = threading.Lock()


def _load():
    global _instance
    if _instance is None:
        with _load_lock:
            if _instance is None:
                try:
                    from tokenizers import Tokenizer
                    data = (files(__package__) / 'tokenizers/qwen3_embedding/tokenizer.json').read_bytes()
                except (ImportError, OSError):
                    raise InferenceUnavailable('embedding_tokenizer_unavailable') from None
                if hashlib.sha256(data).hexdigest() != ASSET_SHA256:
                    raise InferenceSchemaError('embedding_tokenizer_integrity')
                try:
                    # from_buffer only reads the bundled asset. Never download
                    # from_pretrained or apply a model/chat prompt at runtime.
                    instance = Tokenizer.from_buffer(data)
                except Exception:
                    raise InferenceSchemaError('embedding_tokenizer_invalid') from None
                _instance = instance
    return _instance


def encode_text(text):
    return _load().encode(text, add_special_tokens=False)


def count_tokens(text):
    return len(encode_text(text))


def descriptor():
    return dict(model=MODEL, revision=REVISION, asset_sha256=ASSET_SHA256,
        implementation='tokenizers-0.22.2', add_special_tokens=False,
        provider_v4_revision_attested=False, validation='owned-text-usage-equals-local-plus-one',
        runtime_network=False, model_weights=False)
