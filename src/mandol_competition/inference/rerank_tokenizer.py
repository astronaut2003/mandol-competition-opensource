"""Local budget tokenizer, distinct from v4; cloud revision is unattested."""
import hashlib
from importlib.resources import files
import threading
from .errors import InferenceSchemaError

SHA256='5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
REVISION='2fc06364715b967f1860aea9cf38778875588b17'
_instance=None
_lock=threading.Lock()

def encode_text(text):
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                from tokenizers import Tokenizer
                data=(files(__package__)/'tokenizers/qwen35_rerank/tokenizer.json').read_bytes()
                if hashlib.sha256(data).hexdigest()!=SHA256:
                    raise InferenceSchemaError('rerank_tokenizer_integrity')
                _instance=Tokenizer.from_buffer(data)
    return _instance.encode(text,add_special_tokens=False)

def descriptor():
    return dict(model='Qwen/Qwen3.5-0.8B',revision=REVISION,asset_sha256=SHA256,
        provider_revision_attested=False,count_semantics='compatibility-budget-estimate',
        runtime_network=False,model_weights=False)
