"""Pinned Querit-4B official class and explicit Stage A pair/mask contract.

The published snapshot has no auto_map and names a stale architecture. Loading
its base AutoModel would omit the ranking head. Only the hashed official
QueritModel class is loaded here, after verifying every checkpoint file.
"""

import gc
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from importlib.metadata import version

from .reference_contract import (
    bounded_batches,
    file_sha256,
    finite_raw_scores,
    pinned_snapshot,
    validate_settings,
    verify_files,
)

MODEL_NAME = "Querit/Querit-4B"
MODEL_REVISION = "e557bb5ce6555f16b33dbb7d2ca62e8d27c14c13"
MODEL_MAX_TOKENS = 40960
TOKENIZER_DECLARED_MAX_TOKENS = 131072
OFFICIAL_CLASS_SHA256 = (
    "3a6b98dab1aa5505c1e17508618e67e3f628cd1e33d123f2802f8f5647f37e3f"
)
MTEB_REVISION = "ed47a25455157433b458db2625e72960468597ab"
PROMPT = "Judge whether the Content meets the requirements based on the Query. Query: {query}; Content: {document}"
FILE_MANIFEST = {
    "config.json": {
        "sha256": "4fd7167e58d6adbf806ddd06894e65efdee4d3dfa7532bd897f0ca68ac84fb4c"
    },
    "modeling_querit_4b.py": {"sha256": OFFICIAL_CLASS_SHA256},
    "tokenizer_config.json": {
        "sha256": "fcd3ad3c12549e7b9bba58f94c5312c3e80b252592e6fbddf52d34bbed4ede0c"
    },
    "special_tokens_map.json": {
        "sha256": "97d72d503040b49a5455c3b3b8662bca887d7b23b7c5ff88193656dea794e132"
    },
    "tokenizer.json": {
        "sha256": "3d5994f990fc4d9d85f0683ce8c79984ef43cf18e4d5047d9ac2ee72210222b9",
        "bytes": 11423129,
    },
    "chat_template.jinja": {
        "sha256": "44d5f08f3f72b837eaad09f13a54c1f9f4eb58d75240334548b7fd52a5437fa5"
    },
    "generation_config.json": {
        "sha256": "bb52bfdd308deaea4ec800bf0165e75770b0a4e5c105963bee1b0398f4043d3e"
    },
    "model.safetensors.index.json": {
        "sha256": "a2add79bc7d4ed33f397302c63f79b3b4a51d4603261de7debf292f7d0cd7b52"
    },
    "model-00001-of-00002.safetensors": {
        "sha256": "5b2b13727c7138ba8b75e87a9c38f321f1fab710633f19bee2b48cece6d06bbf",
        "bytes": 4965832928,
    },
    "model-00002-of-00002.safetensors": {
        "sha256": "79aa6357725b61757d902afac3ff52e79b9193b2f5db8dd7a9ce3ba312469694",
        "bytes": 3077776988,
    },
}


def expected_profile(
    *,
    dtype="bfloat16",
    device="cuda:0",
    max_pair_tokens=MODEL_MAX_TOKENS,
    batch_size=8,
    max_batch_tokens=MODEL_MAX_TOKENS,
):
    validate_settings(
        dtype, device, max_pair_tokens, MODEL_MAX_TOKENS, batch_size, max_batch_tokens
    )
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "backend": "transformers-official-querit-class",
        "reference_source": f"https://huggingface.co/{MODEL_NAME}/blob/{MODEL_REVISION}/modeling_querit_4b.py",
        "official_class_sha256": OFFICIAL_CLASS_SHA256,
        "pair_reference_source": f"https://github.com/embeddings-benchmark/mteb/blob/{MTEB_REVISION}/mteb/models/model_implementations/querit_models.py",
        "transformers_version": "4.57.3",
        "torch_version": "2.8.0",
        "dtype": dtype,
        "device": device,
        "arithmetic_policy": f"{dtype}-backend-default-v1",
        "attention_implementation": "sdpa",
        "use_cache": False,
        "model_max_tokens": MODEL_MAX_TOKENS,
        "tokenizer_declared_max_tokens": TOKENIZER_DECLARED_MAX_TOKENS,
        "max_pair_tokens": max_pair_tokens,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "padding_side": "left",
        "padding_policy": "dynamic-batch-maximum-v1",
        "truncation_policy": "reject-over-complete-pair-budget-v1",
        "pair_format": "official-mteb-default-prompt-final-cls-v1",
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "instruction": None,
        "attention_mask_policy": "official-content-causal-final-cls-4d-bool-v1",
        "score_semantics": "official-head-softmax-in-model-dtype-p1-minus-p0-v1",
        "output_kind": "raw_score",
        "source_file_manifest_sha256": hashlib.sha256(
            json.dumps(FILE_MANIFEST, sort_keys=True).encode()
        ).hexdigest(),
    }


@dataclass(frozen=True)
class PreparedQueritPair:
    query: str
    document: str
    input_ids: tuple[int, ...]

    @property
    def token_count(self):
        return len(self.input_ids)


class QueritPreparation:
    def __init__(self, tokenizer, max_pair_tokens=MODEL_MAX_TOKENS):
        validate_settings(
            "bfloat16", "cpu", max_pair_tokens, MODEL_MAX_TOKENS, 1, MODEL_MAX_TOKENS
        )
        if (
            tokenizer.convert_tokens_to_ids("[CLS]") != 151665
            or tokenizer.pad_token_id != 151643
            or tokenizer.model_max_length != TOKENIZER_DECLARED_MAX_TOKENS
        ):
            raise ValueError(
                "Querit tokenizer differs from pinned CLS/padding/context contract"
            )
        self.tokenizer, self.max_pair_tokens = tokenizer, max_pair_tokens

    def prepare(self, query, document):
        if any(not isinstance(value, str) for value in (query, document)):
            raise ValueError("Querit query and document must be strings")
        text = PROMPT.format(query=query, document=document)
        ids = self.tokenizer.encode(
            text, add_special_tokens=False, truncation=False
        ) + [151665]
        if len(ids) > self.max_pair_tokens:
            raise ValueError(
                "Querit complete pair exceeds configured context; text was not deleted"
            )
        return PreparedQueritPair(query, document, tuple(ids))


def official_attention_mask(block_types):
    """Literal official MTEB block-type rule, including all-masked pad rows."""
    import torch

    position_types = torch.tensor(block_types, dtype=torch.long)
    n = position_types.numel()
    row_types, col_types = position_types.view(n, 1), position_types.view(1, n)
    row_idx, col_idx = torch.arange(n).view(n, 1), torch.arange(n).view(1, n)
    causal = col_idx <= row_idx
    content = (row_types == 1) & (col_types == 1) & causal
    cls = (row_types == 2) & (col_types != 0) & causal
    return (content | cls).unsqueeze(0)


def collate_pairs(pairs, pad_token_id=151643):
    """Dynamic left padding changes allocation; every complete pair is retained."""
    import torch

    longest = max(pair.token_count for pair in pairs)
    rows, masks = [], []
    for pair in pairs:
        padding = longest - pair.token_count
        rows.append([pad_token_id] * padding + list(pair.input_ids))
        masks.append(
            official_attention_mask([0] * padding + [1] * (pair.token_count - 1) + [2])
        )
    return {
        "input_ids": torch.tensor(rows, dtype=torch.long),
        "attention_mask": torch.stack(masks),
    }


def _load_official_model_class(snapshot):
    """Execute only the exact reviewed official file after integrity checks."""
    path = snapshot / "modeling_querit_4b.py"
    if file_sha256(path) != OFFICIAL_CLASS_SHA256:
        raise ValueError("Querit official class source digest differs")
    spec = importlib.util.spec_from_file_location(
        f"mandol_querit_{MODEL_REVISION}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError("Querit official class cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QueritModel


class NativeQueritReference:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_pair_tokens=MODEL_MAX_TOKENS,
        batch_size=8,
        max_batch_tokens=MODEL_MAX_TOKENS,
    ):
        if (
            model.config.model_type != "qwen3"
            or model.config.max_position_embeddings != MODEL_MAX_TOKENS
            or model.config.hidden_size != 2560
            or model.config.vocab_size != 151666
            or tuple(model.head.weight.shape) != (2, 2560)
            or tuple(model.head.bias.shape) != (2,)
            or model.config.use_cache is not False
            or model.config._attn_implementation != "sdpa"
        ):
            raise ValueError(
                "Querit actual model/head/config differs from pinned reference"
            )
        self.model, self.tokenizer = model.eval(), tokenizer
        self.preparation = QueritPreparation(tokenizer, max_pair_tokens)
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens
        self._profile = expected_profile(
            dtype=str(model.dtype).removeprefix("torch."),
            device=str(model.device),
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        self.last_batch_metrics = {}

    @property
    def profile(self):
        if self.model is None:
            raise RuntimeError("Querit reference is closed")
        return {
            **self._profile,
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "device": str(self.model.device),
            "max_pair_tokens": self.preparation.max_pair_tokens,
            "model_max_tokens": self.model.config.max_position_embeddings,
            "batch_size": self.batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "use_cache": self.model.config.use_cache,
            "attention_implementation": self.model.config._attn_implementation,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path=MODEL_NAME,
        *,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        cache_dir=None,
        dtype="bfloat16",
        device="cuda:0",
        max_pair_tokens=MODEL_MAX_TOKENS,
        batch_size=8,
        max_batch_tokens=MODEL_MAX_TOKENS,
    ):
        if revision != MODEL_REVISION or tokenizer_revision != revision:
            raise ValueError(
                "Querit reference requires its immutable qualified revision"
            )
        expected_profile(
            dtype=dtype,
            device=str(device),
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        for package, expected in (("transformers", "4.57.3"), ("torch", "2.8.0")):
            if version(package).split("+")[0] != expected:
                raise ValueError(f"Querit reference requires {package} {expected}")
        snapshot = pinned_snapshot(model_name_or_path, MODEL_NAME, revision, cache_dir)
        verify_files(snapshot, FILE_MANIFEST)
        import torch
        from transformers import AutoConfig, AutoTokenizer

        config = AutoConfig.from_pretrained(
            str(snapshot),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        QueritPreparation(tokenizer, max_pair_tokens)
        model_class = _load_official_model_class(snapshot)
        model, loading_info = model_class.from_pretrained(
            str(snapshot),
            config=config,
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        if any(
            loading_info.get(key)
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
        ):
            raise ValueError(
                f"Querit checkpoint did not load every official tensor: {loading_info}"
            )
        model.to(device)
        return cls(
            model,
            tokenizer,
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )

    def score(self, query, documents):
        import torch

        if (
            self.profile != self._profile
            or self.tokenizer.pad_token_id != 151643
            or self.tokenizer.convert_tokens_to_ids("[CLS]") != 151665
        ):
            raise ValueError("Querit runtime settings changed after qualification")
        if (
            not isinstance(query, str)
            or not isinstance(documents, (list, tuple))
            or len(documents) > 10000
        ):
            raise ValueError("Querit requires a string query and bounded document list")
        self.last_batch_metrics = {}
        pairs = [self.preparation.prepare(query, document) for document in documents]
        batches = bounded_batches(
            [pair.token_count for pair in pairs], self.batch_size, self.max_batch_tokens
        )
        scores, padded_tokens, mask_elements = [], 0, 0
        with torch.inference_mode():
            for indices in batches:
                features = collate_pairs([pairs[i] for i in indices])
                padded_tokens += features["input_ids"].numel()
                mask_elements += features["attention_mask"].numel()
                output = self.model(
                    **{
                        key: value.to(self.model.device)
                        for key, value in features.items()
                    }
                )
                if not isinstance(output, dict) or "score" not in output:
                    raise ValueError("Querit official model returned no ranking score")
                values = output["score"]
                if not isinstance(values, torch.Tensor) or values.shape != (
                    len(indices),
                    1,
                ):
                    raise ValueError(
                        "Querit official model returned invalid score shape"
                    )
                values = finite_raw_scores(values[:, 0].float().tolist(), len(indices))
                if any(not -1 <= value <= 1 for value in values):
                    raise ValueError(
                        "Querit p1-minus-p0 score is outside its mathematical range"
                    )
                scores.extend(values)
        self.last_batch_metrics = {
            "batch_sizes": [len(batch) for batch in batches],
            "pair_tokens": [pair.token_count for pair in pairs],
            "actual_tokens": sum(pair.token_count for pair in pairs),
            "padded_tokens": padded_tokens,
            "attention_mask_elements": mask_elements,
            "truncated_pairs": 0,
            "output_kind": "raw_score",
            "profile": self.profile,
        }
        return scores

    def close(self, *, release_cuda_cache=False):
        self.model = self.tokenizer = self.preparation = None
        gc.collect()
        if release_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
