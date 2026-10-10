"""Pinned modular Sentence Transformers Mixedbread reference for Stage A.

This is the current model card's CrossEncoder/LogitScore path. It deliberately
does not use the older SDK's separately encoded fields or normalization.
"""

import gc
import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version

from .reference_contract import (
    bounded_batches,
    finite_raw_scores,
    pinned_snapshot,
    validate_settings,
    verify_files,
)

MODEL_NAME = "mixedbread-ai/mxbai-rerank-large-v2"
MODEL_REVISION = "ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6"
MODEL_MAX_TOKENS = 32768
TOKENIZER_DECLARED_MAX_TOKENS = 131072
TEMPLATE_SHA256 = "6922504e8481a822f50ecdf80da9d41f09cb01601c82d1acae04bdba654deafc"
REFERENCE_SOURCE = (
    "https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2/blob/"
    + MODEL_REVISION
    + "/modules.json"
)
FILE_MANIFEST = {
    "config.json": {
        "sha256": "6e928710075fa30df8469596004a2ef5b3be810cab5e2e7315a979e63d5e10ba"
    },
    "modules.json": {
        "sha256": "8b0cf4a17b264e13b267907ee1026e551b1cde537691dd71af4c8aa55a889769"
    },
    "sentence_bert_config.json": {
        "sha256": "8c98fdfffa489b1b78843150aeb7b2992800e0dc8138079ddc4d72a98fd40348"
    },
    "config_sentence_transformers.json": {
        "sha256": "b2787b4c078bc8d55f1a4a904b2e7b52dadfd9b26424517645bb130d9aa3f9b0"
    },
    "1_LogitScore/config.json": {
        "sha256": "2554429e789efe4bb7b78977ed56bb03dd9321d63b1e304c04f94ba61c0281d3"
    },
    "chat_template.jinja": {"sha256": TEMPLATE_SHA256},
    "tokenizer.json": {
        "sha256": "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"
    },
    "tokenizer_config.json": {
        "sha256": "0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614"
    },
    "special_tokens_map.json": {
        "sha256": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"
    },
    "model.safetensors": {
        "sha256": "430e47b1754d2306fce32e560d3386a6a79850a91e5adf78d51fba9647022730",
        "bytes": 3087466808,
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
        "backend": "sentence-transformers-native",
        "reference_source": REFERENCE_SOURCE,
        "sentence_transformers_version": "5.4.1",
        "transformers_version": "4.57.3",
        "torch_version": "2.8.0",
        "dtype": dtype,
        "configured_dtype": "bfloat16",
        "checkpoint_tensor_dtype": "float16",
        "checkpoint_dtype_conversion": f"float16-to-{dtype}",
        "device": device,
        "arithmetic_policy": f"{dtype}-backend-default-v1",
        "attention_implementation": "sdpa",
        "model_max_tokens": MODEL_MAX_TOKENS,
        "tokenizer_declared_max_tokens": TOKENIZER_DECLARED_MAX_TOKENS,
        "max_pair_tokens": max_pair_tokens,
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "padding_side": "left",
        "truncation_policy": "reject-over-complete-pair-budget-v1",
        "pair_format": "official-st-whole-rendered-jinja-v1",
        "template_sha256": TEMPLATE_SHA256,
        "instruction": None,
        "score_semantics": "raw-final-logit-token16-minus-token15-v1",
        "output_kind": "raw_score",
        "activation": "identity",
        "source_file_manifest_sha256": hashlib.sha256(
            json.dumps(FILE_MANIFEST, sort_keys=True).encode()
        ).hexdigest(),
    }


@dataclass(frozen=True)
class PreparedMixedbreadPair:
    query: str
    document: str
    input_ids: tuple[int, ...]

    @property
    def token_count(self):
        return len(self.input_ids)


class MixedbreadPreparation:
    def __init__(self, tokenizer, max_pair_tokens=MODEL_MAX_TOKENS):
        validate_settings(
            "float32", "cpu", max_pair_tokens, MODEL_MAX_TOKENS, 1, MODEL_MAX_TOKENS
        )
        template = tokenizer.chat_template
        if (
            not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest() != TEMPLATE_SHA256
        ):
            raise ValueError("Mixedbread tokenizer has the wrong official template")
        for text, token_id in (("0", 15), ("1", 16)):
            if tokenizer.encode(text, add_special_tokens=False) != [token_id]:
                raise ValueError("Mixedbread answer token differs from pinned snapshot")
        if tokenizer.pad_token_id != 151643:
            raise ValueError("Mixedbread tokenizer has the wrong padding token")
        self.tokenizer, self.max_pair_tokens = tokenizer, max_pair_tokens

    def prepare(self, query, document):
        if any(not isinstance(value, str) for value in (query, document)):
            raise ValueError("Mixedbread query and document must be strings")
        ids = self.tokenizer.apply_chat_template(
            [
                {"role": "query", "content": query},
                {"role": "document", "content": document},
            ],
            tokenize=True,
            truncation=False,
        )
        if len(ids) > self.max_pair_tokens:
            raise ValueError(
                "Mixedbread complete pair exceeds configured context; text was not deleted"
            )
        return PreparedMixedbreadPair(query, document, tuple(ids))


class NativeMixedbreadReference:
    """Own one pinned CrossEncoder; return a complete list of raw native scores."""

    def __init__(
        self,
        cross_encoder,
        *,
        max_pair_tokens=MODEL_MAX_TOKENS,
        batch_size=8,
        max_batch_tokens=MODEL_MAX_TOKENS,
    ):
        import torch
        from sentence_transformers.cross_encoder.modules.logit_score import LogitScore

        if len(cross_encoder) != 2 or not isinstance(cross_encoder[1], LogitScore):
            raise ValueError(
                "Mixedbread requires the official Transformer/LogitScore modules"
            )
        if (
            cross_encoder[1].true_token_id != 16
            or cross_encoder[1].false_token_id != 15
        ):
            raise ValueError("Mixedbread classifier answer IDs differ")
        if not isinstance(cross_encoder.activation_fn, torch.nn.Identity):
            raise TypeError("Mixedbread native output must retain identity activation")
        self.cross_encoder = cross_encoder.eval()
        self.preparation = MixedbreadPreparation(
            cross_encoder.tokenizer, max_pair_tokens
        )
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens
        self._profile = expected_profile(
            dtype=str(cross_encoder.model.dtype).removeprefix("torch."),
            device=str(cross_encoder.model.device),
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        if (
            cross_encoder.tokenizer.padding_side != "left"
            or cross_encoder.max_seq_length != max_pair_tokens
        ):
            raise ValueError(
                "Mixedbread actual padding/context differs from declared profile"
            )
        self.last_batch_metrics = {}

    @property
    def profile(self):
        if self.cross_encoder is None:
            raise RuntimeError("Mixedbread reference is closed")
        return {
            **self._profile,
            "dtype": str(self.cross_encoder.model.dtype).removeprefix("torch."),
            "device": str(self.cross_encoder.model.device),
            "max_pair_tokens": self.cross_encoder.max_seq_length,
            "batch_size": self.batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "padding_side": self.cross_encoder.tokenizer.padding_side,
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
                "Mixedbread reference requires its immutable qualified revision"
            )
        expected_profile(
            dtype=dtype,
            device=str(device),
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )
        for package, expected in (
            ("sentence-transformers", "5.4.1"),
            ("transformers", "4.57.3"),
            ("torch", "2.8.0"),
        ):
            if version(package).split("+")[0] != expected:
                raise ValueError(f"Mixedbread reference requires {package} {expected}")
        snapshot = pinned_snapshot(model_name_or_path, MODEL_NAME, revision, cache_dir)
        verify_files(snapshot, FILE_MANIFEST)
        import torch
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(
            str(snapshot),
            revision=revision,
            local_files_only=True,
            trust_remote_code=False,
            device=str(device),
            max_length=max_pair_tokens,
            model_kwargs={
                "dtype": getattr(torch, dtype),
                "attn_implementation": "sdpa",
            },
        )
        return cls(
            encoder,
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
        )

    def score(self, query, documents):
        import torch

        if (
            self.profile != self._profile
            or self.preparation.max_pair_tokens != self._profile["max_pair_tokens"]
            or not isinstance(self.cross_encoder.activation_fn, torch.nn.Identity)
            or self.cross_encoder[1].true_token_id != 16
            or self.cross_encoder[1].false_token_id != 15
            or hashlib.sha256(
                self.preparation.tokenizer.chat_template.encode()
            ).hexdigest()
            != TEMPLATE_SHA256
        ):
            raise ValueError("Mixedbread runtime settings changed after qualification")
        if (
            not isinstance(query, str)
            or not isinstance(documents, (list, tuple))
            or len(documents) > 10000
        ):
            raise ValueError("Mixedbread requires a bounded document list")
        self.last_batch_metrics = {}
        pairs = [self.preparation.prepare(query, document) for document in documents]
        batches = bounded_batches(
            [pair.token_count for pair in pairs], self.batch_size, self.max_batch_tokens
        )
        scores, actual_tokens, padded_tokens = [], 0, 0
        with torch.inference_mode():
            for indices in batches:
                batch = [(pairs[i].query, pairs[i].document) for i in indices]
                features = self.cross_encoder.preprocess(batch)
                ids, mask = features["input_ids"], features["attention_mask"]
                if (
                    ids.ndim != 2
                    or mask.shape != ids.shape
                    or ids.shape[0] != len(indices)
                ):
                    raise ValueError(
                        "Mixedbread preprocessing returned invalid batch shape"
                    )
                if any(
                    tuple(ids[j][mask[j].bool()].tolist()) != pairs[i].input_ids
                    for j, i in enumerate(indices)
                ):
                    raise ValueError(
                        "Mixedbread official preprocessing changed complete pair tokens"
                    )
                values = self.cross_encoder.predict(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                )
                if values.ndim == 2 and values.shape[1] == 1:
                    values = values[:, 0]
                if values.ndim != 1:
                    raise ValueError(
                        "Mixedbread predictor returned invalid raw-score shape"
                    )
                scores.extend(finite_raw_scores(values.float().tolist(), len(indices)))
                actual_tokens += int(mask.sum().item())
                padded_tokens += ids.numel()
        self.last_batch_metrics = {
            "batch_sizes": [len(batch) for batch in batches],
            "pair_tokens": [pair.token_count for pair in pairs],
            "actual_tokens": actual_tokens,
            "padded_tokens": padded_tokens,
            "truncated_pairs": 0,
            "output_kind": "raw_score",
            "profile": self.profile,
        }
        return scores

    def close(self, *, release_cuda_cache=False):
        self.cross_encoder = self.preparation = None
        gc.collect()
        if release_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
