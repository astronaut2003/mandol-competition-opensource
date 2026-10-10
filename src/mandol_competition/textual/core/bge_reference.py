"""Immutable BGE v2-m3 native scorer for qualification, never an API fallback.

The pinned official model card uses AutoModelForSequenceClassification on
query/document pairs. It emits one raw relevance logit; sigmoid is an optional
monotonic mapping, not calibration against Qwen or another reranker.
"""

import gc
from dataclasses import dataclass
from pathlib import Path

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_LICENSE = "Apache-2.0"
OFFICIAL_CONTEXT_TOKENS = 8192
TRUNCATION_POLICY = "document-prefix-character-boundary-pair-reencode-v1"
SCORE_SEMANTICS = "sigmoid(single-relevance-logit)-v1"
PAIR_FORMAT = "xlm-roberta-query-document-pair-v1"


def expected_profile(
    *,
    dtype="float32",
    device="cuda:0",
    max_pair_tokens=4096,
    batch_size=8,
    max_batch_tokens=8192,
    sort_by="none",
    compile_model=False,
):
    """Declare an independent expected cache identity before running the model."""
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("unsupported BGE reference dtype")
    if type(max_pair_tokens) is not int or not 4 <= max_pair_tokens <= 8192:
        raise ValueError("invalid BGE pair token budget")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("BGE reference batch size must be in [1, 128]")
    if type(max_batch_tokens) is not int or max_batch_tokens < max_pair_tokens:
        raise ValueError("BGE batch token budget must fit one maximum pair")
    if sort_by not in {"none", "char", "token"}:
        raise ValueError("unknown BGE batching order")
    if type(compile_model) is not bool:
        raise ValueError("compile_model must be boolean")
    if not isinstance(device, str) or not device:
        raise ValueError("BGE device must be explicit")
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "backend": "transformers-native",
        "transformers_version": "4.57.3",
        "dtype": dtype,
        "device": device,
        "arithmetic_policy": f"{dtype}-backend-default-v1",
        "attention_implementation": "sdpa",
        "model_max_tokens": OFFICIAL_CONTEXT_TOKENS,
        "max_pair_tokens": max_pair_tokens,
        "pair_format": PAIR_FORMAT,
        "instruction": None,
        "padding_side": "right",
        "truncation_policy": TRUNCATION_POLICY,
        "score_semantics": SCORE_SEMANTICS,
        "score_comparison": "within-model-ordering-only",
        "batch_size": batch_size,
        "max_batch_tokens": max_batch_tokens,
        "sort_by": sort_by,
        "compile_model": compile_model,
    }


@dataclass(frozen=True)
class PreparedBgePair:
    query: str
    document: str
    input_ids: tuple[int, ...]
    document_was_truncated: bool
    original_document_chars: int

    @property
    def token_count(self):
        return len(self.input_ids)


class BgeRerankerPreparation:
    """CPU-only exact HF pair tokenization with document-only truncation.

    Four delimiter tokens are reserved before document truncation. Use the
    model card's batched pair API: it retains all pair delimiters for an empty
    document, unlike the tokenizer's single-string ``text_pair=''`` API.
    """

    def __init__(self, tokenizer, max_pair_tokens=4096):
        expected_profile(max_pair_tokens=max_pair_tokens)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("BGE preparation requires a fast offset tokenizer")
        if (
            tokenizer.bos_token_id != 0
            or tokenizer.pad_token_id != 1
            or tokenizer.eos_token_id != 2
            or tokenizer.model_max_length != OFFICIAL_CONTEXT_TOKENS
            or tokenizer.vocab_size != 250002
            or tokenizer.num_special_tokens_to_add(pair=True) != 4
        ):
            raise ValueError("BGE tokenizer differs from the pinned pair contract")
        self.tokenizer = tokenizer
        self.max_pair_tokens = max_pair_tokens

    def _encode_pair(self, query, document):
        return self.tokenizer(
            [[query, document]],
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )["input_ids"][0]

    def prepare(self, query, document, instruction=None):
        if any(not isinstance(value, str) for value in (query, document)):
            raise ValueError("BGE query and document must be strings")
        if instruction not in (None, ""):
            raise ValueError("BGE v2-m3 has no instruction prompt")
        query_ids = self.tokenizer.encode(query, add_special_tokens=False)
        document_budget = self.max_pair_tokens - len(query_ids) - 4
        if document_budget < 0:
            raise ValueError("full BGE query and pair delimiters exceed token budget")
        original_chars = len(document)
        input_ids = self._encode_pair(query, document)
        if len(input_ids) > self.max_pair_tokens:
            encoded = self.tokenizer(
                document,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_offsets_mapping=True,
            )
            cut = (
                encoded["offset_mapping"][document_budget - 1][1]
                if document_budget
                else 0
            )
            cut = min(cut, original_chars)
            while True:
                shortened = document[:cut]
                input_ids = self._encode_pair(query, shortened)
                if len(input_ids) <= self.max_pair_tokens:
                    document = shortened
                    break
                if not cut:
                    raise ValueError("BGE fixed query exceeds token budget")
                cut -= 1
        return PreparedBgePair(
            query,
            document,
            tuple(input_ids),
            len(document) < original_chars,
            original_chars,
        )


def official_reference_score(logits, *, expected_count=None):
    """Sigmoid of the single model-card relevance logit, evaluated in FP32."""
    import torch

    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("expected BGE logits with shape [batch, 1]")
    if expected_count is not None and logits.shape[0] != expected_count:
        raise ValueError("BGE classifier returned the wrong score count")
    scores = logits[:, 0].float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("non-finite BGE relevance logits")
    return torch.sigmoid(scores).tolist()


class NativeBgeReference:
    """Explicitly owned BGE model with bounded padded-token inference batches."""

    def __init__(
        self,
        tokenizer,
        model,
        *,
        max_pair_tokens=4096,
        batch_size=8,
        max_batch_tokens=8192,
        compile_model=False,
    ):
        # Model injection is for CPU contract tests. from_pretrained enforces the
        # pinned model/config/runtime before loading the real model's weights.
        self._profile = expected_profile(
            dtype=str(model.dtype).removeprefix("torch."),
            device=str(model.device),
            max_pair_tokens=max_pair_tokens,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            compile_model=compile_model,
        )
        self.preparation = BgeRerankerPreparation(tokenizer, max_pair_tokens)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.model = model.eval()
        self.batch_size = batch_size
        self.max_batch_tokens = max_batch_tokens
        self.last_batch_metrics = {}
        self.last_raw_logits = []
        if compile_model:
            import torch

            self.model = torch.compile(self.model, dynamic=True)

    @property
    def profile(self):
        if self.model is None:
            raise RuntimeError("BGE reference is closed")
        return {
            **self._profile,
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "device": str(self.model.device),
            "max_pair_tokens": self.preparation.max_pair_tokens,
            "batch_size": self.batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "padding_side": self.tokenizer.padding_side,
        }

    def profile_for(self, *, sort_by="none"):
        if sort_by not in {"none", "char", "token"}:
            raise ValueError("unknown BGE batching order")
        return {**self.profile, "sort_by": sort_by}

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path=MODEL_NAME,
        *,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        device="cuda:0",
        dtype="float32",
        local_files_only=True,
        **kwargs,
    ):
        if revision != MODEL_REVISION or tokenizer_revision != revision:
            raise ValueError("BGE reference requires its qualified immutable revision")
        if str(model_name_or_path) != MODEL_NAME:
            path = Path(model_name_or_path)
            if (
                path.name != revision
                or path.parent.name != "snapshots"
                or path.parent.parent.name != "models--BAAI--bge-reranker-v2-m3"
            ):
                raise ValueError("BGE path must identify the pinned model snapshot")
        expected_profile(dtype=dtype, device=str(device), **kwargs)
        import torch
        import transformers

        if transformers.__version__ != "4.57.3":
            raise ValueError("BGE reference requires frozen transformers 4.57.3")
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        if (
            config.architectures != ["XLMRobertaForSequenceClassification"]
            or config.model_type != "xlm-roberta"
            or config.num_labels != 1
            or config.max_position_embeddings != 8194
            or config.pad_token_id != 1
        ):
            raise ValueError("BGE checkpoint does not have the qualified classifier")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=tokenizer_revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            padding_side="right",
        )
        BgeRerankerPreparation(tokenizer, kwargs.get("max_pair_tokens", 4096))
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            revision=revision,
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
            local_files_only=local_files_only,
            trust_remote_code=False,
        ).to(device)
        if str(model.dtype).removeprefix("torch.") != dtype:
            raise ValueError("BGE model did not load the requested dtype")
        return cls(tokenizer, model, **kwargs)

    def score(self, query, documents, instruction=None, *, sort_by="none"):
        if self.model is None:
            raise RuntimeError("BGE reference is closed")
        if not isinstance(documents, (list, tuple)) or len(documents) > 10000:
            raise ValueError("BGE reference requires a bounded document list")
        if not isinstance(query, str) or instruction not in (None, ""):
            raise ValueError("BGE query must be text without an instruction prompt")
        pairs = [self.preparation.prepare(query, doc, instruction) for doc in documents]
        return self.score_prepared(pairs, sort_by=sort_by)

    def score_prepared(self, pairs, *, sort_by="none"):
        import torch

        if self.model is None:
            raise RuntimeError("BGE reference is closed")
        if self.profile != self._profile:
            raise ValueError("BGE runtime settings changed after profile qualification")
        if not isinstance(pairs, (list, tuple)) or len(pairs) > 10000:
            raise ValueError("BGE reference requires a bounded prepared-pair list")
        self.profile_for(sort_by=sort_by)
        # Validate the full request before inference; a forged prepared pair
        # must not silently substitute token IDs or remove the full query.
        for pair in pairs:
            if (
                not isinstance(pair, PreparedBgePair)
                or pair.token_count > self.preparation.max_pair_tokens
                or self.preparation.prepare(pair.query, pair.document).input_ids
                != pair.input_ids
            ):
                raise ValueError("invalid prepared BGE pair")
        order = list(range(len(pairs)))
        if sort_by != "none":
            order.sort(
                key=lambda i: (
                    pairs[i].token_count
                    if sort_by == "token"
                    else pairs[i].original_document_chars
                ),
                reverse=True,
            )
        batches, current, current_max = [], [], 0
        for index in order:
            next_max = max(current_max, pairs[index].token_count)
            if current and (
                len(current) >= self.batch_size
                or next_max * (len(current) + 1) > self.max_batch_tokens
            ):
                batches.append(current)
                current, current_max = [], 0
            current.append(index)
            current_max = max(current_max, pairs[index].token_count)
        if current:
            batches.append(current)
        scores, raw_logits = [None] * len(pairs), [None] * len(pairs)
        actual_tokens = padded_tokens = 0
        self.last_batch_metrics, self.last_raw_logits = {}, []
        with torch.inference_mode():
            for indices in batches:
                inputs = self.tokenizer.pad(
                    {"input_ids": [list(pairs[i].input_ids) for i in indices]},
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.model.device) for key, value in inputs.items()
                }
                output = self.model(**inputs, return_dict=True)
                batch_scores = official_reference_score(
                    output.logits, expected_count=len(indices)
                )
                batch_logits = output.logits[:, 0].float().tolist()
                actual_tokens += int(inputs["attention_mask"].sum().item())
                padded_tokens += inputs["input_ids"].numel()
                for index, value, logit in zip(
                    indices, batch_scores, batch_logits, strict=True
                ):
                    scores[index], raw_logits[index] = value, logit
        self.last_raw_logits = raw_logits
        self.last_batch_metrics = {
            "batch_sizes": [len(batch) for batch in batches],
            "actual_tokens": actual_tokens,
            "padded_tokens": padded_tokens,
            "padding_ratio": 1 - actual_tokens / padded_tokens if padded_tokens else 0,
            "sort_by": sort_by,
            "profile": self.profile_for(sort_by=sort_by),
        }
        return scores

    def close(self, *, release_cuda_cache=False):
        self.model = self.tokenizer = self.preparation = None
        gc.collect()
        if release_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
