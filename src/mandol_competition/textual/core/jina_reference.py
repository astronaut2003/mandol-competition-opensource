"""Pinned Jina v3.5 listwise reference, with complete native input auditing.

The unmodified official ``JinaForRanking.rerank`` computes the scores. This
adapter observes its forward inputs and restores its results to caller order;
it never substitutes pairwise scoring, smaller document blocks, or a sigmoid.
"""

import copy
import gc
import hashlib
import importlib.metadata
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

MODEL_NAME = "jinaai/jina-reranker-v3.5"
MODEL_REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
SOURCE_SHA256 = "ec6612461b4307eb3bab089e6916c00881b7e6b2e8edfbd56a9ace4560244837"
CONTEXT_TOKENS = 131072
QUERY_TOKENS = 1024
DOCUMENT_TOKENS = 8192
BLOCK_DOCUMENTS = 125
DOC_MARKER = "<|embed_token|>"
QUERY_MARKER = "<|rerank_token|>"
DOC_TOKEN_ID = 151670
QUERY_TOKEN_ID = 151671
SCORE_SEMANTICS = "native-listwise-weighted-block-query-cosine-v1"
LAYER_TYPES = (
    "sliding_attention",
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "full_attention",
)
SNAPSHOT_SHA256 = {
    "modeling.py": SOURCE_SHA256,
    "config.json": "6750b018f53b2dc66a9a584ce0c1993542166ce39f9a996f71e0d031ea9c426f",
    "tokenizer.json": "4e95945ab0cef486709f760b81efcc7a6e75747f9165d13ead29159737455803",
    "tokenizer_config.json": "88bbd231e995ec6cadadf7115691703b0376a64e9dae124fd59aa1508ad02ee3",
    "special_tokens_map.json": "9c61cd9afb8a31aea657bb51fd6e53d758a1ebb288f284b0b7420cad3e8417e7",
    "added_tokens.json": "7e57ca349de401cc35df7a7f04f6202f444ffaeb9ee83a5c1709f87bc99bf01b",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "model.safetensors": "50c684b703a1770389cd2992c6ca3c5f76ad0a270e975374c182cde3e5dfb6fe",
}


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def expected_profile(
    *, device="cuda:0", dtype="bfloat16", attention_implementation="sdpa"
):
    """Declare a cache identity independently, before loading or scoring."""
    if dtype != "bfloat16":
        raise ValueError("Jina Stage A reference requires explicit bfloat16")
    if attention_implementation not in {"sdpa", "flash_attention_2"}:
        raise ValueError("Jina reference requires SDPA or official FlashAttention-2")
    if not isinstance(device, str) or (
        device != "cpu" and not device.startswith("cuda:")
    ):
        raise ValueError("Jina device must be cpu or an explicit CUDA device index")
    if device.startswith("cuda:") and not device[5:].isdigit():
        raise ValueError("invalid Jina CUDA device index")
    if device == "cpu" and attention_implementation != "sdpa":
        raise ValueError("FlashAttention-2 requires CUDA")
    flash_attention_version = None
    if attention_implementation == "flash_attention_2":
        try:
            flash_attention_version = importlib.metadata.version("flash-attn")
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(
                "FlashAttention-2 requires the declared frozen cuda extra"
            ) from error
        if flash_attention_version != "2.8.3":
            raise ValueError("Jina FlashAttention-2 requires the frozen 2.8.3 package")
    return {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "code_revision": MODEL_REVISION,
        "reference_source_sha256": SOURCE_SHA256,
        "adapter_implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "config_sha256": SNAPSHOT_SHA256["config.json"],
        "tokenizer_sha256": SNAPSHOT_SHA256["tokenizer.json"],
        "weights_sha256": SNAPSHOT_SHA256["model.safetensors"],
        "license": "CC-BY-NC-4.0",
        "backend": "transformers-native-official-rerank",
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": "4.57.3",
        "numpy_version": importlib.metadata.version("numpy"),
        "dtype": dtype,
        "device": device,
        "attention_implementation": attention_implementation,
        "flash_attention_version": flash_attention_version,
        "sdpa_math_policy": "disabled-on-cuda; allowed-for-cpu-reference",
        "input_order": "caller-order-preserved-within-native-blocks",
        "output_kind": "ranking",
        "ranking_output": "official-native-permutation-without-retiebreaking",
        "score_semantics": SCORE_SEMANTICS,
        "score_comparison": "within-model-ordering-only",
        "mathematical_score_range": [-1, 1],
        "instruction": None,
        "query_token_limit": QUERY_TOKENS,
        "query_overflow": "error-before-native-truncation",
        "document_token_limit": DOCUMENT_TOKENS,
        "model_max_tokens": CONTEXT_TOKENS,
        "block_document_cap": BLOCK_DOCUMENTS,
        "truncation_policy": "official-token-prefix-decode-and-reencode-v1",
        "block_policy": "official-append-then-capacity-flush-v1",
        "padding_side": "left",
        "top_n": None,
        "compile_model": False,
        "resource_preflight": "six-mask-equivalents-plus-hidden-states-v2",
        "allocator_preflight": "release-inactive-cache-before-driver-free-check",
    }


def format_native_prompt(query, documents):
    """Independent transcription of the pinned formatter, default instruction."""

    def sanitize(text):
        return text.replace(QUERY_MARKER, "").replace(DOC_MARKER, "")

    query = sanitize(query)
    documents = [sanitize(document) for document in documents]
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    prompt = (
        f"I will provide you with {len(documents)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    prompt += (
        "\n".join(
            f'<passage id="{index}">\n{document}{DOC_MARKER}\n</passage>'
            for index, document in enumerate(documents)
        )
        + "\n"
    )
    prompt += f"<query>\n{query}{QUERY_MARKER}\n</query>"
    return (
        prefix + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


@dataclass(frozen=True)
class PreparedJinaBlock:
    indexes: tuple[int, ...]
    prompt: str
    input_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparedJinaList:
    query: str
    native_query: str
    documents: tuple[str, ...]
    native_documents: tuple[str, ...]
    original_query_tokens: int
    original_document_tokens: tuple[int, ...]
    native_document_tokens: tuple[int, ...]
    blocks: tuple[PreparedJinaBlock, ...]

    def metadata(self):
        return {
            "candidate_count": len(self.documents),
            "query_sha256": _digest(self.query),
            "native_query_sha256_before_marker_sanitization": _digest(
                self.native_query
            ),
            "original_query_tokens": self.original_query_tokens,
            "query_was_truncated": False,
            "query_decode_applied": self.original_query_tokens >= QUERY_TOKENS,
            "documents": [
                {
                    "index": index,
                    "source_sha256": _digest(original),
                    "native_text_sha256_before_marker_sanitization": _digest(native),
                    "rendered_document_text_sha256": _digest(
                        native.replace(QUERY_MARKER, "").replace(DOC_MARKER, "")
                    ),
                    "original_tokens": original_tokens,
                    "native_item_tokens_before_reencode": native_tokens,
                    "tokens_truncated": max(0, original_tokens - native_tokens),
                    "decode_applied": native_tokens >= DOCUMENT_TOKENS,
                    "text_changed_by_native_truncation_decode": native != original,
                }
                for index, (
                    original,
                    native,
                    original_tokens,
                    native_tokens,
                ) in enumerate(
                    zip(
                        self.documents,
                        self.native_documents,
                        self.original_document_tokens,
                        self.native_document_tokens,
                        strict=True,
                    )
                )
            ],
            "blocks": [
                {
                    "indexes": list(block.indexes),
                    "token_count": len(block.input_ids),
                    "prompt_sha256": _digest(block.prompt),
                    "input_ids_sha256": _digest(",".join(map(str, block.input_ids))),
                    "document_marker_count": block.input_ids.count(DOC_TOKEN_ID),
                    "query_marker_count": block.input_ids.count(QUERY_TOKEN_ID),
                }
                for block in self.blocks
            ],
        }


class JinaRerankerPreparation:
    """Audit native truncation/block construction without changing its inputs."""

    def __init__(self, tokenizer):
        if (
            tokenizer.model_max_length != CONTEXT_TOKENS
            or tokenizer.pad_token_id is None
        ):
            raise ValueError(
                "Jina tokenizer context/padding differs from pinned source"
            )
        for marker, token_id in (
            (DOC_MARKER, DOC_TOKEN_ID),
            (QUERY_MARKER, QUERY_TOKEN_ID),
        ):
            if tokenizer.encode(marker, add_special_tokens=False) != [token_id]:
                raise ValueError("Jina embedding marker is not the pinned single token")
        self.tokenizer = tokenizer

    def prepare(self, query, documents):
        if not isinstance(query, str) or not isinstance(documents, (list, tuple)):
            raise TypeError("Jina expects one string query and a document list")
        if any(not isinstance(document, str) for document in documents):
            raise ValueError("Jina documents must all be strings")
        if self.tokenizer.model_max_length != CONTEXT_TOKENS:
            raise ValueError("Jina tokenizer context was mutated")
        query_ids = self.tokenizer(query, truncation=False)["input_ids"]
        if len(query_ids) > QUERY_TOKENS:
            raise ValueError("full Jina query exceeds native 1024-token limit")
        # Official code decodes even an exactly-full query. Preserve that path.
        native_query = (
            self.tokenizer.decode(query_ids)
            if len(query_ids) >= QUERY_TOKENS
            else query
        )
        original_lengths, native_lengths, native_docs = [], [], []
        for document in documents:
            original = self.tokenizer(document, truncation=False)["input_ids"]
            native = self.tokenizer(
                document, truncation=True, max_length=DOCUMENT_TOKENS
            )["input_ids"]
            original_lengths.append(len(original))
            native_lengths.append(len(native))
            native_docs.append(
                self.tokenizer.decode(native)
                if len(native) >= DOCUMENT_TOKENS
                else document
            )
        blocks, indexes = [], []
        capacity = CONTEXT_TOKENS - 2 * len(query_ids)

        def flush():
            prompt = format_native_prompt(
                native_query, [native_docs[index] for index in indexes]
            )
            ids = tuple(self.tokenizer(prompt, truncation=False)["input_ids"])
            if len(ids) > CONTEXT_TOKENS:
                raise ValueError(
                    "official Jina block plus markup exceeds context; no hidden splitting"
                )
            if (
                ids.count(DOC_TOKEN_ID) != len(indexes)
                or ids.count(QUERY_TOKEN_ID) != 1
            ):
                raise ValueError(
                    "Jina rendered marker counts differ from document coverage"
                )
            blocks.append(PreparedJinaBlock(tuple(indexes), prompt, ids))

        for index, length in enumerate(native_lengths):
            indexes.append(index)
            capacity -= length
            if len(indexes) >= BLOCK_DOCUMENTS or capacity <= DOCUMENT_TOKENS:
                flush()
                indexes = []
                capacity = CONTEXT_TOKENS - 2 * len(query_ids)
        if indexes:
            flush()
        return PreparedJinaList(
            query,
            native_query,
            tuple(documents),
            tuple(native_docs),
            len(query_ids),
            tuple(original_lengths),
            tuple(native_lengths),
            tuple(blocks),
        )


def estimate_additional_cuda_bytes(
    tokens, attention_implementation="sdpa", *, previous_block_tokens=0
):
    """Conservative preflight estimate, not a measured peak or memory promise."""
    masks = 6 * tokens * tokens if attention_implementation == "sdpa" else 0
    # In native rerank, the previous `outputs` stays live until the next block
    # call returns and assignment replaces it. Preserve that reference behavior.
    hidden_snapshots = 29 * (tokens + previous_block_tokens) * 1024 * 2
    qkv_and_mlp = tokens * (48 * 128 * 2 + 4 * 3072 * 2)
    return masks + hidden_snapshots + qkv_and_mlp + 512 * 1024 * 1024


def verify_snapshot(snapshot):
    """Verify approved source, tokenizer and LFS weight hashes before import."""
    snapshot = Path(snapshot)
    if (
        snapshot.name != MODEL_REVISION
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != "models--jinaai--jina-reranker-v3.5"
    ):
        raise ValueError("Jina requires the pinned local Hugging Face snapshot")
    for name, expected in SNAPSHOT_SHA256.items():
        digest = hashlib.sha256()
        with (snapshot / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Jina immutable snapshot hash mismatch: {name}")
    return snapshot


class NativeJinaReference:
    """Own the unmodified native model and audit its complete listwise call."""

    def __init__(self, tokenizer, model, *, attention_implementation="sdpa"):
        self._profile = expected_profile(
            device=str(model.device),
            dtype=str(model.dtype).removeprefix("torch."),
            attention_implementation=attention_implementation,
        )
        self.preparation = JinaRerankerPreparation(tokenizer)
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.model = model.eval()
        self.model._tokenizer = tokenizer
        self._lock = Lock()
        self.last_metrics = {}
        self._validate_runtime()

    def _validate_runtime(self):
        if self.model is None:
            raise RuntimeError("Jina reference is closed")
        config = self.model.config
        if (
            tuple(config.layer_types) != LAYER_TYPES
            or config.sliding_window != 1024
            or config.max_position_embeddings != CONTEXT_TOKENS
            or config._attn_implementation != self._profile["attention_implementation"]
            or str(self.model.dtype).removeprefix("torch.") != self._profile["dtype"]
            or str(self.model.device) != self._profile["device"]
            or self.model._tokenizer is not self.tokenizer
            or self.tokenizer.padding_side != "left"
        ):
            raise ValueError("Jina native runtime differs from frozen profile")

    @property
    def profile(self):
        self._validate_runtime()
        return copy.deepcopy(self._profile)

    @property
    def last_batch_metrics(self):
        return copy.deepcopy(self.last_metrics)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path=MODEL_NAME,
        *,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        device="cuda:0",
        dtype="bfloat16",
        attention_implementation="sdpa",
        local_files_only=True,
    ):
        expected_profile(
            device=device,
            dtype=dtype,
            attention_implementation=attention_implementation,
        )
        if (
            revision != MODEL_REVISION
            or tokenizer_revision != revision
            or local_files_only is not True
        ):
            raise ValueError(
                "Jina requires matching pinned revisions and local-only loading"
            )
        import torch
        import transformers

        if (
            transformers.__version__ != "4.57.3"
            or importlib.metadata.version("torch") != "2.8.0"
        ):
            raise ValueError(
                "Jina reference requires frozen Transformers 4.57.3 / Torch 2.8.0"
            )
        if str(model_name_or_path) == MODEL_NAME:
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                MODEL_NAME, revision=revision, local_files_only=True
            )
        else:
            snapshot = model_name_or_path
        snapshot = verify_snapshot(snapshot)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(snapshot),
            trust_remote_code=False,
            local_files_only=True,
            padding_side="left",
        )
        JinaRerankerPreparation(tokenizer)
        # Custom code is executed only after its exact reviewed bytes are verified.
        model = (
            transformers.AutoModel.from_pretrained(
                str(snapshot),
                revision=revision,
                code_revision=revision,
                local_files_only=True,
                trust_remote_code=True,
                dtype=getattr(torch, dtype),
                attn_implementation=attention_implementation,
            )
            .to(device)
            .eval()
        )
        return cls(tokenizer, model, attention_implementation=attention_implementation)

    def _cuda_preflight(self, prepared):
        if self._profile["device"] == "cpu" or not prepared.blocks:
            return nullcontext()
        import torch

        longest = max(len(block.input_ids) for block in prepared.blocks)
        reserved_before = torch.cuda.memory_reserved(self.model.device)
        allocated_before = torch.cuda.memory_allocated(self.model.device)
        # Driver free memory excludes reusable allocator cache. Releasing only
        # inactive cached segments avoids rejecting a smaller follow-up list
        # merely because the preceding native call populated that cache.
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(self.model.device)
        previous_tokens = 0
        estimates = []
        for block in prepared.blocks:
            estimates.append(
                estimate_additional_cuda_bytes(
                    len(block.input_ids),
                    self._profile["attention_implementation"],
                    previous_block_tokens=previous_tokens,
                )
            )
            previous_tokens = len(block.input_ids)
        estimate = max(estimates)
        self.last_metrics["resource_preflight"] = {
            "classification": "STATIC CONSERVATIVE ESTIMATE",
            "longest_native_block_tokens": longest,
            "free_cuda_bytes": free,
            "total_cuda_bytes": total,
            "estimated_additional_bytes": estimate,
            "allocated_bytes_before_cache_release": allocated_before,
            "reserved_bytes_before_cache_release": reserved_before,
            "inactive_cache_released_before_free_measurement": True,
            "per_block_estimated_additional_bytes": estimates,
            "native_prior_block_hidden_state_retention_included": True,
        }
        if free < estimate:
            raise RuntimeError(
                "Jina full native list exceeds conservative CUDA preflight; no documents were dropped"
            )
        if self._profile["attention_implementation"] == "sdpa":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # All efficient backends remain eligible; failure cannot silently
            # allocate a 16-head quadratic FP32 math attention matrix.
            return sdpa_kernel(
                [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                ]
            )
        return nullcontext()

    def score(self, query, documents):
        return self._execute(query, documents, ranking=False)

    def rank(self, query, documents):
        """Preserve the official permutation, including its exact-score ties."""
        return self._execute(query, documents, ranking=True)

    def _execute(self, query, documents, *, ranking):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "one Jina native reference cannot score concurrent lists"
            )
        try:
            self.last_metrics = {"status": "RUNNING"}
            scores = self._score(query, documents)
            return (
                list(self.last_metrics["native_sorted_indexes"]) if ranking else scores
            )
        except Exception as error:
            self.last_metrics.update(
                status="ERROR", error_type=type(error).__name__, error=str(error)
            )
            raise
        finally:
            self._lock.release()

    def _score(self, query, documents):
        self._validate_runtime()
        prepared = self.preparation.prepare(query, documents)
        self.last_metrics = {
            **prepared.metadata(),
            "status": "RUNNING",
            "observed_blocks": [],
            "math_sdpa_allowed_on_cuda": False,
        }
        if not documents:
            self.last_metrics.update(
                status="PASS",
                native_sorted_indexes=[],
                raw_scores_in_input_order=[],
                restored_count=0,
            )
            return []
        import torch

        try:
            attention_context = self._cuda_preflight(prepared)
            model = self.model
            original_forward = model.forward
            had_instance_forward = "forward" in model.__dict__
            previous_instance_forward = model.__dict__.get("forward")

            def observed_forward(*args, **kwargs):
                index = len(self.last_metrics["observed_blocks"])
                if args or index >= len(prepared.blocks) or "input_ids" not in kwargs:
                    raise ValueError(
                        "Jina official forward block count/signature drift"
                    )
                ids = kwargs["input_ids"]
                expected = prepared.blocks[index]
                if (
                    ids.ndim != 2
                    or ids.shape[0] != 1
                    or tuple(ids[0].tolist()) != expected.input_ids
                ):
                    raise ValueError(
                        "Jina official forward token IDs differ from audited preparation"
                    )
                mask = kwargs.get("attention_mask")
                if (
                    mask is None
                    or tuple(mask.shape) != tuple(ids.shape)
                    or not bool(mask.all())
                ):
                    raise ValueError(
                        "Jina single native block has unexpected padding mask"
                    )
                self.last_metrics["observed_blocks"].append(
                    {
                        "indexes": list(expected.indexes),
                        "token_count": ids.shape[1],
                        "token_ids_match": True,
                    }
                )
                return original_forward(**kwargs)

            model.forward = observed_forward
            try:
                with torch.inference_mode(), attention_context:
                    results = model.rerank(
                        query, list(documents), top_n=None, return_embeddings=False
                    )
            finally:
                if had_instance_forward:
                    model.forward = previous_instance_forward
                else:
                    del model.forward
            if len(self.last_metrics["observed_blocks"]) != len(prepared.blocks):
                raise ValueError("Jina official native call skipped a prepared block")
            if not isinstance(results, list) or len(results) != len(documents):
                raise ValueError(
                    "Jina official native output omits requested documents"
                )
            restored, permutation = [None] * len(documents), []
            for row in results:
                index = row.get("index")
                if isinstance(index, bool) or not hasattr(index, "__index__"):
                    raise ValueError("Jina returned a noninteger original index")
                index = int(index)
                if not 0 <= index < len(documents) or restored[index] is not None:
                    raise ValueError(
                        "Jina returned duplicate/out-of-range original indexes"
                    )
                if row.get("document") != documents[index]:
                    raise ValueError(
                        "Jina result document differs from original indexed text"
                    )
                score = float(row["relevance_score"])
                if not math.isfinite(score) or not -1.001 <= score <= 1.001:
                    raise ValueError(
                        "Jina returned a nonfinite or non-cosine raw score"
                    )
                restored[index] = score
                permutation.append(index)
            self.last_metrics.update(
                status="PASS",
                native_sorted_indexes=permutation,
                native_sorted_scores=[restored[index] for index in permutation],
                raw_scores_in_input_order=restored,
                restored_count=len(restored),
            )
            return restored
        except Exception as error:
            self.last_metrics.update(
                status="ERROR", error_type=type(error).__name__, error=str(error)
            )
            raise

    def close(self, *, empty_cuda_cache=False):
        self.model = None
        self.tokenizer = None
        self.preparation = None
        gc.collect()
        if empty_cuda_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
