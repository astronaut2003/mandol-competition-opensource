"""Explicit offline model channels for coding-memory experiments.

No default model IDs, downloads, provider keys, or silent zero-score fallback.
The caller owns candidate scope and can fall back to lexical retrieval.
"""

from __future__ import annotations

from pathlib import Path


class LocalDenseEncoder:
    """Shared local encoder; each index explicitly contains one user's units."""

    def __init__(self, model_path, *, device="cuda", max_length=2048):
        import threading
        import torch
        from sentence_transformers import SentenceTransformer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError("A prepared local embedding snapshot is required")
        self.lock = threading.RLock()
        self.model = SentenceTransformer(str(path), local_files_only=True, device=device,
                                         model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                                                       "attn_implementation": "sdpa"})
        self.model.max_seq_length = max_length

    def encode(self, texts):
        with self.lock:
            return self.model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)

    def index(self, documents):
        return _DenseIndex(self, documents)


class _DenseIndex:
    def __init__(self, encoder, documents):
        self.encoder = encoder
        self.ids = sorted(documents)
        self.vectors = encoder.encode([documents[uid] for uid in self.ids]) if self.ids else None

    def search(self, query, *, top_k):
        if not self.ids:
            return []
        query_vector = self.encoder.encode([query])[0]
        scores = self.vectors @ query_vector
        return sorted(zip(self.ids, map(float, scores)), key=lambda pair: (-pair[1], pair[0]))[:top_k]


class LocalQwenReranker:
    """Qwen3 yes/no scoring with a suffix preserved across truncation.

    Protocol reference: Qwen/Qwen3-Reranker-0.6B official model card.
    ``legacy`` exists only to measure Mandol's previous prompt/Yes-logit
    protocol; it is not selected by a runtime default.
    """

    default_instruction = "Given a web search query, retrieve relevant passages that answer the query"
    coding_instruction = "Given a software engineering task, retrieve historical tasks whose evidence or engineering experience is relevant to this task"

    def __init__(self, model_path: str | Path, *, device="cuda", batch_size=4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError("A prepared local model snapshot is required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device).eval()
        self.batch_size = batch_size
        self.true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.legacy_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
                  'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
        self.config = {"model_path": str(path), "revision": path.name, "device": device,
                       "batch_size": batch_size, "dtype": str(self.model.dtype),
                       "attention": "sdpa", "torch": torch.__version__}

    def score(self, query: str, documents: list[str], *, max_length=2048, instruction=None, protocol="qwen3"):
        import torch

        if protocol not in {"qwen3", "legacy"} or max_length < len(self.prefix) + len(self.suffix) + 32:
            raise ValueError("Invalid Qwen scoring protocol or token budget")
        instruction = instruction or (self.default_instruction if protocol == "qwen3" else
                                      "Given a query and a passage, determine if the passage is relevant to answer the query. Please answer Yes or No.")
        scores = []
        for start in range(0, len(documents), self.batch_size):
            bodies = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                      for doc in documents[start:start + self.batch_size]]
            if protocol == "legacy":
                prefix = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
                          'Note that the answer can only be "Yes" or "No".<|im_end|>\n<|im_start|>user\n')
                prompts = [prefix + body + "<|im_end|>\n<|im_start|>assistant\n" for body in bodies]
                inputs = self.tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            else:
                encoded = self.tokenizer(bodies, padding=False, truncation=True,
                                         max_length=max_length - len(self.prefix) - len(self.suffix),
                                         add_special_tokens=False, return_attention_mask=False)
                encoded["input_ids"] = [self.prefix + ids + self.suffix for ids in encoded["input_ids"]]
                inputs = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                # Only the last hidden state needs a vocabulary projection.
                logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                values = logits[:, self.legacy_id] if protocol == "legacy" else torch.sigmoid(logits[:, self.true_id] - logits[:, self.false_id])
                if not torch.isfinite(values).all():
                    raise RuntimeError("Non-finite reranker scores")
                scores.extend(values.cpu().tolist())
        return scores


class LocalBGEReranker:
    """Explicit local BGE sequence-classification scorer for A/B experiments."""

    def __init__(self, model_path, *, device="cuda", batch_size=4):
        import torch
        import transformers

        path = Path(model_path)
        if not path.is_dir() or batch_size < 1:
            raise ValueError("A prepared local snapshot and positive batch size are required")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(device).eval()
        self.batch_size = batch_size
        self.config = {"model_path": str(path), "revision": path.name, "device": device,
                       "batch_size": batch_size, "dtype": str(self.model.dtype), "attention": "sdpa",
                       "torch": torch.__version__, "transformers": transformers.__version__}

    def score(self, query, documents, *, max_length=512, instruction=None, protocol="bge"):
        import torch

        if protocol != "bge" or instruction is not None:
            raise ValueError("BGE uses query/document pairs without an instruction prompt")
        scores = []
        for start in range(0, len(documents), self.batch_size):
            pairs = [(query, doc) for doc in documents[start:start + self.batch_size]]
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                values = self.model(**inputs).logits.view(-1).float()
                if not torch.isfinite(values).all():
                    raise RuntimeError("Non-finite BGE scores")
                scores.extend(values.cpu().tolist())
        return scores
