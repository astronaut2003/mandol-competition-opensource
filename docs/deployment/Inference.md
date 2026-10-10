# Inference contracts and model serving

The Memory APIs and HTTP bridges run in a CPU application environment. Native
model servers use separate GPU environments and communicate over HTTP. Obtain
model artifacts under their applicable [terms](../ATTRIBUTION.md); weights are
not included in this project.

## Frozen contracts

| Capability | Model / revision | Contract |
|---|---|---|
| Dense | text-embedding-v4 / provider-managed-unattested | 1024 dimensions; provider L2 validated; query/document input types; empty prefixes; batch ≤10; max 8192; no client truncation |
| Native Sparse | Same paired v4 provider request | uint32 indices; finite nonnegative weights; threshold 0; no normalization |
| Textual reranker | Querit/Querit-4B / e557bb5ce6555f16b33dbb7d2ca62e8d27c14c13 | BF16; vLLM 0.29.0; final CLS 151665; original-order B1; complete pair ≤40960; probs[1]−probs[0] |
| Multimodal reranker | BAAI/bge-reranker-v2-m3 / 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e | BF16; vLLM 0.25.1; pair-right ≤4096; microbatch 1; sigmoid, higher-is-better |
| Multimodal VLM | Logical gpt-4o-mini-closeai / wire gpt-4o-mini | multimodal_sgm_v1; schema 1; external OpenAI-compatible base URL |

T0/M0 consume v4 Dense and native Sparse. C0 uses optional Dense with its frozen
Dot metric and no Sparse or neural reranker. GPU precision, token limits, pair
construction, and score interpretation are part of the method; other recipes
need separately versioned profiles and validation.

An HTTP model alias cannot identify provider weights, tokenizer revision,
pooling internals, or dtype. Provider-managed fields are recorded accordingly.
Serving bindings and `/version`/`/v1/models` validate declared local identities;
they do not independently prove tensor or numerical equality.

## Prepare Querit artifacts

Download permitted original artifacts for the pinned revision into a directory
outside the checkout. The preparation script checks each file against the
`FILE_MANIFEST` constant in the Querit reference adapter. It renames safetensors
header fields `head.weight/bias` to `score.weight/bias`, preserves tensor payload
bytes, and writes a conversion manifest. It refuses an existing output directory.

```bash
.venv/bin/python scripts/qualification/prepare_querit_vllm.py \
  --source /models/querit-source/e557bb5ce6555f16b33dbb7d2ca62e8d27c14c13 \
  --output /models/Querit-4B
```

These are example artifact paths. This command does not download weights.

## Querit server

Use a separate environment pinned to **vLLM 0.29.0** and the converted artifacts:

```bash
vllm serve /models/Querit-4B --runner pooling --dtype bfloat16 \
  --enforce-eager --served-model-name querit-4b --host 127.0.0.1 --port 8003 \
  --disable-uvicorn-access-log --no-enable-log-requests \
  --max-model-len 40960 --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --kv-cache-memory-bytes 6442450944 --gpu-memory-utilization 0.60 \
  --pooler-config '{"seq_pooling_type":"LAST","use_activation":true}'
```

Admission checks `/health`, `/version`, `/v1/models`, `/openapi.json`, `/tokenize`,
`/detokenize`, and `/classify`. Model-side pooling ends at the explicit final CLS;
the adapter does not replace the pair prompt or generate an answer.

## BGE server and bridge

Use a separate environment pinned to **vLLM 0.25.1**:

```bash
vllm serve /models/bge-reranker-v2-m3 --runner pooling --dtype bfloat16 \
  --enforce-eager --served-model-name BAAI/bge-reranker-v2-m3 \
  --host 127.0.0.1 --port 8004 --disable-uvicorn-access-log --no-enable-log-requests \
  --max-model-len 4096 --max-num-batched-tokens 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.12
```

Set `MANDOL_VLLM_RERANKER_ENDPOINT=http://127.0.0.1:8004` and start the CPU bridge
at port 18621 as shown in [Local deployment](Local-Reproduction.md#4-start-querit-and-bge).
Configure both rerankers in a local copy of
[serving-bindings.example.json](../../configs/deployment/serving-bindings.example.json)
to match actual roots and identities.

## Resource planning and validation

The resource arguments above describe the frozen serving recipes, not a minimum
GPU specification for every deployment. Available memory must fit the pinned
models, context lengths, and scheduling limits. Validate model identity, complete
pair/tokenizer behavior, and score semantics on the intended hardware before
using a different serving environment.

Warmup includes paired v4 encodes, a complete Querit pair, a BGE pair, and a small
VLM image. It may incur provider charges but does not write canonical memories.
Use the API environment only for CPU bridges; installing its optional historical
`models` extra does not provision these two native vLLM servers.
