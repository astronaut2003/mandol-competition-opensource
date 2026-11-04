# Inference services

The current cloud recipe uses this package's CPU-only v4 bridge. It calls the
authorized `text-embedding-v4` service and exposes paired Dense/native Sparse
results to the three isolated Memory APIs. Native Qwen3.7 reranking is called
directly by the API clients; it does not require a separate reranker GPU worker.

From the repository root:

```bash
uv sync --frozen --python 3.12.13 --extra all --extra dev
uv pip install --python .venv/bin/python --no-deps -e services/inference
.venv/bin/python -m uvicorn mandol_inference.dashscope_bridge:create_app \
  --factory --host 127.0.0.1 --port 18620 --workers 1 --no-access-log
```

Supply `DASHSCOPE_API_KEY` through the bridge's own environment. Do not embed
credentials in URLs, source, logs, or committed configuration. API processes
also need their own provider credential for the native reranker. A shared
`MANDOL_RELIABILITY_CONFIG` can coordinate the provider account's real windows
across bridge/API processes. Configure your own quota limits and a fresh shared
admission database as explained in [local reproduction](../../docs/deployment/Local-Reproduction.md).

Admission/health recovery can issue real provider requests. Never attach
synthetic regression to a live service; use the fixtures and owned scratch
Qdrant described in [testing](../../docs/Reproduction.md). API and bridge
processes use the existing end-to-end request budget, typed errors and bounded
retry/cooldown policy. They cannot guarantee that a provider never returns 429.

The optional `models` extra and `vllm_bridge` adapters provide historical
GPU serving configurations with their own model, revision and precision
bindings. The standard CPU workflow uses the cloud model services above.

See [current inference contracts](../../docs/deployment/Inference.md) for token
budgets, tokenizer provenance, multi-view rank fusion and staged VLM/OCR.
