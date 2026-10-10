# Mandol HTTP inference bridges

CPU HTTP bridges connect Mandol Competition to external model services using
persistent clients. They validate model identities and translate provider/native
serving schemas without loading GPU models into the API environment.

## Install

After the root workspace installation:

```bash
uv pip install --python .venv/bin/python --no-deps -e services/inference
```

The distribution name is `mandol-inference-worker`; its import name is
`mandol_inference`.

## Frozen bridge factories

| Factory | Purpose | Default documented port |
|---|---|---|
| `mandol_inference.dashscope_bridge:create_app` | Paired text-embedding-v4 Dense/native Sparse | 18620 |
| `mandol_inference.vllm_bridge:create_reranker_app` | BGE native-server schema and identity bridge | 18621 |

Start these factories with Uvicorn's `--factory` option, one worker, and a private
bind address. See [Local deployment](../../docs/deployment/Local-Reproduction.md)
for commands and environment variables, and
[Inference contracts](../../docs/deployment/Inference.md) for model recipes.
Querit is served directly by its native classification server.

## Optional model worker

`multimodal.py` retains an optional model-worker API for research and compatibility
tests. Its `create_app` is separate from the frozen bridge factories. The `models`
extra installs Torch/Transformers/SentenceTransformers and belongs in a separate
environment. It does not provision the pinned Querit/BGE vLLM servers.
Qwen/SPLADE research branches are separate from the frozen native-v4 method.

The package is distributed under [Apache-2.0](LICENSE); attribution is retained
in [NOTICE](NOTICE) and [project attribution](../../docs/ATTRIBUTION.md).
External model artifacts retain their own terms.
