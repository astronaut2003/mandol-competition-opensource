# Local deployment

Run these commands from the repository root after the
[Quick Start installation](../../README.md#quick-start). Use new storage and
your own authorized model services. This guide deploys the frozen T0/C0/M0
profiles; storage locations and endpoint origins are configurable.

## 1. Start Qdrant

```bash
docker run -d --name mandol-public-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v mandol-public-qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
```

Use an unused name, port, and volume. The application uses qdrant-client **1.19.0**.
With the supplied configuration, track roots are `data/textual`, `data/code`,
and `data/multimodal`; collections are `textual_aml_v4_v1`, `code_aml_v4_v1`, and
`multimodal_aml_v4_v1`. Root paths must be distinct and non-nested. Keep the
Multimodal MediaStore with its canonical SQLite state.

Qdrant and inference ports should remain private. The API processes do not start,
stop, or clear Qdrant. Never attach destructive tests to an existing deployment.

## 2. Configure local templates

```bash
mkdir -p configs/local
cp configs/deployment/frozen-bf16.example.toml configs/local/deployment.toml
cp configs/deployment/serving-bindings.example.json configs/local/serving-bindings.json
export MANDOL_INFERENCE_BINDINGS="$PWD/configs/local/serving-bindings.json"
```

Edit the TOML to set storage and endpoint locations. Replace its VLM
`example.invalid` origin. Edit the JSON bindings to match actual model artifact
roots and served aliases, preserving the pinned model revisions, serving
versions, and BF16 recipes. Set bindings before initializing canonical storage;
the runtime refuses incompatible profile/precision adoption.

TOML does not expand `${VARIABLE}` strings. Supported environment overrides take
precedence over configured values:

| Setting | Environment variable |
|---|---|
| Runtime parent directory | `MANDOL_RUNTIME_ROOT` |
| Qdrant origin | `MANDOL_QDRANT_ENDPOINT` |
| Track storage | `MANDOL_<TRACK>_STORAGE_ROOT` |
| Track namespace | `MANDOL_<TRACK>_NAMESPACE` |
| Track inference origin | `MANDOL_<TRACK>_<CHANNEL>_ENDPOINT` |
| Serving bindings JSON | `MANDOL_INFERENCE_BINDINGS` |
| Memory API credential | `MANDOL_MEMORY_SYSTEM_KEY` or `MANDOL_MEMORY_SYSTEM_KEY_FILE` |
| Qdrant credential, if required | `MANDOL_QDRANT_API_KEY` or `MANDOL_QDRANT_API_KEY_FILE` |

`<TRACK>` is `TEXTUAL`, `CODE`, or `MULTIMODAL`; `<CHANNEL>` is an applicable
`EMBEDDING`, `SPARSE`, `RERANKER`, or `VLM`. Track-specific Memory System Key
variables and their `_FILE` forms override the shared API credential. Do not set
both a secret's direct value and its file mount.

Provider clients read `DASHSCOPE_API_KEY` and `CLOSEAI_API_KEY` from their service
environments. Inject them with your secret manager; these clients do not
automatically read `_FILE` aliases. [.env.example](../../.env.example) is a
variable-name reference and is not loaded automatically. Do not put secrets in
the TOML, serving JSON, commands, or URLs.

## 3. Start the CPU v4 bridge

The HTTP bridge package is installed by the Quick Start. In its environment with
`DASHSCOPE_API_KEY` supplied:

```bash
.venv/bin/python -m uvicorn mandol_inference.dashscope_bridge:create_app \
  --factory --host 127.0.0.1 --port 18620 --workers 1 --no-access-log
```

The bridge issues paired `text-embedding-v4` Dense/native Sparse requests to the
provider. `AML_V4_PROVIDER_ORIGIN` overrides the provider origin if needed.
Health/admission may make real provider requests. T0/M0 consume both lanes;
C0 consumes only Dense and retains lexical fallback. No SPLADE server is needed.

## 4. Start Querit and BGE

Use [Inference deployment](Inference.md) for the pinned GPU environments and
serving arguments. Querit exposes its native classification endpoints directly.
The CPU BGE bridge translates the native serving schema:

```bash
export MANDOL_MM_RETRIEVAL_PROFILE="$PWD/configs/profiles/aml-official-eligible/multimodal-v1.json"
export MANDOL_VLLM_RERANKER_ENDPOINT=http://127.0.0.1:8004
.venv/bin/python -m uvicorn mandol_inference.vllm_bridge:create_reranker_app \
  --factory --host 127.0.0.1 --port 18621 --workers 1 --no-access-log
```

Optional `MANDOL_TEXTUAL_RERANKER_REPLICAS` and
`MANDOL_MULTIMODAL_RERANKER_REPLICAS` select same-model replicas. One replica
scores a complete Search attempt; bounded failover discards partial results and
rescores the complete candidate set. Keep model revision, dtype, and recipe
identical within each pool.

## 5. Configure VLM and start the APIs

Set `MANDOL_MULTIMODAL_VLM_ENDPOINT` to your authorized OpenAI-compatible base URL
and supply `CLOSEAI_API_KEY` in the API environment. The logical model is
`gpt-4o-mini-closeai`; the wire request model is `gpt-4o-mini`. An alternative
credential variable name can be selected with `MANDOL_MM_VLM_API_KEY_ENV`.
For the unified TOML, `MANDOL_MM_VLM_BASE_URL` alone does not override a configured
unified VLM endpoint; use `MANDOL_MULTIMODAL_VLM_ENDPOINT`.

With Qdrant and inference services ready, run one track per terminal:

```bash
.venv/bin/python scripts/start_api.py textual --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py code --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py multimodal --config configs/local/deployment.toml
```

The launcher binds to loopback ports **18600 / 18601 / 18602**, verifies frozen
artifacts, and rejects incompatible research/ranking overrides. It uses one
worker per track. Use an HTTPS reverse proxy with an access policy for remote
clients; keep storage, inference, and operational routes on a trusted network.
Warmup validates dependencies before readiness and may call authorized models.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing Memory System Key | Supply the environment variable or its supported `_FILE` mount. |
| `/health` succeeds but `/readyz` fails | Check the track's inference endpoints, model identities, and publication state. |
| VLM connection or identity failure | Replace the placeholder origin, supply the configured credential variable, and verify the wire model. |
| Frozen startup rejects an override | Unset research/quality variables; use the fixed profile and model/prompt/schema selections. |
| Serving contract mismatch | Compare bindings with artifact root, served alias, `/version`, and dtype. |
| Collection/profile/namespace mismatch | Restore matching bindings and canonical state; do not clear or silently adopt a collection. |
| C0 Dense is degraded | Check provider/Qdrant availability and wait for a complete projection frontier; lexical Add/Search remains available. |

Inspect operational status without logging memory payloads or credentials. Stop
only processes you started, using Ctrl-C/SIGTERM and graceful shutdown before
restarting a SQLite writer.

## Persistence and recovery

Keep runtime roots, matching serving identities, and the Qdrant volume across
restarts. Published state is reused and pending projection work is reconciled.
T0/M0 keep their required-vector acknowledgement policy; C0 can remain lexical
while optional Dense is unavailable.

Back up consistent SQLite state together with MediaStore. Qdrant is a vector
projection and does not replace canonical state. Validate restores on scratch
resources before operational use. A relocated Multimodal retrieval profile is
accepted only if its bytes match the frozen artifact.
