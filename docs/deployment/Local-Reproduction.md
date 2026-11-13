# Local reproduction

Use Linux, Python 3.12, and the locked installation in [README](../../README.md).
This guide selects `mm-source-preserving-add-20261113-v3`, the current cloud workflow. Earlier Querit/BGE examples are
historical recipes with different model bindings.

## 1. Create isolated storage and Qdrant

```bash
docker run -d --name mandol-public-qdrant \
  -p 127.0.0.1:6333:6333 \
  -v mandol-public-qdrant:/qdrant/storage qdrant/qdrant:v1.19.1
mkdir -p configs/local
cp configs/deployment/current-cloud.example.toml configs/local/deployment.toml
```

Use unused container/volume names. The default runtime roots are `data/textual`,
`data/code` and `data/multimodal`, with distinct `*_aml_v4_v1` collections.
Do not point reproduction or regression tests at a live memory store.
The APIs never manage Qdrant's lifecycle. Keep Multimodal MediaStore with its
canonical state when moving or backing up a completed reproduction.

## 2. Configure authorized services and secrets

In your local TOML, replace `https://vlm.example.invalid/v1` with an authorized
OpenAI-compatible service that supports image input and the `gpt-4o-mini` wire
model. The current recipe uses direct native DashScope Qwen3.7 reranking and a
local CPU bridge for paired v4 Dense/native Sparse. The provider endpoint can
be adjusted to the authorized region, preserving model and API contracts.

Supply these variables through your environment or secret manager:

| Variable | Purpose |
|---|---|
| `MANDOL_MEMORY_SYSTEM_KEY` or its `_FILE` form | Authenticate calls to the Memory APIs. |
| `DASHSCOPE_API_KEY` | v4 bridge and native Qwen reranker account credential. |
| `CLOSEAI_API_KEY` | Credential read by the current VLM client; `MANDOL_MM_VLM_API_KEY_ENV` can select another variable name. |
| `MANDOL_MULTIMODAL_VLM_ENDPOINT` | Optional environment override for the configured VLM URL. |
| `MANDOL_QDRANT_API_KEY` or its `_FILE` form | Optional Qdrant authentication. |
| `MANDOL_RELIABILITY_CONFIG` | Optional local JSON for process-shared provider admission. |

TOML does not expand shell variables. `.env.example` is a reference, not an
automatically loaded file. Do not configure a secret value and its file mount
simultaneously. Provider clients may have different supported mount aliases;
the v4 bridge expects its provider key in the service environment.

The application `DASHSCOPE_API_KEY` and a participant leaderboard/Eval Key have
different purposes. This reproduction does not need a leaderboard key or
create/continue an official evaluation.

## 3. Start the CPU embedding bridge

In a terminal with the authorized `DASHSCOPE_API_KEY` supplied:

```bash
.venv/bin/python -m uvicorn mandol_inference.dashscope_bridge:create_app \
  --factory --host 127.0.0.1 --port 18620 --workers 1 --no-access-log
```

All tracks use this origin; Textual and Multimodal consume paired Dense/native
Sparse. Coding's selected policy supports native Sparse/Dense with its explicit
frontier/fallback behavior. No SPLADE GPU service is required. Bridge admission
and API startup may make real model calls and consume the authorized account's
quota. Keep these services private.

## 4. Configure shared provider admission

If using `MANDOL_RELIABILITY_CONFIG`, create `configs/local/reliability.json`:

```json
{
  "retry_budget_seconds": 1740,
  "retry_max_attempts": 8,
  "provider_origins": ["https://dashscope.aliyuncs.com", "https://cn-beijing.maas.qianwenaiapi.com"],
  "state_path": "data/provider-admission.sqlite3",
  "models": {
    "text-embedding-v4": {"usage_limit": 80000, "usage_period": 60, "request_limit": 80, "request_period": 60},
    "qwen3.7-text-rerank": {"usage_limit": 80000, "usage_period": 60, "request_limit": 80, "request_period": 60}
  }
}
```

These numbers are artificial examples, not account entitlements. Replace each
window with **80% of your own verified provider limit**, preserving its period.
Use an absolute path for `state_path` and share the same file among all bridge/API
processes using that provider account. Configure every actual provider origin.
The admission database contains counters, not memories or credentials. Do not
reuse any existing database from another deployment.

## 5. Start one process per API

In three terminals with the necessary environment:

```bash
.venv/bin/python scripts/start_api.py textual --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py code --config configs/local/deployment.toml
.venv/bin/python scripts/start_api.py multimodal --config configs/local/deployment.toml --initialize
```

Use `--initialize` only for the first Multimodal start against fresh storage.
It creates and verifies the new projection with a full initial audit, and refuses
an existing state database. On later starts omit `--initialize` to use the selected
fast restart path. Missing projections for existing state remain an error.

The launcher checks the public integrity manifest, selects the versioned OCR
prompt and `source-grounded-recall-v4` evidence policy, and uses 1,740-second request budgets with
8 active + 8 waiting operations per process. Ports are 18600, 18601 and 18602 on
loopback. Explicitly opting into another profile, prompt, model or precision
defines a different reproduction method. See [current workflow](../method/Current-Workflow.md).

Supply an HTTPS reverse proxy/access policy when exposing the APIs remotely.
The application does not install certificates, alter DNS/firewall rules, or
discover/stop unrelated processes. Stop foreground processes using Ctrl-C and
allow ongoing requests to drain before restarting the SQLite writer.

## Validation and troubleshooting

First run synthetic tests and then your own artificial Add/Search example.
Check credentials, endpoint contract, provider capacity, tokenizer resources,
and typed error metadata when a dependency is unavailable. A healthy API does
not by itself establish provider/model equivalence or retrieval accuracy.

Persist canonical state, MediaStore and matching vector projections across
normal restarts. Do not change model identities under an existing collection,
clear caches to hide failures, or restore an older memory database over newer
data. Offline repair utilities require their own ownership, identity and
stopped-writer checks; fresh reproduction does not require historical repair.

## Visual recovery policy and quality counts

`scripts/start_api.py` explicitly selects `MANDOL_MM_VISION_SOURCE_FALLBACK=1`
and `MANDOL_MM_FULL_DATABASE_AUDIT=0`. The same flags reach legacy, local and
cluster factories. Local/cluster model bindings still define their own recipes;
changing topology is not evidence that different provider models are equivalent.

The frozen launcher rejects conflicting values. A manually configured strict-503
deployment can set source fallback to `0` using the ASGI factory directly; this
is a different reproduction selection and must be recorded as such.

Quality metadata distinguishes `complete`, `incomplete_ocr`, `source_only` and
`unverified`. On your own stopped or read-only-audited reproduction store:

```bash
.venv/bin/python -m mandol_competition.multimodal.add_quality \
  --database /absolute/path/to/your/state.sqlite3 --timeout 30
```

The command prints aggregate counts and rates, no user/request identifiers.
Only visible published images carrying the new quality marker enter the rates;
older unclassified images and failed Add attempts are separate. A zero
denominator produces `null`, not a claimed zero failure rate. It does not
re-extract or rewrite memories. Include Add failures when assessing completeness.
