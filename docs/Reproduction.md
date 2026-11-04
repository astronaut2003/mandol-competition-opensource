# Testing and reproduction

This guide separates mechanism tests, real-model runs, and official leaderboard
evaluation. All commands run from the repository root after the
[Quick Start installation](../README.md#installation).

## What the tests cover

| Test scope | Resources | What it verifies |
|---|---|---|
| Unit and fault tests | Deterministic model fixtures, temporary files | Schemas, ranking contracts, failures, idempotency, isolation, and recovery |
| SQLite integration | Fresh SQLite/WAL databases | Canonical state, ledgers, publication frontiers, and restart behavior |
| Qdrant integration | Owned disposable Qdrant 1.19.1 | Real vector publication, identity checks, retrieval, and reconciliation |
| Synthetic HTTP API | Actual Uvicorn factories with fixture inference | Add/Search/Health wire contracts and lifecycle behavior |
| Runtime integrity | Checked-in selections and source/profile hashes | Current public reproduction recipe |
| Real-model integration | Authorized external endpoints and model artifacts | API behavior with real inference dependencies |

## Local tests

For a short check without model credentials or Qdrant:

```bash
.venv/bin/python -m pytest tests/common/test_neutral_contracts.py tests/common/test_retrieval_primitives.py -q
python3 deployment/cluster/frozen.py
```

For the complete local regression suite:

```bash
docker run -d --name mandol-opensource-tests \
  --label mandol.test=opensource \
  -p 127.0.0.1:19633:6333 qdrant/qdrant:v1.19.1
.venv/bin/python scripts/run_tests.py \
  --qdrant http://127.0.0.1:19633 --container mandol-opensource-tests
```

Use an unused container name and port. The controller checks ownership and empty
storage, strips ambient model/provider configuration, and writes results under
`.test-results/`. Never point these tests at production or benchmark storage.
Delete or recreate only the disposable container you created for testing; the
controller does not manage unrelated services.

Additional process-restart tests need an executable Qdrant 1.19.1 binary and its
system libraries. Pass `--qdrant-binary /absolute/path/to/qdrant` to the controller
to enable them. They start their own new temporary storage and process; without
this explicit binary, these cases skip rather than adopt an existing server.

Synthetic inference tests are reproducible checks of system behavior. They do
not measure real-model quality or produce official AML scores.

## Real-model HTTP integration

Deploy Qdrant, the paired v4 bridge, native Qwen3.7 reranking, and an authorized VLM
using [Local deployment](deployment/Local-Reproduction.md) and
[Inference deployment](deployment/Inference.md). Use fresh runtime roots and
collections, matching versioned profiles, and serving bindings for the actual
models. The current cloud recipe does not require local GPU serving.

The opt-in HTTP contract tests exercise all three deployed APIs. Point the URLs
at your disposable deployment, and set `MANDOL_AML_KEY_FILE` to a file outside
the checkout containing that deployment's Memory System Key:

```bash
export MANDOL_AML_TEXTUAL_URL=http://127.0.0.1:18600
export MANDOL_AML_CODE_URL=http://127.0.0.1:18601
export MANDOL_AML_MULTIMODAL_URL=http://127.0.0.1:18602
# Set MANDOL_AML_KEY_FILE using your secret manager or private key-file path.
MANDOL_AML_REAL=1 .venv/bin/python -m pytest tests/aml_contract/test_real_wire.py -q --allow-remote-inference
```

These tests write synthetic histories and images, perform Search, and check
authentication, replay, conflict handling, and isolation. They can make real
provider requests and leave their test records in the disposable storage. They
do not run the official platform evaluator or measure every retrieval channel.

For quality experiments, submit your own permitted histories and queries through
[Add/Search](API.md). Record method hashes, model revisions, serving versions,
dtype, provider identity, data provenance, and evaluation protocol. Do not
substitute fixture output or skipped checks for a real-model result. Provider
calls can incur charges. Historical self-hosted GPU model recipes have their own
serving and memory requirements.

## Optional local QA

The local Textual QA tooling supports LoCoMo and LongMemEval prompt/judge
semantics from a pinned public AML evaluation source. The upstream scripts and
datasets are not distributed with this project.

Retrieve the scripts explicitly and verify the cache:

```bash
python3 scripts/fetch_aml_protocol_sources.py
python3 scripts/fetch_aml_protocol_sources.py --verify-only
```

The default location is
`$XDG_CACHE_HOME/mandol-competition/aml/1b8142bfe0f20f1c5218d6b554aa0012de34e504`
(or `$HOME/.cache/...` when XDG_CACHE_HOME is unset). Use `--output` for another
directory **outside the checkout**, then set `MANDOL_AML_PROTOCOL_SOURCE_DIR` to
that directory. You may also obtain the two
[pinned upstream scripts](ATTRIBUTION.md#aml-evaluation-sources) yourself, naming
them `locomo.py` and `longmemeval.py` in that directory.

Downloads retain their upstream terms. The loader checks the original complete
file SHA-256 before extracting the prompt literals and executing only five
named pure renderer/parser functions. It never imports upstream API configuration,
HTTP clients, or the upstream main routine. Missing or modified sources fail
before model calls; there is no automatic network download or legacy fallback.

The prompt contract, source revision, renderer hashes, and judge parser are
recorded in the local evaluation manifest. Answer/judge model parity with the
hosted leaderboard is outside this local protocol. Obtain authorized datasets
and configure separate answer/judge services for actual QA experiments.

## Interpreting results

Frozen integrity checks identify selected method artifacts; they do not establish
model tensor equality or measure retrieval quality. External aliases alone do
not identify a provider's tokenizer, weights, or numerical implementation.

Upstream Mandol paper results, local QA measurements, synthetic regression
results, and official AML leaderboard scores are distinct. Official results must
come from the platform's evaluated system version and contract. See the
[AML documentation](https://agentmemories.ai/docs) for participation
and evaluation procedures.
