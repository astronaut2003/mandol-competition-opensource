# Current inference contracts

The current public startup recipe is `configs/deployment/current-cloud.example.toml`.
API/bridge processes are CPU-only; cloud services perform embedding, reranking
and visual extraction. Supply your own authorized accounts and endpoints.

| Capability | Current selection | Contract |
|---|---|---|
| Dense and native Sparse | `text-embedding-v4` | Paired native query/document encodes; 1,024-dimensional Dense plus finite native Sparse weights. |
| Reranking | `qwen3.7-text-rerank` | Native DashScope endpoint, complete candidate scoring, scores returned in original document order. |
| Visual extraction | `gpt-4o-mini` wire model | Authorized OpenAI-compatible image API, prompt `multimodal_sgm_v3_ocr_lines`, canonical schema `1`. |

Cloud model aliases do not attest a fixed weight revision, tokenizer version,
pooling implementation, or precision. Runtime descriptors disclose these as
provider managed/undisclosed rather than inventing a weight attestation.

## Embedding budget

Bundled Qwen3-Embedding tokenizer assets provide deterministic local budgeting.
Overlong queries are split without dropping the original text, with at most
8,000 content tokens per view below the provider's 8,192-token bound. The
implemented weighted Dense/Sparse fusion approximates an overlong query;
it is not mathematically identical to encoding the entire original input in
one model call. Short inputs retain the normal path. Original queries remain
available for lexical/graph selection and reranking.

The assets include public-source provenance and licenses. No runtime tokenizer
download is required. The compatibility choice does not prove that every input
has identical tokenization inside the provider.

## Reranking budget and fusion

The Qwen3.5 compatibility tokenizer and local plan guard 29,800 query tokens,
49,800 query characters, 32,568 pair tokens, and 119,800 aggregate tokens. The
aggregate counts the query separately for every document. Candidate count is
bounded by the provider's 500-document request capacity.

Short requests retain native relevance scores. Multiple views keep candidate
identities/order and use tie-aware normalized RRF60. Numeric provider relevance
scores from independent requests are not averaged. The fused score is an
ordering score, not calibrated answer confidence. Incomplete or failed views
do not silently become a full successful ranking. Known length rejections can
trigger at most two replans; other errors retain their typed failure semantics.

## Visual extraction and OCR

The current VLM prompt emits typed visual fields and literal `ocr_lines`.
Staged recovery separates bounded scene output, OCR localization/review and
full-coverage image bands. Validation and checkpoints bind each paid stage to
the original image/context/model/prompt/schema. Replay can resume validated
work without a second canonical memory or duplicate publication.

The model capacity is 16,384 output tokens; current per-call configured output
is 1,024. Stage/crop/recovery and deadline bounds are independent. Invalid JSON,
truncation, uncertain paid outcomes and repeated-text signals have distinct
handling; not every failure is safe to retry. When bounded OCR recovery is
exhausted with a validated scene, Add retains the original source, scene and
validated text fragments with explicit incomplete-OCR status. The receipt is
bound to the request ledger, and the incomplete result is excluded from the
success cache. Replay reuses that ledger without new provider calls.
Exhaustion without a validated scene, invalid bindings and failed checkpoint
persistence remain errors.

The scene, OCR and review stages use the configured VLM service. Their
prompts, validation and recovery behavior are defined in the VLM modules.

## Historical compatibility recipes

The source retains Querit-4B and BGE adapters and historical BF16 binding
examples. They have different tokenizers, score semantics and model identities.
They are not the current default Qwen recipe. Select and validate a complete
historical configuration explicitly if reproducing that earlier method; do not
mix old serving bindings into a new cloud runtime or reuse its vector identity.

See [local reproduction](Local-Reproduction.md) for startup, shared admission
configuration, credentials, and isolated storage.
