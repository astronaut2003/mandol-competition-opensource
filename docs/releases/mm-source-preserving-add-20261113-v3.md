# Visual recovery and source-preserving Add

Release recipe: `mm-source-preserving-add-20261113-v3`.
Earlier public baseline: [`1be0776`](https://github.com/astronaut2003/mandol-competition-opensource/tree/1be077658277b020e528db42242cb7a146b9cc19).

## Changes included

- Search/selector follow-up: normalize valid duplicate or oversized selections,
  retain baseline candidates when optional evidence review is malformed, and
  make full database audits opt-in without removing binding/frontier/hit checks.
- Operational follow-up: disable periodic paid reranker probes and recover on
  actual requests. This change followed the original formal evaluation.
- Visual Add: bound looping output; recover scene fields in pages; retain input-
  bound paid receipts and completed checkpoints; reuse a proven concise OCR
  prompt for later crops instead of spending an additional retry.
- When confirmed content failures exhaust recovery without a trustworthy scene,
  preserve durable original image/text and validated partial fields under
  `source_only`. Existing trusted-scene / incomplete-OCR handling remains.
- Publish per-image quality markers atomically and provide a read-only aggregate
  quality command. Replayed requests do not duplicate publication or counts.

## Effective selection

The public launcher uses `gpt-4o-mini`, prompt `multimodal_sgm_v3_ocr_lines`,
schema `1`, 1,024 output tokens per call, at most 16,384 cumulative output tokens
and 20 paid calls per image ledger. Field recovery has its own page/call caps
within that budget; retries do not reset it. The extraction library implements
the more detailed task-specific limits. Invalid credentials/input, unresolved
network outcomes and persistence/publication failures remain errors.

`MANDOL_MM_VISION_SOURCE_FALLBACK=1` and
`MANDOL_MM_FULL_DATABASE_AUDIT=0` are explicit in the public launcher and
submission descriptor. Setting source fallback to `0` retains strict failure
behavior through a separately recorded factory configuration.

For a fresh Multimodal store, the public launcher requires explicit
`--initialize` to create and verify the initial projection with a full audit.
This flag refuses existing state. Subsequent starts omit it and use the selected
fast path. Local/cluster adapters now propagate the same recovery flags, and the
launcher explicitly chooses the OpenAI-compatible visual backend. These are
public entrypoint adaptations; they do not change retrieval math.

Dense/native-Sparse models, lexical recall, RRF/rerank formulas, candidate pools,
evidence budgets, user isolation and Add replay/persistence rules are retained.
The Add failure policy changes which failed-extraction images remain visible;
it is part of this new recipe. A source-only image without usable text or visual
fields can still have weak semantic recall. Search returns original evidence
and extraction status; it does not inject answer or abstention instructions.

## Reproduction and data boundary

Use the locked dependencies, public startup script, fresh owned SQLite/media/
Qdrant resources and your own provider credentials. Synthetic tests cover recovery,
idempotence, original-source preservation, isolation and HTTP contracts. They do
not establish a benchmark score or identical provider responses.

This export contains no production databases, real evaluation images/questions/
answers, request captures, credentials or private Git history. Historical stores
may include interrupted runs and maintenance; they are not equivalent to a fresh
replay. Older published commits remain available for historical review.
