# Frozen competition pipelines

T0, C0, and M0 define the project's fixed competition retrieval and publication
policies. This document explains their representations, model calls, ranking,
and persistence semantics. Deployment settings choose storage and endpoint
locations without selecting a different retrieval method.

## T0: Textual

Ordered conversation messages become canonical source memories, retaining source
attribution. SQLite lexical BM25 uses regex tokenization, k1=1.5 and b=0.75.
BM25, 1024-dimensional v4 Dense, and paired native Sparse each use candidate depth
100. RRF k=60 fuses the lanes before graph/semantic context expansion and reranking.
Querit uses complete original-order B1 pairs and final CLS token 151665, with no
pair truncation or probability renormalization; relevance is `probs[1]-probs[0]`.

HLM and MMR are off. The return policy uses a final window of 10 and 24,000
context characters, further capped by the request's `top_k`.

## C0: Coding

Raw content, whitespace, source hashes, session, and request identity are retained.
Engineering episode construction, raw-rank/evidence aggregation, and EvidencePacker
build source-backed evidence cards. Add succeeds after canonical and lexical
durability.

Dense is optional: a failed channel is recorded as degraded, and a new complete
projection frontier must be ready before hybrid retrieval resumes. C0 uses v4
Dense with the Dot metric. It has no Sparse, neural reranker, or generative runtime
stage. Output is capped at 8 cards, 16,000 total characters, and 1,200 per source.
Repository parsing/preparation is separate from the online AML memory path.

## M0: Multimodal

Strict validation and a per-user content-addressed MediaStore preserve text/image
order. GPT-4o-mini performs `multimodal_sgm_v1`, schema 1 extraction.
`mm-representation-v1` builds DM + SGM; `mm-query-v1` defines the options policy.

BM25, v4 Dense, and native Sparse use candidate depth 100 and RRF k=60. BGE uses
its pinned BF16 recipe, maximum pair length 4096, right truncation, and microbatch
1; sigmoid relevance scores are higher-is-better. Selectors and alternative rank
blends are off. Media references, durable preparation replay, unknown VLM dispatch
outcome handling, image restoration, and purge hooks are part of the implementation.

## Canonical state and vector publication

Track roots and canonical UUIDs are independent. Resource/profile/schema and
projection identities must match; incompatible bindings fail closed. SQLite holds
canonical memory state. Qdrant is a rebuildable versioned projection containing
compact metadata rather than raw memories or images.

| Track | Publication policy | Acknowledgement |
|---|---|---|
| T0 | `textual-strict-vector-v1` | Required projection; encode before prepare |
| C0 | `code-canonical-lexical-optional-dense-v1` | Durable canonical/lexical state; optional Dense |
| M0 | `multimodal-post-prepare-strict-v1` | Required projection after prepare |

The same request ID and canonical body replay one logical result; changed content
conflicts. Completed preparations and published vectors survive ordinary API
restarts. Pending and unknown outcomes use the reconciliation protocol.
Search is user-scoped across sessions. Physical track isolation also separates
identical user IDs in different tracks.

## Artifact integrity

```bash
python3 deployment/cluster/frozen.py
```

The checker validates three selection hashes and seven key source/profile files
against [frozen-pipelines.json](../../configs/frozen-pipelines.json).
`configs/profiles/submission/*.json` are immutable historical selection envelopes;
they are integrity references rather than executable API configuration. Use the
[deployment TOML](../../configs/deployment/frozen-bf16.example.toml) and
[serving bindings](../../configs/deployment/serving-bindings.example.json) for
runtime locations. Integrity hashes identify method artifacts, while actual
model behavior and quality require [real-model evaluation](../Reproduction.md).
