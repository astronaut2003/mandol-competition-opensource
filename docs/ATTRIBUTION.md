# Attribution and licenses

## Mandol and the competition implementation

[Mandol](https://github.com/AgentCombo/Mandol) is the original agent memory system,
distributed under Apache-2.0 by the Mandol Team. Mandol Competition builds on its
Textual, Coding, and Multimodal implementations and adds a unified durable
substrate, isolated AML APIs, external inference orchestration, and recovery.

Mandol Competition's unified additions are distributed under Apache-2.0 along
with the inherited implementation. [LICENSE](../LICENSE) and [NOTICE](../NOTICE)
preserve the original license and copyright. The substrate and HTTP bridge
packages retain their own LICENSE and NOTICE files. The substrate's
[upstream.json](../packages/substrate/src/mandol_substrate/upstream.json) records
file-level extraction provenance; historical graph, BM25, and context-render test
oracles retain their Mandol source attribution.

Original technical report:
Yuhan Zhang, Zhiyuan Guo, Ziheng Zeng, Wei Wang, Wentao Wu, and Lijie Xu,
[Mandol: An Agglomerative Agent Memory System for Long-Term Conversations](https://arxiv.org/abs/2606.29778),
arXiv:2606.29778 (2026). The upstream paper artifact and this competition
implementation are distinct versions. Results should identify the version and
evaluation protocol used.

## AML evaluation sources

Optional local QA uses the public evaluation protocol from
[AML-memory/agent-memory-leaderboard](https://github.com/AML-memory/agent-memory-leaderboard)
at Commit `1b8142bfe0f20f1c5218d6b554aa0012de34e504`:

- [LoCoMo-Refined pipeline](https://github.com/AML-memory/agent-memory-leaderboard/blob/1b8142bfe0f20f1c5218d6b554aa0012de34e504/data/locomo-refined/pipeline.py)
- [LongMemEval pipeline](https://github.com/AML-memory/agent-memory-leaderboard/blob/1b8142bfe0f20f1c5218d6b554aa0012de34e504/data/longmemeval-s/pipeline.py)

These sources are retrieved separately into the user's cache and verified
against pinned SHA-256 values. They are not bundled or covered by this project's
Apache-2.0 license. Consult the upstream terms before use or redistribution.
[Reproduction](Reproduction.md#optional-local-qa) describes retrieval and the
limited pure-function execution used by the local evaluator. Benchmark datasets
must also be obtained separately under their own terms.

## Models and services

The competition configuration uses these model services:

| Component | Model | Terms |
|---|---|---|
| Embedding | `text-embedding-v4` | Provider service terms |
| Reranking | `qwen3.7-text-rerank` | Provider service terms |
| Visual extraction and evidence selection | `gpt-4o-mini` | Configured service's terms |

### Historical model adapters

| Component | Reference | Terms and distribution scope |
|---|---|---|
| Querit adapter | [Querit/Querit-4B](https://huggingface.co/Querit/Querit-4B), revision `e557bb5ce6555f16b33dbb7d2ca62e8d27c14c13` | Obtain permitted artifacts for the pinned revision; weights are not bundled. Check the applicable revision's terms. |
| BGE adapter | [BAAI/bge-reranker-v2-m3 pinned card](https://huggingface.co/BAAI/bge-reranker-v2-m3/blob/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e/README.md) | Model card declares Apache-2.0; weights are not bundled. |
| Embedding | Provider `text-embedding-v4` | Provider service terms; a model alias does not grant rights to provider weights. |
| VLM | Logical `gpt-4o-mini-closeai`, wire model `gpt-4o-mini` | External OpenAI-compatible service terms; no provider model files are bundled. |

The Querit adapter references the
[MTEB pair specification](https://github.com/embeddings-benchmark/mteb/blob/ed47a25455157433b458db2625e72960468597ab/mteb/models/model_implementations/querit_models.py).
Model-serving contracts and revisions are documented in
[Inference deployment](deployment/Inference.md). Model licenses are separate from
the license of the client and adapter code in this repository.

## Optional reference adapters

Jina and Mixedbread reference adapters support research experiments and are not
used by frozen T0/C0/M0:

- [Jina reranker v3.5 pinned model card](https://huggingface.co/jinaai/jina-reranker-v3.5/blob/e8a93f33f0b22108f8c2364f8484ce3422552fbc/README.md)
  declares **CC-BY-NC-4.0**. The project's Apache-2.0 license does not override
  these noncommercial model restrictions.
- [Mixedbread reranker pinned model card](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2/blob/ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6/README.md)
  declares Apache-2.0.

Only adapter code is included, not model tensors or downloaded remote model
implementations. The standard application installation does not enable these
adapters or install their GPU environments.

## Dependencies

Qdrant, FastAPI, Pydantic, HTTPX, NumPy, rustworkx, NetworkX, Tree-sitter,
tree-sitter-language-pack, Pillow, the OpenAI SDK, uv, and vLLM retain their own
licenses. See the [dependency inventory](Dependency-Licenses.md). Preserve their
notices when separately redistributing binary dependencies.
