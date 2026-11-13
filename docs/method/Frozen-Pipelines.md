# Competition method artifacts

The current `mm-source-preserving-add-20261113-v3` recipe combines v4 embedding, Qwen3.7 reranking, Textual
observation-history output, Coding native-Sparse/Dense RRF, and Multimodal
staged OCR with source-grounded evidence review. The [workflow](Current-Workflow.md)
describes how these components are used by each track.

`configs/profiles/submission/*.json` record the selected models, retrieval
policies, evidence formats and operating limits. Each descriptor includes a
SHA-256 digest of its canonical profile.

`configs/frozen-pipelines.json` binds the three selections to the source and
resource files used by the competition implementation. Check the artifacts with:

```bash
python deployment/cluster/frozen.py
```

The public launcher runs this check before starting an API. A matching manifest
keeps the documented configuration and implementation in sync.

Historical adapters retain their corresponding profile resources. When running
an earlier recipe, use its matching model identities, schema and collections.
New deployments use separate storage roots and collections for the three tracks.

The earlier public M0 baseline remains available at public commit `1be0776`.
Current descriptors identify the new recipe; they do not attest that every
historical evaluation request used it. External model outputs and a maintained
historical database need not match a fresh replay.
