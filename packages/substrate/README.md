# mandol-substrate

The shared storage and retrieval substrate used by Mandol Competition's Textual,
Coding, and Multimodal adapters. It implements canonical SQLite state,
lifecycle/request ledgers, exact BM25, publication frontiers, immutable Qdrant
projections, verification, reconciliation, and garbage collection.

The package has no imports from track graph/parser code, media handlers, VLM
clients, or API handlers. Encoders and track-specific views are injected.

## Installation and build

From the repository root:

```bash
uv sync --frozen --python 3.12.13 --extra all --extra dev
uv build packages/substrate --wheel --out-dir dist
```

The uv workspace installs this local package. Import it as `mandol_substrate`;
its distribution name is `mandol-substrate`. Runtime dependencies include
qdrant-client 1.19.0, HTTPX, and NumPy; the documented deployment uses an external
Qdrant 1.19.1 server.

## Track policies

T0/M0 use native-v4 Dense/Sparse projection mechanisms with separate physical
resources and strict vector publication. C0 acknowledges canonical/lexical
durability and uses optional complete-frontier Dense. Historical `splade` DTO
names do not require a SPLADE model service.

Compatibility shims preserve module/class identity for existing adapter imports.
[upstream.json](src/mandol_substrate/upstream.json) retains extraction provenance.
The package is distributed under [Apache-2.0](LICENSE), with original notices
preserved in [NOTICE](NOTICE). See [Attribution](../../docs/ATTRIBUTION.md),
[Frozen pipelines](../../docs/method/Frozen-Pipelines.md), and
[Local deployment](../../docs/deployment/Local-Reproduction.md).
