# Contributing

Follow the [Quick Start](README.md#installation) and
[test guide](docs/Reproduction.md). Use the locked Python 3.12 environment and
disposable storage for integration tests.

Keep persistence and publication mechanisms in `mandol_substrate`, and memory
representation/retrieval policy in the corresponding track adapter. The substrate
must remain independent of API handlers, graph/parsing logic, media, and VLM code.

Preserve source attribution, request replay, recovery, and user/track isolation.
Changes to frozen prompts, model recipes, candidate limits, ranking, output
windows, or publication policies need a separately versioned profile and tests
that show the resulting behavior. Keep the existing T0/C0/M0 artifacts reproducible.

Describe the concrete change, its purpose, and relevant validation in pull
requests. Include only synthetic or permitted test data. Keep secrets, personal
data, runtime databases, logs, and model weights out of contributions; see
[Security](SECURITY.md). Retain third-party attribution and licensing conditions.

Report local test and quality results with their environment and evaluation
protocol. Local measurements are separate from official AML scores.
