# Security

## Credentials and network access

Keep real credentials out of source files, templates, Git, URLs, metric labels,
and request logs. Supply the Memory System Key through environment variables or
supported secret-file mounts. Provider clients use the credential environment
variables documented in [Local deployment](docs/deployment/Local-Reproduction.md).

Configure authentication before exposing Add/Search. The unified factories
reject an empty Memory System Key unless `MANDOL_ALLOW_UNAUTHENTICATED_DEV=1` is
explicitly enabled. Keep that option limited to isolated local development.
Use HTTPS for remote Memory API access. Keep Qdrant, inference, `/livez`, `/readyz`,
and `/metrics` on a trusted network, with an appropriate access policy.
Disable API access logs when they would disclose memory or image content.

## Storage and testing

Run one API worker per track with distinct, non-nested runtime roots. Each
Qdrant collection must match the expected track, canonical UUID, profile, and
projection identity. Investigate a mismatch instead of clearing or adopting an
existing collection. API processes do not own the external Qdrant lifecycle.

Protect SQLite databases, MediaStore, backups, and generated evaluation outputs
as user data. Back up consistent canonical state and media together. Use fresh
disposable storage for destructive, recovery, and integration tests; never use
a production database. Fetch external model and evaluator artifacts from their
documented sources and verify pinned identities before use.

## Reporting a vulnerability

Use GitHub private vulnerability reporting if it is available for this repository.
Otherwise, open a minimal issue asking the maintainer for a private reporting
channel, without credentials, personal data, exploit payloads, or live endpoints.
Include the affected component/version and a sanitized description once a
private channel is established.

If a credential is exposed, contact its owner so they can revoke or rotate it
and assess affected services. Do not publish the value in an issue or pull request.
