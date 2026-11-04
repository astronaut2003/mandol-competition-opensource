# Competition runtime selections

`textual.json`, `code.json` and `multimodal.json` identify the T0, C0 and M0
competition configurations. Each file contains the selected runtime parameters
and the SHA-256 digest of its canonical profile.

The deployment TOML supplies service endpoints and storage locations. The
integrity checker verifies these selections alongside the source and resources.
See the [competition workflow](../../../docs/method/Current-Workflow.md).
