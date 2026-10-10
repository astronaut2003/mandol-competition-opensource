# Memory API reference

All three tracks implement `POST /add`, `POST /search`, and `GET /health`.
Select the track through its API origin: by default Textual uses port **18600**,
Coding **18601**, and Multimodal **18602**. Search retrieves evidence; it does not
generate a final answer.

Factories are `mandol_competition.runtime.apps:textual`, `:code`, and
`:multimodal`. Use [the launcher](../scripts/start_api.py) with one worker per
track and a [unified deployment configuration](deployment/Local-Reproduction.md).

## Authentication

Configure `MANDOL_MEMORY_SYSTEM_KEY` or its supported `_FILE` mount. A track can
use its own `MANDOL_<TRACK>_MEMORY_SYSTEM_KEY` override. Add and Search accept:

- `Authorization: Bearer <your-key>`
- `Authorization: Token <your-key>`
- `X-Api-Key: <your-key>`

These header examples are placeholders. Do not put credentials in URLs.
Missing or incorrect credentials return HTTP 401. The unified factories reject
an empty configured key unless `MANDOL_ALLOW_UNAUTHENTICATED_DEV=1` is explicitly
enabled for isolated development. See [Security](../SECURITY.md).

## Add memory

```json
{
  "request_id": "example-1",
  "user_id": "example-user",
  "session_id": "session-1",
  "messages": [
    {"role": "user", "content": "I prefer coffee without sugar.", "timestamp": 1791504000000}
  ]
}
```

The three identifiers and `messages` are required. `timestamp` is an optional
Unix time in milliseconds. Textual and Coding message content is a string.
Multimodal accepts strings or ordered text/image parts:

```json
{
  "request_id": "image-example-1",
  "user_id": "example-user",
  "session_id": "session-1",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "The photo shows my coffee cup."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_ENCODED_IMAGE"}}
  ]}]
}
```

Replace the image placeholder with a valid JPEG, PNG, or WebP data URI. Remote
image URLs, malformed Base64, MIME mismatches, and unsupported image types are
rejected. The frozen service allows at most **10 MiB** per decoded image,
**30 MiB** of image bytes per Add, **10 images**, and **40 million decoded pixels**.
Image responses use a **28 MiB** budget. Original image bytes are held in
MediaStore rather than embedding Base64 in SQLite or Qdrant.

A successful Add returns HTTP 200 and echoes the identifiers exactly:

```json
{"success": true, "request_id": "example-1", "user_id": "example-user", "session_id": "session-1"}
```

T0/M0 acknowledge after their required vector publication completes. C0 can
acknowledge durable canonical/lexical data when optional Dense is unavailable.
Repeating the same request ID and canonical payload replays one logical result;
changing its payload conflicts with HTTP 409. A failed/unknown inference or
publication outcome is tracked by the request ledger rather than treated as a
successful write.

## Search memory

```json
{
  "query": "What is my coffee preference?",
  "user_id": "example-user",
  "top_k": 100,
  "options": ["without sugar", "sweet"]
}
```

`query` and `user_id` select the query and user scope; `top_k` controls the
requested maximum. `options` is an optional top-level array of choices supplied
with the actual query. Do not insert a gold answer into retrieval inputs.
Multimodal queries may also contain ordered text/image parts. Search spans the
user's sessions; `session_id` is not a Search filter.

```json
{"data": [{"id": "stable-source-id", "content": "I prefer coffee without sugar.", "score": 0.8}]}
```

`data` is always present and may be `[]`. Every result has `id` and `content`;
scores, when present, are finite and higher-is-better. The response returns at
most `top_k`, further limited by the track's context, card, and media budgets.
Multimodal results preserve ordered source parts and restore permitted images
from MediaStore. Matching user IDs on different tracks remain physically isolated.

## Health and operational routes

`GET /health` is unauthenticated and returns `{"status": "ok"}` without exposing
model or storage configuration. `/livez` checks process liveness; `/readyz` checks the
track's required dependencies; `/metrics` exposes operational metrics. Keep the
operational routes on a trusted network. Readiness can fail even when `/health`
succeeds, particularly when inference or vector publication is unavailable.

The upstream [AML API guide](https://agentmemoryleaderboard.ai/api-guide) describes
the platform contract. [Frozen pipelines](method/Frozen-Pipelines.md) documents
the retrieval and acknowledgement policies implemented here.
