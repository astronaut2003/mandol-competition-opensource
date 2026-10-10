"""Structured vision extraction via a configurable OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import math
import os
import time
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.inference.errors import InferenceIdentityError
from types import SimpleNamespace

from ..config import MultimodalConfig
from ..profiling import stage
from .base import Extraction, ExtractionError, parse_extraction
from .prompt import build_extraction_prompt

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status_code: int | None) -> bool:
    return status_code is None or status_code in RETRYABLE_STATUS_CODES


class VisionRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


@dataclass(frozen=True)
class VisionResponse:
    payload: dict[str, Any]
    model: str
    usage: dict[str, Any] | None
    latency_ms: float


def _extract_json_object(content: object) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ExtractionError("model returned a non-string response")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for position, character in enumerate(content):
            if character == "{":
                try:
                    payload, _ = decoder.raw_decode(content, position)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
        raise ExtractionError("model response contained no valid JSON object") from None

    if not isinstance(payload, dict):
        raise ExtractionError("model returned JSON that is not an object")
    return payload


class OpenAICompatVisionClient:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float,
        request_model: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.model = model
        self.request_model = request_model or model
        self._client = client_factory() if client_factory is not None else None
        self._http = None if client_factory is not None else HTTPVLMClient(base_url, self.request_model, api_key=api_key, timeout=timeout)

    @stage('vlm_http')
    def complete(self, *, prompt: str, image_data_uri: str) -> VisionResponse:
        blocks = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
        started = time.monotonic()
        try:
            if self._http is not None:
                value = self._http.complete([{ "role": "user", "content": blocks}], request_id=None, temperature=0.0)
                response = json.loads(json.dumps(value), object_hook=lambda x: SimpleNamespace(**x))
            else:
                response = self._client.chat.completions.create(
                    model=self.request_model, messages=[{"role": "user", "content": blocks}], temperature=0.0)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                status_code = None
            raise VisionRequestError(
                "vision request unavailable",
                retryable=is_retryable_status(status_code) and not getattr(exc, "unknown_outcome", False)
                    and not isinstance(exc, InferenceIdentityError), status_code=status_code,
                retry_after=getattr(exc, 'retry_after', None),
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise ExtractionError("model returned no choices")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        payload = _extract_json_object(content)
        raw_usage = getattr(response, "usage", None)
        usage = None if raw_usage is None else {
            field: value
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
            if (value := getattr(raw_usage, field, None)) is not None
        }
        return VisionResponse(payload, self.model, usage, (time.monotonic() - started) * 1000)


    def close(self):
        if self._http is not None:
            self._http.close()


def build_vision_client(config: MultimodalConfig) -> OpenAICompatVisionClient:
    if not config.vlm_base_url:
        raise ExtractionError("vlm_base_url must be configured for an OpenAI-compatible VLM")
    api_key = os.getenv(config.vlm_api_key_env)
    if not api_key:
        raise ExtractionError(f"set the API key environment variable {config.vlm_api_key_env}")
    return OpenAICompatVisionClient(
        model=config.vlm_model, api_key=api_key,
        request_model=config.vlm_request_model,
        base_url=config.vlm_base_url, timeout=config.vlm_timeout_seconds,
    )


class OpenAICompatExtractor:
    def __init__(
        self, *, model_name: str, config: MultimodalConfig, client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.config = config
        self._client = client
        # One shared extractor can serve distinct users on separate deadline
        # workers. Usage attribution must not drain another request's response.
        self._observation_local = threading.local()

    def close(self):
        if hasattr(self._client, "close"):
            self._client.close()
        self._observation_local.values = []

    def take_observations(self) -> list[tuple[str, dict | None, float]]:
        observations = getattr(self._observation_local, 'values', [])
        self._observation_local.values = []
        return observations

    def extract(
        self, *, image_data_uri: str, context_text: str,
        image_position: int, image_count: int,
    ) -> Extraction:
        prompt = build_extraction_prompt(context_text, image_position, image_count, version=self.config.prompt_version)
        if self._client is None:
            self._client = build_vision_client(self.config)
        attempts = max(1, self.config.vlm_max_attempts)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            retry_after = None
            try:
                response = self._client.complete(prompt=prompt, image_data_uri=image_data_uri)
                extraction = parse_extraction(
                    response.payload, model=self.model_name,
                    prompt_version=self.config.prompt_version,
                    schema_version=self.config.schema_version,
                )
            except VisionRequestError as exc:
                last_error = exc
                if not exc.retryable:
                    break
                retry_after = exc.retry_after
            except ExtractionError as exc:
                last_error = exc
            else:
                if not hasattr(self._observation_local, 'values'):
                    self._observation_local.values = []
                self._observation_local.values.append((image_data_uri, response.usage, response.latency_ms))
                return extraction

            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 4)
                if retry_after is not None:
                    if not math.isfinite(retry_after) or retry_after > min(60., self.config.vlm_timeout_seconds):
                        # Fail within the bounded budget instead of violating a
                        # provider's longer cooldown or sleeping without a limit.
                        break
                    delay = max(delay, retry_after)
                time.sleep(delay)

        raise ExtractionError(
            f"vision extraction failed after {attempt} attempt(s): {type(last_error).__name__}"
        ) from last_error
