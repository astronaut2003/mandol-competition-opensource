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
from mandol_competition.inference.errors import InferenceIdentityError, InferenceDeadlineExceeded, InferenceRateLimited, InferenceUnavailable
from types import SimpleNamespace

from ..config import MultimodalConfig
from ..profiling import stage
from .base import Extraction, ExtractionError, parse_extraction, has_visual_evidence
from .prompt import build_extraction_prompt
from .output_guard import (OUTPUT_POLICY, VisionOutputError, VisionOutputTruncated,
                           extraction_fingerprint, validate_output_budgets, VISION_REQUEST_OUTPUT_LIMIT)

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_status(status_code: int | None) -> bool:
    return status_code is None or status_code in RETRYABLE_STATUS_CODES


class VisionRequestError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None,
                 retry_after: float | None = None, retry_managed: bool = False,
                 unknown_outcome: bool = False, paid_output_failure: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after
        self.retry_managed = retry_managed
        self.unknown_outcome = unknown_outcome
        self.paid_output_failure = paid_output_failure


class VisionExtractionUnavailable(InferenceUnavailable):
    """No successful visual memory can be acknowledged for this preparation."""
    def __init__(self, *, retry_safe: bool, unknown_outcome=False, status_code=None,
                 error_class="vision_extraction_unavailable"):
        self.retry_safe = retry_safe
        super().__init__(error_class, status_code=status_code,
                         unknown_outcome=unknown_outcome)


class VisionRecoveryExhausted(VisionExtractionUnavailable):
    """Malformed recovery metadata cannot safely authorize a provider call."""
    def __init__(self):
        super().__init__(retry_safe=False, unknown_outcome=True,
                         error_class="vision_recovery_exhausted")


@dataclass(frozen=True)
class VisionResponse:
    payload: dict[str, Any]
    model: str
    usage: dict[str, Any] | None
    latency_ms: float
    output_validation: dict[str, Any] | None = None


def _extract_json_object(content: object) -> dict[str, Any]:
    if not isinstance(content, str):
        raise ExtractionError("model returned a non-string response")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ExtractionError('model returned duplicate JSON fields')
            result[key] = value
        return result

    def invalid_constant(_):
        raise ExtractionError('model returned non-finite JSON values')

    try:
        payload = json.loads(content, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except json.JSONDecodeError:
        raise ExtractionError('model response must be one complete JSON object') from None

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
        output_tokens: int = VISION_REQUEST_OUTPUT_LIMIT,
        max_output_tokens: int = VISION_REQUEST_OUTPUT_LIMIT,
        wire_format: str = 'legacy-json-v1',
    ) -> None:
        validate_output_budgets(output_tokens, max_output_tokens)
        self.output_tokens, self.max_output_tokens = output_tokens, max_output_tokens
        self.timeout_seconds = timeout
        from .ocr_lines import FORMAT_POLICY
        if wire_format not in {'legacy-json-v1', FORMAT_POLICY}:
            raise ValueError('unknown visual output wire format')
        self.wire_format = wire_format
        from .capture import VisionWireCapture
        self._capture = VisionWireCapture.from_environment()
        self.model = model
        self.request_model = request_model or model
        self._client = client_factory() if client_factory is not None else None
        self._http = None if client_factory is not None else HTTPVLMClient(base_url, self.request_model, api_key=api_key, timeout=timeout)

    def _dispatch(self, messages, options):
        try:
            if self._http is not None:
                diagnostic = {'diagnostic': self._capture.observe} if self._capture is not None else {}
                value = self._http.complete(messages, request_id=None,
                    **diagnostic, **options)
                response = json.loads(json.dumps(value), object_hook=lambda x: SimpleNamespace(**x))
            else:
                response = self._client.chat.completions.create(
                    model=self.request_model, messages=messages, **options)
        except InferenceDeadlineExceeded:
            raise  # No provider dispatch occurred; Add may safely retry later.
        except Exception as exc:
            retry_managed = self._http is not None and self._http.http.reliability is not None
            status_code = getattr(exc, "status_code", None)
            if not isinstance(status_code, int):
                status_code = None
            raise VisionRequestError(
                "vision request unavailable",
                retryable=is_retryable_status(status_code) and not getattr(exc, "unknown_outcome", False)
                    and not isinstance(exc, InferenceIdentityError) and not retry_managed, status_code=status_code,
                retry_after=getattr(exc, 'retry_after', None),
                retry_managed=retry_managed,
                unknown_outcome=getattr(exc, "unknown_outcome", status_code is None),
                paid_output_failure=getattr(exc, 'vision_paid_output_failure', False),
            ) from exc

        return response

    def _dispatch_stream(self, messages, options, observer):
        """Opt-in bounded SSE using the same production admission/transport."""
        if self._http is None:
            response = self._dispatch(messages, options)
            _, message, _ = self._response_metadata(response, options['max_completion_tokens'])
            observer(getattr(message, 'content', '') or '')
            return response
        from .ocr_recovery import OCRRecoveryError
        try:
            diagnostic = {'diagnostic': self._capture.observe} if self._capture is not None else {}
            value = self._http.complete_stream(messages, request_id=None, on_content=observer,
                                               **diagnostic, **options)
            return json.loads(json.dumps(value), object_hook=lambda x: SimpleNamespace(**x))
        except (OCRRecoveryError, InferenceDeadlineExceeded):
            raise
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            status = status if type(status) is int else None
            managed = self._http.http.reliability is not None
            raise VisionRequestError('vision stream unavailable',
                retryable=is_retryable_status(status) and not getattr(exc, 'unknown_outcome', False) and not managed,
                status_code=status, retry_after=getattr(exc, 'retry_after', None), retry_managed=managed,
                unknown_outcome=getattr(exc, 'unknown_outcome', status is None),
                paid_output_failure=getattr(exc, 'vision_paid_output_failure', False)) from exc

    @staticmethod
    def _response_metadata(response, budget):
        raw_usage = getattr(response, 'usage', None)
        usage = None if raw_usage is None else {field:value
            for field in ('prompt_tokens','completion_tokens','total_tokens')
            if type(value := getattr(raw_usage,field,None)) is int and 0 <= value <= 1000000}
        choices = getattr(response,'choices',None)
        choice = choices[0] if choices and len(choices) == 1 else None
        reason = getattr(choice,'finish_reason',None)
        message = getattr(choice,'message',None)
        completion = usage.get('completion_tokens') if usage else None
        return usage, message, dict(finish_reason=reason,completion_tokens=completion,output_token_budget=budget)

    @stage('vlm_http')
    def complete(self, *, prompt: str, image_data_uri: str,
                 output_token_budget: int | None = None) -> VisionResponse:
        from .ocr_lines import FORMAT_POLICY
        if self.wire_format == FORMAT_POLICY:
            from mandol_competition.inference.http import deadline_budget
            with deadline_budget(self.timeout_seconds):
                return self._complete(prompt,image_data_uri,output_token_budget)
        return self._complete(prompt,image_data_uri,output_token_budget)

    def _complete(self, prompt, image_data_uri, output_token_budget):
        budget = self.output_tokens if output_token_budget is None else output_token_budget
        validate_output_budgets(budget,self.max_output_tokens)
        started = time.monotonic()
        blocks = [{'type':'text','text':prompt}, {'type':'image_url','image_url':{'url':image_data_uri}}]
        options = dict(temperature=0.,max_completion_tokens=budget,response_format={'type':'json_object'})
        messages = [{'role':'user','content':blocks}]
        from .ocr_lines import FORMAT_POLICY, SYSTEM, response_format
        if self.wire_format == FORMAT_POLICY:
            options['response_format'] = response_format()
            messages.insert(0,{'role':'system','content':SYSTEM})
        response = self._dispatch(messages,options)
        usage,message,metadata = self._response_metadata(response,budget)
        if self.wire_format == FORMAT_POLICY:
            return self._structured_result(message,usage,metadata,image_data_uri,started)

        reason,completion = metadata['finish_reason'],metadata['completion_tokens']
        if reason == 'length' or completion is not None and completion >= budget:
            raise VisionOutputTruncated(**metadata)
        if reason != 'stop':
            raise VisionOutputError('vision_output_finish_unverified', **metadata)
        if getattr(message, 'refusal', None):
            raise VisionOutputError('vision_output_refused', **metadata)
        try:
            payload = _extract_json_object(getattr(message, 'content', None))
        except ExtractionError as exc:
            raise VisionOutputError('vision_output_invalid_json', retryable=True, **metadata) from exc
        except (ValueError, TypeError):
            raise VisionOutputError('vision_output_invalid_json', retryable=True, **metadata) from None
        receipt = dict(policy=OUTPUT_POLICY, finish_reason='stop', output_token_budget=budget,
                       completion_tokens=completion)
        return VisionResponse(payload, self.model, usage, (time.monotonic() - started) * 1000, receipt)

    def _structured_result(self, message, usage, metadata, image_data_uri, started):
        from .json_syntax import complete_object
        from .ocr_lines import (FORMAT_POLICY, canonical_payload, completed_scene_prefix,
            scene_payload, sequential_numbers_need_review, ocr_needs_review)
        reason, tokens = metadata['finish_reason'],metadata['completion_tokens']
        truncated = reason == 'length' or tokens is not None and tokens >= metadata['output_token_budget']
        if reason not in {'stop','length'}:
            raise VisionOutputError('vision_output_finish_unverified',**metadata)
        if getattr(message,'refusal',None):
            raise VisionOutputError('vision_output_refused',**metadata)
        content = getattr(message,'content',None)
        syntax_repair = None
        if truncated:
            try:
                scene = completed_scene_prefix(content)
            except (ValueError,TypeError):
                raise VisionOutputTruncated(**metadata) from None
            review = 'truncated_ocr'
        else:
            try:
                raw, syntax_repair = complete_object(content)
            except (ValueError,TypeError):
                try:
                    scene = completed_scene_prefix(content)
                except (ValueError,TypeError):
                    raise VisionOutputError('vision_output_invalid_json',retryable=True,**metadata) from None
                review = 'invalid_ocr'
            else:
                try:
                    payload = canonical_payload(raw)
                except (ValueError,TypeError):
                    try:
                        scene = scene_payload(raw)
                    except (ValueError,TypeError):
                        raise VisionOutputError('vision_output_invalid_json',retryable=True,**metadata) from None
                    review = 'invalid_ocr'
                else:
                    if not ocr_needs_review(raw['ocr_lines']):
                        receipt = dict(policy=OUTPUT_POLICY,**metadata,wire_format=FORMAT_POLICY)
                        if syntax_repair is not None:
                            receipt['json_syntax_repair'] = syntax_repair
                        return VisionResponse(payload,self.model,usage,(time.monotonic()-started)*1000,receipt)
                    scene = scene_payload(raw)
                    review = 'sequential_numbers' if sequential_numbers_need_review(raw['ocr_lines']) else 'repeated_lines'
        if syntax_repair is not None:
            metadata = dict(metadata,json_syntax_repair=syntax_repair)
        return self._repair_ocr(scene,usage,metadata,image_data_uri,started,review)

    def _repair_ocr(self, scene, usage, primary, image_data_uri, started, review):
        from .json_syntax import complete_object
        from .ocr_lines import (FORMAT_POLICY, OCR_SYSTEM, OCR_PROMPT, ocr_response_format,
            canonical_payload, ocr_needs_review, region_lines, OCRRepetitionError)
        from .output_guard import COMPOSITE_POLICY, MODEL_OUTPUT_LIMIT
        spent = primary['completion_tokens']
        reserve = spent if spent is not None else primary['output_token_budget']
        budget = min(self.max_output_tokens,MODEL_OUTPUT_LIMIT-reserve)
        if budget <= 0:
            if review == 'truncated_ocr':
                raise VisionOutputTruncated(**primary)
            raise VisionOutputError('vision_output_invalid_json',retryable=False,**primary)
        messages = [{'role':'system','content':OCR_SYSTEM}, {'role':'user','content':[
            {'type':'text','text':OCR_PROMPT},
            {'type':'image_url','image_url':{'url':image_data_uri}}]}]
        try:
            response = self._dispatch(messages,dict(temperature=0.,max_completion_tokens=budget,
                response_format=ocr_response_format()))
        except Exception as exc:
            # A rejected second stage cannot erase the first paid outcome or
            # authorize blind whole-image repeats inside this extraction.
            if isinstance(exc,VisionRequestError):
                exc.paid_output_failure = True
                exc.retryable = False
            else:
                exc.vision_paid_output_failure = True
            raise
        other, message, final = self._response_metadata(response,budget)
        if final['finish_reason'] == 'length' or final['completion_tokens'] is not None and final['completion_tokens'] >= budget:
            raise VisionOutputError('vision_output_truncated',retryable=False,**final)
        if final['finish_reason'] != 'stop':
            raise VisionOutputError('vision_output_finish_unverified',**final)
        if getattr(message,'refusal',None):
            raise VisionOutputError('vision_output_refused',**final)
        try:
            raw, syntax_repair = complete_object(getattr(message,'content',None))
            lines = {'ocr_lines':region_lines(raw)}
            payload = canonical_payload({**scene,**lines})
        except OCRRepetitionError:
            raise VisionOutputError('vision_output_ocr_repetition',retryable=False,**final) from None
        except (ValueError,TypeError):
            raise VisionOutputError('vision_output_invalid_json',retryable=False,**final) from None
        if ocr_needs_review(lines['ocr_lines']):
            # A second complete but still implausibly repetitive OCR result is
            # not proof of readable evidence. Keep a failed paid checkpoint;
            # never publish invented counters or silently discard actual text.
            raise VisionOutputError('vision_output_ocr_repetition',retryable=False,**final)
        combined = None if usage is None or other is None else {key:usage[key]+other[key]
            for key in ('prompt_tokens','completion_tokens','total_tokens') if key in usage and key in other}
        completion = combined.get('completion_tokens') if combined else None
        receipt = dict(policy=COMPOSITE_POLICY,finish_reason='stop',output_token_budget=reserve+budget,
            completion_tokens=completion,wire_format=FORMAT_POLICY,ocr_review_reason=review,
            stages=[dict(component='scene',complete=True,**primary),dict(component='ocr',complete=True,**final)])
        if syntax_repair is not None:
            receipt['stages'][1]['json_syntax_repair'] = syntax_repair
        return VisionResponse(payload,self.model,combined,(time.monotonic()-started)*1000,receipt)


    def close(self):
        if self._http is not None:
            self._http.close()


def build_vision_client(config: MultimodalConfig) -> OpenAICompatVisionClient:
    if not config.vlm_base_url:
        raise ExtractionError("vlm_base_url must be configured for an OpenAI-compatible VLM")
    from mandol_competition.config.unified import secret_from_environment
    api_key = secret_from_environment(config.vlm_api_key_env)
    if not api_key:
        raise ExtractionError(f"set the API key environment variable {config.vlm_api_key_env}")
    return OpenAICompatVisionClient(
        model=config.vlm_model, api_key=api_key,
        request_model=config.vlm_request_model,
        base_url=config.vlm_base_url, timeout=config.vlm_timeout_seconds,
        output_tokens=min(config.vlm_output_tokens, VISION_REQUEST_OUTPUT_LIMIT),
        max_output_tokens=min(config.vlm_max_output_tokens, VISION_REQUEST_OUTPUT_LIMIT),
        wire_format=('visual-memory-ocr-lines-v2'
            if config.prompt_version == 'multimodal_sgm_v3_ocr_lines' else 'legacy-json-v1'),
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

    @property
    def supports_resumable_ocr(self):
        from .ocr_lines import PROMPT_VERSION, FORMAT_POLICY
        return self.config.prompt_version == PROMPT_VERSION and (
            self._client is None or isinstance(self._client, OpenAICompatVisionClient)
            and self._client.wire_format == FORMAT_POLICY)

    def extract_resumable(self, *, resume=None, checkpoint=None, binding=None, legacy=False,
                          image_data_uri, context_text, image_position, image_count):
        """A request-scoped stage session; callbacks never enter shared fields."""
        if not self.supports_resumable_ocr:
            return self.extract(image_data_uri=image_data_uri, context_text=context_text,
                image_position=image_position, image_count=image_count)
        if self._client is None:
            self._client = build_vision_client(self.config)
        from .staged_ocr import complete_staged
        from .stage_state import RecoveryStateError, RecoveryBudgetExhausted
        from mandol_competition.inference.http import deadline_budget
        prompt = build_extraction_prompt(context_text, image_position, image_count, version=self.config.prompt_version)
        try:
            with deadline_budget(self._client.timeout_seconds):
                payload, usage, receipt, latency = complete_staged(self._client,
                    prompt=prompt, image_uri=image_data_uri, context=context_text,
                    position=image_position, count=image_count, parent=binding,
                    resume=resume, checkpoint=checkpoint, legacy=legacy,
                    schema_version=self.config.schema_version, preserve_exhausted=True)
            extraction = parse_extraction(payload, model=self.model_name,
                prompt_version=self.config.prompt_version, schema_version=self.config.schema_version)
            if self.config.vision_failure_policy == 'retry-required-v1' and not has_visual_evidence(extraction):
                raise VisionOutputError('vision_output_invalid_json', retryable=False)
        except RecoveryBudgetExhausted as exc:
            raise VisionExtractionUnavailable(retry_safe=False, unknown_outcome=False,
                error_class='vision_recovery_exhausted') from exc
        except RecoveryStateError as exc:
            raise VisionRecoveryExhausted() from exc
        except VisionOutputError as exc:
            raise VisionExtractionUnavailable(retry_safe=False, unknown_outcome=False,
                error_class=exc.error_class) from exc
        except VisionRequestError as exc:
            if exc.status_code == 429 and not exc.unknown_outcome and not exc.paid_output_failure:
                raise InferenceRateLimited(retry_after=exc.retry_after) from exc
            raise VisionExtractionUnavailable(retry_safe=not exc.unknown_outcome and not exc.paid_output_failure,
                unknown_outcome=exc.unknown_outcome, status_code=exc.status_code) from exc
        from .cache import _media_id, context_digest
        receipt = dict(receipt, media_id=_media_id(image_data_uri), context_sha256=context_digest(context_text),
            image_position=image_position, image_count=image_count, extraction_sha256=extraction_fingerprint(extraction))
        if not hasattr(self._observation_local, 'values'):
            self._observation_local.values = []
            self._observation_local.receipts = []
        self._observation_local.values.append((image_data_uri, usage, latency))
        self._observation_local.receipts.append(receipt)
        return extraction

    def close(self):
        if hasattr(self._client, "close"):
            self._client.close()
        self._observation_local.values = []
        self._observation_local.receipts = []

    def take_observations(self) -> list[tuple[str, dict | None, float]]:
        observations = getattr(self._observation_local, 'values', [])
        self._observation_local.values = []
        return observations

    def take_output_receipts(self) -> list[dict]:
        values = getattr(self._observation_local, 'receipts', [])
        self._observation_local.receipts = []
        return values

    def extract(
        self, *, image_data_uri: str, context_text: str,
        image_position: int, image_count: int,
    ) -> Extraction:
        prompt = build_extraction_prompt(context_text, image_position, image_count, version=self.config.prompt_version)
        if self._client is None:
            self._client = build_vision_client(self.config)
        attempts = max(1, self.config.vlm_max_attempts)
        last_error: Exception | None = None
        output_budget = (min(self.config.vlm_output_tokens, self._client.max_output_tokens)
            if isinstance(self._client, OpenAICompatVisionClient) else self.config.vlm_output_tokens)
        maximum_output_budget = (min(self.config.vlm_max_output_tokens, self._client.max_output_tokens)
            if isinstance(self._client, OpenAICompatVisionClient) else self.config.vlm_max_output_tokens)
        expanded_output = False
        paid_output_failure = False

        def preserve_paid_attempt(error):
            if paid_output_failure:
                error.vision_paid_output_failure = True
            return error

        for attempt in range(1, attempts + 1):
            retry_after = None
            if self.config.vision_failure_policy == "retry-required-v1":
                from mandol_competition.inference.http import remaining_budget
                if remaining_budget(1.) <= 0:
                    raise preserve_paid_attempt(InferenceDeadlineExceeded())
            try:
                options = dict(output_token_budget=output_budget) if isinstance(self._client, OpenAICompatVisionClient) else {}
                response = self._client.complete(prompt=prompt, image_data_uri=image_data_uri, **options)
                extraction = parse_extraction(
                    response.payload, model=self.model_name,
                    prompt_version=self.config.prompt_version,
                    schema_version=self.config.schema_version,
                )
                if self.config.vision_failure_policy == "retry-required-v1" and not has_visual_evidence(extraction):
                    raise ExtractionError("model returned no observable visual evidence")
            except VisionOutputTruncated as exc:
                last_error = exc
                paid_output_failure = True
                if expanded_output or output_budget >= maximum_output_budget:
                    break
                output_budget = maximum_output_budget
                expanded_output = True
            except VisionOutputError as exc:
                last_error = exc
                paid_output_failure = True
                if not exc.retryable:
                    break
            except InferenceDeadlineExceeded as exc:
                raise preserve_paid_attempt(exc)
            except VisionRequestError as exc:
                if exc.paid_output_failure:
                    paid_output_failure = True
                if exc.status_code == 429 and exc.retry_managed:
                    # The HTTP transport owns its bounded provider retries.
                    # Do not multiply them here or turn quota rejection into a
                    # successful but degraded visual-memory publication.
                    raise preserve_paid_attempt(InferenceRateLimited(retry_after=exc.retry_after)) from None
                last_error = exc
                if not exc.retryable or (self.config.vision_failure_policy == "retry-required-v1" and exc.unknown_outcome):
                    break
                retry_after = exc.retry_after
                if exc.status_code == 429 and retry_after is None:
                    retry_after = 60.
            except ExtractionError as exc:
                last_error = exc
            else:
                if not hasattr(self._observation_local, 'values'):
                    self._observation_local.values = []
                self._observation_local.values.append((image_data_uri, response.usage, response.latency_ms))
                if response.output_validation is not None:
                    from .cache import _media_id, context_digest
                    if not hasattr(self._observation_local, 'receipts'):
                        self._observation_local.receipts = []
                    self._observation_local.receipts.append(dict(response.output_validation,
                        media_id=_media_id(image_data_uri), context_sha256=context_digest(context_text),
                        image_position=image_position, image_count=image_count,
                        extraction_sha256=extraction_fingerprint(extraction)))
                return extraction

            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 4)
                if retry_after is not None:
                    if not math.isfinite(retry_after) or retry_after > min(60., self.config.vlm_timeout_seconds):
                        # Fail within the bounded budget instead of violating a
                        # provider's longer cooldown or sleeping without a limit.
                        break
                    delay = max(delay, retry_after)
                if self.config.vision_failure_policy == "retry-required-v1":
                    from mandol_competition.inference.http import remaining_budget
                    if remaining_budget(float('inf')) <= delay:
                        raise preserve_paid_attempt(InferenceDeadlineExceeded())
                time.sleep(delay)

        if isinstance(last_error, VisionRequestError) and last_error.status_code == 429:
            raise preserve_paid_attempt(InferenceRateLimited(retry_after=last_error.retry_after)) from None
        if isinstance(last_error, VisionOutputError):
            # Preserve a durable failed checkpoint and its recovery window. A
            # paid, known truncation must not reset the budget on every Add.
            raise VisionExtractionUnavailable(retry_safe=False, unknown_outcome=False,
                error_class=last_error.error_class) from last_error
        if self.config.vision_failure_policy == "retry-required-v1" or paid_output_failure:
            unknown = isinstance(last_error, VisionRequestError) and last_error.unknown_outcome
            raise VisionExtractionUnavailable(retry_safe=not unknown and not paid_output_failure, unknown_outcome=unknown,
                status_code=getattr(last_error, "status_code", None)) from last_error
        raise ExtractionError(
            f"vision extraction failed after {attempt} attempt(s): {type(last_error).__name__}"
        ) from last_error
