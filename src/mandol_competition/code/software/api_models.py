"""Audited coding-only chat access through Mandol's existing CloseAI provider.

No textual-provider generation/truncation wrapper is called: full tool messages,
explicit prompt selection and provider usage are needed for paired evaluation.
"""

import logging
import os
import re
import time
from urllib.parse import urlsplit

from .api_context import MaximumSafeBudget, MaximumSafeContext, canonical_json, digest


MODEL_ALIAS = "gpt-4o-mini-closeai"
ACTUAL_MODEL = "gpt-4o-mini"
PROVIDER = "openai-proxy"


class CodingAPIError(RuntimeError):
    """Safe, structured failure; never stringify provider exceptions/headers."""

    def __init__(self, reason, record):
        super().__init__(reason)
        self.record = {"status": "NOT_EVALUABLE", "reason": reason, **record}


class CodingAPIClient:
    def __init__(self, llm_client, *, budget=None, secret="", retries=2, sampling=None):
        self.budget = budget or MaximumSafeBudget()
        if (llm_client.model_name != MODEL_ALIAS or llm_client.actual_model != ACTUAL_MODEL
                or llm_client.provider_name != PROVIDER or llm_client.context_length != 128000
                or llm_client.max_output_tokens != 16384
                or llm_client.max_context_tokens != self.budget.max_input_tokens):
            raise ValueError("Formal provider identity or explicit input reservation mismatch")
        self.llm = llm_client
        self.context = MaximumSafeContext(llm_client.tokenizer, self.budget)
        self.secret = secret
        self.retries = retries
        self.sampling = sampling or {"temperature": 0, "top_p": 1, "seed": 17}

    @classmethod
    def from_env(cls, *, budget=None, timeout=300, retries=2):
        # Read before importing project settings (which may load a .env file).
        key = os.environ.get("CLOSEAI_API_KEY")
        if not key:
            raise CodingAPIError("missing_CLOSEAI_API_KEY_environment", {})
        for name in ("openai", "httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)
        from mandol.llm.llm_client import LLMClient
        from mandol.llm.configs import API_PROVIDERS

        url = API_PROVIDERS[PROVIDER]["base_url"]
        parsed = urlsplit(url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or key in url):
            raise CodingAPIError("unsafe_provider_base_url", {})
        budget = budget or MaximumSafeBudget()
        llm = LLMClient(model_name=MODEL_ALIAS, api_key=key, base_url=url,
                        max_context_ratio=budget.max_context_ratio,
                        request_max_retries=0, request_timeout=timeout)
        return cls(llm, budget=budget, secret=key, retries=retries)

    def identity(self):
        import importlib.metadata

        return {"model_alias": MODEL_ALIAS, "actual_model": ACTUAL_MODEL, "provider": PROVIDER,
                **self.budget.record(), "sampling": self.sampling,
                "request_timeout_seconds": self.llm.request_timeout,
                "request_retries": self.retries, "provider_base_url": self.llm.base_url,
                "openai_sdk_version": importlib.metadata.version("openai"),
                "tiktoken_version": importlib.metadata.version("tiktoken")}

    def prepare(self, messages, *, task_text, memory_text="", tools=None, json_format=False):
        tools = tools or []
        selected, audit = self.context.fit(messages, tools, task_text=task_text, memory_text=memory_text)
        request = {"model": ACTUAL_MODEL, "messages": selected,
                   "max_tokens": self.budget.max_tokens, **self.sampling}
        if tools:
            request.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if json_format:
            request["response_format"] = {"type": "json_object"}
        if self.secret and self.secret in canonical_json(request):
            raise CodingAPIError("credential_in_request_body_rejected", {})
        return request, {**self.identity(), **audit, "request_sha256": digest(request)}

    def complete(self, request, audit):
        # The caller persists this exact request and audit before any API call.
        if (digest(request) != audit["request_sha256"] or request["max_tokens"] != self.budget.max_tokens
                or request["model"] != ACTUAL_MODEL):
            raise ValueError("Prepared request changed before transmission")
        attempts = []
        for attempt in range(self.retries + 1):
            start = time.perf_counter()
            try:
                response = self.llm.client.chat.completions.create(**request).model_dump(mode="json")
            except Exception as error:
                status = getattr(error, "status_code", None)
                retryable = (status in {408, 409, 425, 429, 500, 502, 503, 504}
                             or type(error).__name__ in {"APITimeoutError", "APIConnectionError"})
                attempts.append({"attempt": attempt, "error_type": type(error).__name__,
                                 "http_status": status if type(status) is int else None,
                                 "seconds": time.perf_counter() - start})
                if retryable and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise CodingAPIError("provider_request_failed", {"attempts": attempts, "context": audit}) from None
            attempts.append({"attempt": attempt, "seconds": time.perf_counter() - start})
            # A misbehaving proxy might echo credentials in its body. Never
            # persist or pass such a response to the actor/container.
            if self.secret and self.secret in canonical_json(response):
                raise CodingAPIError("credential_echo_response_rejected", {"attempts": attempts})
            usage = response.get("usage") or {}
            record = {"attempts": attempts, "context": audit, "response": response}
            if not re.fullmatch(r"gpt-4o-mini(?:-\d{4}-\d{2}-\d{2})?", response.get("model") or ""):
                raise CodingAPIError("unexpected_returned_model", record)
            if (any(type(usage.get(k)) is not int or usage[k] < 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
                raise CodingAPIError("missing_or_invalid_provider_usage", record)
            if (usage["prompt_tokens"] > self.budget.max_input_tokens
                    or usage["prompt_tokens"] + self.budget.max_tokens > self.budget.context_length
                    or usage["completion_tokens"] > self.budget.max_tokens):
                raise CodingAPIError("actual_context_budget_violation", record)
            if len(response.get("choices") or []) != 1:
                raise CodingAPIError("invalid_choice_count", record)
            return {**record, "context": {**audit,
                "actual_prompt_tokens": usage["prompt_tokens"],
                "actual_completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"], "usage_source": "provider_response",
                "returned_model": response["model"], "system_fingerprint": response.get("system_fingerprint")}}
        raise AssertionError("Unreachable retry state")
