"""Explicit output reservation and deterministic, auditable chat retention.

This opt-in coding profile does not change Mandol's textual context defaults.
Preflight counts are local tokenizer measurements; provider usage is recorded
separately and is authoritative for actual token consumption.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def exchange_digest(pair):
    """Ignore transport IDs when recognizing identical command observations."""
    clean = deepcopy(pair)
    for call in clean[0].get("tool_calls") or []:
        call.pop("id", None)
    clean[1].pop("tool_call_id", None)
    return digest(clean)


class ProtectedContextOverflow(ValueError):
    pass


@dataclass(frozen=True)
class MaximumSafeBudget:
    context_length: int = 128000
    max_tokens: int = 8192
    safety_margin: int = 512
    context_mode: str = "maximum_safe"

    def __post_init__(self):
        if (self.context_mode != "maximum_safe" or self.context_length != 128000
                or type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 16384
                or type(self.safety_margin) is not int or not 128 <= self.safety_margin <= 2048):
            raise ValueError("Invalid fixed GPT-4o-mini maximum-safe budget")

    @property
    def max_input_tokens(self):
        return self.context_length - self.max_tokens - self.safety_margin

    @property
    def max_context_ratio(self):
        return self.max_input_tokens / self.context_length

    def record(self):
        return {**asdict(self), "reserved_output_tokens": self.max_tokens,
                "max_input_tokens": self.max_input_tokens,
                "configured_max_context_ratio": self.max_context_ratio}


class MaximumSafeContext:
    """Retain task/memory verbatim; prune complete older exchanges only on overflow.

    The newest two exchanges are protected. Exact duplicate earlier command /
    observation pairs are removed first, then oldest earlier exchanges. If the
    protected tool output alone is too large, retain deterministic head/tail
    excerpts with explicit omission markers and hashes. Task/memory are never
    excerpted. No summarizer, relevance model, reflection or planner is used.
    """

    def __init__(self, tokenizer, budget=None):
        if tokenizer is None or tokenizer.name != "o200k_base":
            raise ValueError("Formal context accounting requires o200k_base")
        self.tokenizer = tokenizer
        self.budget = budget or MaximumSafeBudget()

    def count_text(self, text):
        return len(self.tokenizer.encode(text, disallowed_special=()))

    def count_prompt(self, messages, tools):
        # Chat content is tokenized after JSON decoding. Counting escaped
        # newlines in the HTTP body would waste a large fraction of the window
        # for source/log-heavy histories. Framing/schema remain local estimates;
        # the separate 512-token reserve covers template serialization variance.
        total = 3
        for message in messages:
            total += 3
            for key, value in message.items():
                if key == "tool_calls":
                    for call in value:
                        total += 8 + self.count_text(call["id"])
                        total += self.count_text(call["function"]["name"])
                        total += self.count_text(call["function"]["arguments"])
                elif isinstance(value, str):
                    total += self.count_text(value) + (1 if key == "name" else 0)
        if tools:
            total += self.count_text(canonical_json(tools)) + 20 * len(tools) + 12
        return total

    def fit(self, messages, tools, *, task_text, memory_text):
        if (len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user"
                or task_text not in messages[1]["content"]
                or (memory_text and memory_text not in messages[1]["content"])):
            raise ValueError("Task and selected memory must be present in the protected prefix")
        prefix = deepcopy(messages[:2])
        tail = messages[2:]
        if len(tail) % 2:
            raise ValueError("Context must end at a complete assistant/observation exchange")
        pairs = []
        for offset in range(0, len(tail), 2):
            pair = deepcopy(tail[offset:offset + 2])
            if pair[0]["role"] != "assistant" or pair[1]["role"] not in {"tool", "user"}:
                raise ValueError("Invalid actor exchange")
            calls = pair[0].get("tool_calls") or []
            if pair[1]["role"] == "tool" and (len(calls) != 1 or calls[0]["id"] != pair[1].get("tool_call_id")):
                raise ValueError("Unmatched tool response")
            if pair[1]["role"] == "user" and calls:
                raise ValueError("A tool call requires its matching tool response")
            pairs.append((offset // 2, pair))
        original = self.count_prompt(messages, tools)
        if self.count_prompt(prefix, tools) > self.budget.max_input_tokens:
            raise ProtectedContextOverflow("Protected task/memory exceed the explicit input budget")
        retained = list(pairs)
        protected = {i for i, _ in pairs[-2:]}
        last_occurrence = {exchange_digest(pair): i for i, pair in pairs}
        duplicates = [i for i, pair in pairs if i not in protected and last_occurrence[exchange_digest(pair)] != i]
        removal_order = duplicates + [i for i, _ in pairs if i not in protected and i not in duplicates]
        dropped, excerpts = [], []

        def assembled():
            return prefix + [message for _, pair in retained for message in pair]

        for index in removal_order:
            if self.count_prompt(assembled(), tools) <= self.budget.max_input_tokens:
                break
            retained = [(i, pair) for i, pair in retained if i != index]
            dropped.append({"exchange": index, "priority": "P2_duplicate" if index in duplicates else "P1_older",
                            "sha256": digest(pairs[index][1])})
        # Preserve tools' identities and most recent failure tails; omit only
        # body bytes when protected observations cannot all fit in the window.
        for index, pair in retained:
            if self.count_prompt(assembled(), tools) <= self.budget.max_input_tokens:
                break
            raw = pair[1]["content"]
            tokens = self.tokenizer.encode(raw, disallowed_special=())
            if len(tokens) <= 128:
                continue
            raw_sha = hashlib.sha256(raw.encode()).hexdigest()
            low, high, best = 128, len(tokens), None
            while low <= high:
                keep = (low + high) // 2
                marker = f"\n[Tool output excerpt: {len(tokens) - keep} tokens omitted; raw SHA-256 {raw_sha}]\n"
                pair[1]["content"] = self.tokenizer.decode(tokens[:keep // 2]) + marker + self.tokenizer.decode(tokens[-(keep - keep // 2):])
                if self.count_prompt(assembled(), tools) <= self.budget.max_input_tokens:
                    best = pair[1]["content"]
                    low = keep + 1
                else:
                    high = keep - 1
            if best is None:
                pair[1]["content"] = (self.tokenizer.decode(tokens[:64])
                    + f"\n[Tool output excerpt; raw SHA-256 {raw_sha}]\n" + self.tokenizer.decode(tokens[-64:]))
            else:
                pair[1]["content"] = best
            excerpts.append({"exchange": index, "raw_sha256": raw_sha, "raw_tokens": len(tokens),
                             "retained_text_tokens": self.count_text(pair[1]["content"])})
        selected = assembled()
        final = self.count_prompt(selected, tools)
        if final > self.budget.max_input_tokens:
            raise ProtectedContextOverflow("Protected actions exceed the budget after explicit tool-output excerpts")
        assert selected[:2] == messages[:2]
        audit = {**self.budget.record(), "task_chars": len(task_text), "task_tokens": self.count_text(task_text),
                 "memory_chars": len(memory_text), "memory_tokens": self.count_text(memory_text),
                 "conversation_tokens": self.count_text(canonical_json(messages[2:])),
                 "tool_history_tokens": sum(self.count_text(m.get("content", "")) for m in messages[2:] if m["role"] == "tool"),
                 "original_prompt_tokens_estimate": original, "final_prompt_tokens_estimate": final,
                 "token_count_method": "o200k_base decoded chat fields plus framing/schema allowance; provider usage saved separately",
                 "dropped_exchanges": dropped, "excerpted_observations": excerpts,
                 "retained_exchanges": [i for i, _ in retained], "protected_prefix_sha256": digest(prefix),
                 "selected_messages_sha256": digest(selected), "tool_schema_sha256": digest(tools)}
        return selected, audit
