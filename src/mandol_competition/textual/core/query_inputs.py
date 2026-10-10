"""Opt-in, gold-free choice hints for candidate generation, never for writeback.

Every supplied choice is included as an unverified hypothesis. Reranking and
memory navigation retain the original question. An oversized hint set falls
back to the whole original question rather than favoring early choices.
"""
from dataclasses import asdict, dataclass

from mandol_substrate.state.models import digest

HEADER = "\n\nUnverified answer choices, used only as retrieval hints:\n"


@dataclass(frozen=True)
class QueryInputPolicy:
    version: str = "textual-all-choice-hints-v1"
    max_rendered_chars: int = 16000

    def __post_init__(self):
        if self.version != "textual-all-choice-hints-v1" or type(self.max_rendered_chars) is not int or not 512 <= self.max_rendered_chars <= 16000:
            raise ValueError("Explicit bounded query-input recipe required")

    @property
    def hash(self):
        return digest({"policy": asdict(self), "header": HEADER, "render": "all-or-original-v1",
                       "guard": "remote-sparse-full-input-fit-before-any-encoding-v1"})

    def render(self, query, choices):
        if choices is None or choices == []:
            return query, "NO_CHOICES_ORIGINAL"
        if not isinstance(choices, list) or any(not isinstance(choice, str) for choice in choices):
            raise ValueError("Only actual wire choices may be retrieval hints")
        rendered = query + HEADER + "\n".join(f"- {choice}" for choice in choices)
        if len(rendered) > self.max_rendered_chars:
            return query, "ALL_CHOICES_EXCEED_BUDGET_ORIGINAL"
        return rendered, "ALL_CHOICES_UNVERIFIED_HINTS"


def candidate_query(candidate, query, choices, *, sparse_client=None):
    policy = candidate.get("query_input_policy") if candidate else None
    if not policy:
        return query
    rendered = QueryInputPolicy(**policy).render(query, choices)[0]
    if rendered != query:
        if sparse_client is None or not hasattr(sparse_client, "count_tokens"):
            raise ValueError("Choice-hint candidate requires remote sparse token-budget admission")
        if sparse_client.count_tokens(rendered) > sparse_client.max_tokens:
            return query  # All choices or none, including the frozen sparse token budget.
    return rendered
