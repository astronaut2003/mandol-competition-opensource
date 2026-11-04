"""Compatibility entry point for the shared named-Qwen request guard."""
from mandol_competition.inference.rerank_query_views import (
    POLICY,MAX_VIEW_TOKENS,MAX_PAIR_TOKENS,MAX_REQUEST_TOKENS,MAX_VIEW_CHARACTERS,
    MAX_QUERY_UTF8_BYTES,MAX_QUERY_VIEWS,RRF_K,RerankQueryLimitError,QueryPlan,
    plan_query,request_plan,rerank_query_views,descriptor,
)
# Legacy diagnostic imports only. This is a character guard, not a byte cut.
MAX_VIEW_UTF8_BYTES=MAX_VIEW_CHARACTERS
