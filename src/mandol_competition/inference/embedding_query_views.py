"""Bounded, lossless input views for the three tracks' existing v4 query encoder.

Only overlong query representations are approximated. Documents and the full
query passed to lexical retrieval, graph selection and reranking are unchanged.
"""
from dataclasses import dataclass
from bisect import bisect_left
import math

from mandol_competition.inference.errors import (
    InferenceDeadlineExceeded, InferenceInputRejected, InferenceSchemaError,
)
from mandol_substrate.deadlines import remaining_budget
from mandol_substrate.state.models import digest
from .embedding_tokenizer import encode_text, descriptor as tokenizer_descriptor

POLICY = 'v4-query-token-views-v2'
# User-selected content budget; observed provider framing adds one more token.
MAX_VIEW_TOKENS = 8_000
PROVIDER_TOKEN_LIMIT = 8_192
OBSERVED_FRAMING_TOKENS = 1
# Keep the existing whole-request capacity guard. This is not a per-view cut.
MAX_QUERY_UTF8_BYTES = 1_024_000
MAX_QUERY_VIEWS = 128


@dataclass(frozen=True)
class QueryPlan:
    parts: tuple[str, ...]
    token_counts: tuple[int, ...]

    @property
    def views(self):
        return tuple(part for part in self.parts if part.strip())

    @property
    def view_token_counts(self):
        return tuple(count for part, count in zip(self.parts, self.token_counts, strict=True)
                     if part.strip())


def _budget():
    if remaining_budget(1.) <= 0:
        raise InferenceDeadlineExceeded()


def _natural_end(text, start, end):
    lower = max(start + 1, end - 256)
    for separator in ('\n\n', '\n', ' '):
        position = text.rfind(separator, lower, end)
        if position >= lower:
            return position + len(separator)
    return end


def plan_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('nonempty original embedding query required')
    _budget()
    try:
        size = len(query.encode('utf-8'))
    except UnicodeEncodeError:
        raise InferenceInputRejected('embedding_query_invalid_unicode', status_code=400) from None
    if size > MAX_QUERY_UTF8_BYTES:
        raise InferenceInputRejected('embedding_query_capacity', status_code=413)
    encoded = encode_text(query)
    _budget()
    if len(encoded) <= MAX_VIEW_TOKENS:
        return QueryPlan((query,), (len(encoded),))
    # Offsets point into the original Unicode string, including when the
    # tokenizer normalizes combining marks or multiple byte tokens share a char.
    # Slice the original text, never decode arbitrary groups of token IDs.
    offsets = encoded.offsets
    starts = [offset[0] for offset in offsets]
    parts, counts, start = [], [], 0
    while start < len(query):
        _budget()
        first = bisect_left(starts, start)
        stop = first + MAX_VIEW_TOKENS
        end = offsets[stop][0] if stop < len(offsets) else len(query)
        if end < len(query):
            end = _natural_end(query, start, end)
        if end <= start:
            raise InferenceSchemaError('embedding_tokenizer_invalid_offsets')
        piece = encode_text(query[start:end])
        # A new boundary can change BPE merges/normalization. Recount the actual
        # dispatched substring and shrink at its own offsets until it fits.
        while len(piece) > MAX_VIEW_TOKENS:
            _budget()
            smaller = start + piece.offsets[MAX_VIEW_TOKENS][0]
            if not start < smaller < end:
                raise InferenceSchemaError('embedding_tokenizer_invalid_offsets')
            end = smaller
            piece = encode_text(query[start:end])
        if not len(piece):
            raise InferenceSchemaError('embedding_tokenizer_empty_view')
        parts.append(query[start:end])
        counts.append(len(piece))
        start = end
        if len(parts) > MAX_QUERY_VIEWS:
            raise InferenceInputRejected('embedding_query_capacity', status_code=413)
    _budget()
    return QueryPlan(tuple(parts), tuple(counts))


def _checked_pair(pair, dimension=None):
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise InferenceSchemaError('embedding_view_invalid_pair')
    dense, sparse = pair
    if (not isinstance(dense, (list, tuple)) or not 0 < len(dense) <= 2048
            or dimension is not None and len(dense) != dimension
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in dense)
            or not any(dense)):
        raise InferenceSchemaError('embedding_view_invalid_dense')
    if (not isinstance(sparse, dict) or not sparse
            or any(type(k) is not int or not 0 <= k < 2**32
                   or type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for k, v in sparse.items()) or not any(sparse.values())):
        raise InferenceSchemaError('embedding_view_invalid_sparse')
    return dense, sparse


def encode_query_views(encoders, query):
    plan = plan_query(query)  # Validate the complete plan before any dispatch.
    views = plan.views
    encode = encoders if callable(encoders) else encoders.encode_query
    if remaining_budget(1.) <= 0:
        raise InferenceDeadlineExceeded()
    if len(views) == 1:
        # Preserve the provider's exact ordinary-query vector and sparse values.
        return encode(views[0])
    pairs, dimension = [], None
    weights = plan.view_token_counts
    for view in views:
        if remaining_budget(1.) <= 0:
            raise InferenceDeadlineExceeded()
        pair = _checked_pair(encode(view), dimension)
        dimension = len(pair[0])
        pairs.append(pair)
    if remaining_budget(1.) <= 0:
        raise InferenceDeadlineExceeded()
    total = sum(weights)
    weights = [weight / total for weight in weights]
    try:
        dense = [math.fsum(w * pair[0][i] for w, pair in zip(weights, pairs, strict=True))
                 for i in range(dimension)]
        norm = math.hypot(*dense)
        if not math.isfinite(norm) or norm <= 0:
            raise InferenceSchemaError('embedding_view_zero_merged_dense')
        dense = [v / norm for v in dense]
        sparse = {}
        for weight, (_, values) in zip(weights, pairs, strict=True):
            for index, value in values.items():
                sparse[index] = sparse.get(index, 0.) + weight * value
        sparse = {k: v for k, v in sparse.items() if v > 0}
    except (OverflowError, ValueError):
        raise InferenceSchemaError('embedding_view_invalid_merge') from None
    _checked_pair((dense, sparse), dimension)
    if remaining_budget(1.) <= 0:
        raise InferenceDeadlineExceeded()
    return dense, sparse


def descriptor():
    policy = dict(id=POLICY, model='text-embedding-v4', query_only=True,
        max_view_tokens=MAX_VIEW_TOKENS, max_views=MAX_QUERY_VIEWS,
        max_query_utf8_bytes=MAX_QUERY_UTF8_BYTES, tokenizer=tokenizer_descriptor(),
        provider_token_limit=PROVIDER_TOKEN_LIMIT, observed_framing_tokens=OBSERVED_FRAMING_TOKENS,
        remaining_headroom_tokens=PROVIDER_TOKEN_LIMIT-MAX_VIEW_TOKENS-OBSERVED_FRAMING_TOKENS,
        split='original-unicode-offsets-paragraph-word-boundaries-recounted',
        short_query='original-text-and-provider-pair',
        long_query='approximate-token-weighted-dense-mean-l2-and-sparse-union-mean',
        lexical_and_reranker_query='original-full-query',
        partial_failure='propagate-no-partial-results',
        storage_or_document_encoding_changed=False)
    return policy | {'sha256': digest(policy)}
