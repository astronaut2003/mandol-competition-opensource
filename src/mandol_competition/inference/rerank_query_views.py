"""Shared Qwen3.7 budgets and tie-aware RRF; never discard candidates or text."""
from bisect import bisect_left
from dataclasses import dataclass
import math

from .errors import InferenceDeadlineExceeded,InferenceInputRejected,InferenceSchemaError,InferenceUnavailable
from .rerank_tokenizer import encode_text,descriptor as tokenizer_descriptor
from mandol_substrate.deadlines import remaining_budget
from mandol_substrate.state.models import digest

POLICY='qwen37-token-budget-rrf-v2'
MAX_VIEW_TOKENS=29_800
MAX_PAIR_TOKENS=32_568
MAX_REQUEST_TOKENS=119_800
MAX_VIEW_CHARACTERS=49_800
MAX_QUERY_UTF8_BYTES=928_000
MAX_QUERY_VIEWS=256
MAX_PROVIDER_REQUESTS=256
MAX_DOCUMENTS=500
RRF_K=60

class RerankQueryLimitError(ValueError):
    pass

def _budget():
    if remaining_budget(1.)<=0:raise InferenceDeadlineExceeded()

@dataclass(frozen=True)
class QueryPlan:
    parts:tuple[str,...]
    token_counts:tuple[int,...]
    original_chars:int
    original_utf8_bytes:int
    @property
    def views(self):return tuple(p for p in self.parts if p.strip())

def _split(text,tokens,chars=MAX_VIEW_CHARACTERS):
    _budget()
    if not isinstance(text,str) or not text.strip():raise ValueError('nonempty reranking text required')
    try:size=len(text.encode('utf-8'))
    except UnicodeEncodeError:raise InferenceInputRejected('reranker_invalid_unicode',status_code=400) from None
    if size>MAX_QUERY_UTF8_BYTES:raise RerankQueryLimitError('reranking input capacity')
    encoded=encode_text(text);_budget()
    if len(encoded)<=tokens and len(text)<=chars:
        return QueryPlan((text,),(len(encoded),),len(text),size)
    offsets=encoded.offsets;starts=[r[0] for r in offsets]
    parts=[];counts=[];start=0
    while start<len(text):
        _budget();first=bisect_left(starts,start);stop=first+tokens
        end=min(offsets[stop][0] if stop<len(offsets) else len(text),start+chars)
        if end<len(text):
            lower=max(start+1,end-256)
            for sep in ('\n\n','\n',' '):
                pos=text.rfind(sep,lower,end)
                if pos>=lower:end=pos+len(sep);break
        if end<=start:raise InferenceSchemaError('reranker_invalid_offsets')
        piece=encode_text(text[start:end])
        while len(piece)>tokens:
            _budget();smaller=start+piece.offsets[tokens][0]
            if not start<smaller<end:raise InferenceSchemaError('reranker_invalid_offsets')
            end=smaller;piece=encode_text(text[start:end])
        parts.append(text[start:end]);counts.append(len(piece));start=end
        if len(parts)>MAX_QUERY_VIEWS:raise RerankQueryLimitError('reranking view capacity')
    return QueryPlan(tuple(parts),tuple(counts),len(text),size)

def plan_query(query):return _split(query,MAX_VIEW_TOKENS)

def _checked_scores(scores,count):
    if (not isinstance(scores,(list,tuple)) or len(scores)!=count or
        any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in scores)):
        raise InferenceSchemaError('query_view_invalid_rerank_scores')
    return scores

@dataclass(frozen=True)
class RequestView:
    query:str
    documents:tuple[str,...]

def request_plan(query,documents,*,scale=1.):
    _budget()
    if not isinstance(documents,(tuple,list)) or any(not isinstance(d,str) or not d.strip() for d in documents):
        raise ValueError('original nonempty reranking documents required')
    if len(documents)>MAX_DOCUMENTS:raise InferenceInputRejected('reranker_document_capacity',status_code=413)
    initial=plan_query(query)
    if not documents:return ()
    n=len(documents);dc=[len(encode_text(d)) for d in documents];_budget()
    item=max(1,int(MAX_VIEW_TOKENS*scale));pair=max(2,int(MAX_PAIR_TOKENS*scale));total=max(2*n,int(MAX_REQUEST_TOKENS*scale))
    chars=max(1,int(MAX_VIEW_CHARACTERS*scale))
    qc=sum(initial.token_counts)
    if (len(initial.parts)==1 and qc<=item and len(query)<=chars and
        max(dc)<=item and all(len(d)<=chars for d in documents) and
        qc+max(dc)<=pair and qc*n+sum(dc)<=total):
        return (RequestView(query,tuple(documents)),)
    # Preserve full documents whenever the original pool fits with any query.
    minimum_query=min(8,qc)
    if (max(dc)<=item and all(len(d)<=chars for d in documents)
            and sum(dc)+n*minimum_query<=total and max(dc)+minimum_query<=pair):
        docs=[(d,) for d in documents]
    else:
        qt=max(1,min(qc,item,pair//2,total//(2*n)))
        dt=max(1,min(item,pair-qt,total//n-qt))
        docs=[_split(d,dt,chars).views for d in documents]
    requests=[]
    for round_index in range(max(map(len,docs))):
        # Same original candidate identities and order in every request. Each
        # document's lossless parts is visited; shorter lists repeat cyclically.
        ds=tuple(parts[round_index%len(parts)] for parts in docs)
        counts=[len(encode_text(d)) for d in ds]
        qb=min(item,pair-max(counts),(total-sum(counts))//n)
        if qb<1:raise RerankQueryLimitError('reranking request capacity')
        for q in _split(query,qb,chars).views:
            count=len(encode_text(q))
            if count*n+sum(counts)>total or count+max(counts)>pair:
                raise InferenceSchemaError('reranker_budget_plan_invalid')
            requests.append(RequestView(q,ds))
            if len(requests)>MAX_PROVIDER_REQUESTS:raise RerankQueryLimitError('reranking request capacity')
    _budget()
    return tuple(requests)

def _execute(client,query,documents,plan):
    if not plan:return []
    call=getattr(client,'_score_native',None) or client.rerank
    if len(plan)==1:
        ds=documents if plan[0].documents==tuple(documents) else list(plan[0].documents)
        _budget();values=_checked_scores(call(plan[0].query,ds),len(documents));_budget()
        return values
    fused=[0.]*len(documents)
    for view in plan:
        ds=documents if view.documents==tuple(documents) else list(view.documents)
        _budget();scores=_checked_scores(call(view.query,ds),len(documents))
        rank=0;previous=None
        for position,index in enumerate(sorted(range(len(scores)),key=lambda i:-scores[i]),1):
            if scores[index]!=previous:rank,previous=position,scores[index]
            fused[index]+=1/(RRF_K+rank)
    _budget()
    return [min(1.,value*(RRF_K+1)/len(plan)) for value in fused]

def rerank_query_views(client,query,documents):
    for reduction in range(3):
        try:
            plan=request_plan(query,documents,scale=2.**-reduction)
            return _execute(client,query,documents,plan)
        except RerankQueryLimitError:
            raise InferenceInputRejected('reranker_input_capacity',status_code=413) from None
        except InferenceUnavailable as exc:
            if (exc.status_code==400 and getattr(exc,'input_length_rejected',False)
                    and reduction<2):
                # Replan known rejected input, never replay an unknown paid
                # outcome, a 429, deadline, or arbitrary InvalidParameter.
                continue
            if exc.status_code in {400,413,422} and not isinstance(exc,InferenceInputRejected):
                raise InferenceInputRejected('reranker_input_rejected',status_code=exc.status_code,
                    provider_error_code=exc.provider_error_code,provider_request_id=exc.provider_request_id) from None
            raise

def descriptor():
    value=dict(id=POLICY,model='qwen3.7-text-rerank',tokenizer=tokenizer_descriptor(),
        max_view_tokens=MAX_VIEW_TOKENS,max_view_characters=MAX_VIEW_CHARACTERS,
        max_pair_tokens=MAX_PAIR_TOKENS,max_request_tokens=MAX_REQUEST_TOKENS,
        request_cost='query_tokens*document_count+sum(document_tokens)',headroom_tokens=200,
        max_views=MAX_QUERY_VIEWS,max_provider_requests=MAX_PROVIDER_REQUESTS,rrf_k=RRF_K,
        short_query='original-text-and-provider-scores',long_query='tie-aware-normalized-rrf',
        score_semantics='long-query-fused-rank-score-not-provider-probability',
        same_document_pool=True,candidate_identities_and_order='preserved-in-every-view',
        document_views='lossless-parts-aligned-cyclic-rounds-if-required',
        known_length_rejection='at-most-two-budget-reductions-no-partial-scores',
        partial_failure='propagate-no-partial-results',storage_or_encoding_changed=False)
    return value|{'sha256':digest(value)}
