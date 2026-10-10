"""Opt-in evidence selection over existing raw sources, never Search answers."""
from dataclasses import asdict, dataclass
import re

from mandol_substrate.state.models import digest


@dataclass(frozen=True)
class AnswerUtilityPolicy:
    version: str = 'textual-answer-utility-v1'
    output_k: int = 8
    candidate_k: int = 30
    depth: str = 'fixed'
    selector: str = 'none'
    max_context_chars: int = 24000

    def __post_init__(self):
        if (self.version!='textual-answer-utility-v1' or self.depth not in {'fixed','adaptive-v1'}
            or self.selector not in {'none','mini','lifecycle'}
            or type(self.output_k) is not int or not 1<=self.output_k<=100
            or type(self.candidate_k) is not int or not 1<=self.candidate_k<=100
            or type(self.max_context_chars) is not int or not 128<=self.max_context_chars<=100000):
            raise ValueError('Invalid bounded Textual Answer-utility policy')

    @property
    def hash(self):return digest(asdict(self))

    def limit(self,query,requested):
        depth=self.output_k
        if self.depth=='adaptive-v1':
            # General question-text signals, no benchmark/category/ID input.
            if re.search(r'\b(all|list|compare|before|after|between|events|steps|both|sequence)\b|所有|列出|比较|之前|之后|步骤',query,re.I):
                depth=16
            else:depth=6
        return min(requested,depth)


def deduplicated_indices(hits):
    # Dates, entities and corrections remain part of content. Never conflate
    # identical wording from distinct dated observations into a current fact.
    seen=set();result=[]
    for i,hit in enumerate(hits):
        key=digest(hit['content'])
        if key not in seen:result.append(i);seen.add(key)
    return result


def candidate_views(hits, *, canonical=None, lifecycle=False):
    indices=deduplicated_indices(hits) if lifecycle else list(range(len(hits)))
    views=[]
    for i in indices:
        text=hits[i]['content']
        if lifecycle and canonical is not None:
            memory=canonical[hits[i]['id']]
            # Canonical active/version is projection validity, not real-world
            # current/historical status. Only explicit source mention dates are
            # supplied; no temporal label inferred from ingestion time.
            text='[Raw source; canonical revision verified; fact current status unknown]\n'+text
            if memory.business_timestamp_ms is not None:
                text+='\n[Explicit source mention Unix milliseconds: '+str(memory.business_timestamp_ms)+']'
        views.append({'text':text})
    return indices,views


def select_existing(policy,hits,*,query,options,requested,selector=None,canonical=None):
    limit=policy.limit(query,requested)
    if policy.selector=='none':return hits[:limit],False
    if selector is None:raise ValueError('Explicit mini evidence selector required')
    candidates=hits[:policy.candidate_k]
    indices,views=candidate_views(candidates,canonical=canonical,lifecycle=policy.selector=='lifecycle')
    selected=selector.select(query=query,options=options,candidates=views,limit=min(limit,len(views)))
    if selected.status!='selected':return hits[:requested],True
    # Return only original evidence IDs/content. Reciprocal ordinal score says
    # higher ranks are better; it is explicitly not a Querit neural probability.
    return [{**candidates[indices[i]],'score':1/(rank+1)} for rank,i in enumerate(selected.indices)],False
