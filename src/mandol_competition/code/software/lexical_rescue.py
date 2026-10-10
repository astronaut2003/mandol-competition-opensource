"""Opt-in bounded exact-code candidate insertion, never global fusion."""
import re
from .anchors import CodeAnchorExtractor
from .raw_evidence import exact_matches


def rescue_anchors(query):
    urls=[m.span() for m in re.finditer(r'https?://\S+',query)]
    kinds={'path','symbol','error','test','config','constant','version'}
    candidates=[]
    for a in CodeAnchorExtractor().extract(query):
        if a.kind not in kinds or any(s<a.end and a.start<e for s,e in urls):continue
        raw=a.raw.split(':',1)[0] if a.kind=='error' else a.raw
        candidates.append((a.start,a.start+len(raw),raw))
    selected=[]
    for start,end,raw in sorted(set(candidates),key=lambda r:(-(r[1]-r[0]),r[0],r[2])):
        if not any(start<b and a<end for a,b,_ in selected):selected.append((start,end,raw))
    return sorted({raw for _,_,raw in selected},key=str.casefold)


class ConditionalLexicalRescue:
    def apply(self, dense_hits, lexical_hits, sources, *, query, top_k=100):
        anchors=rescue_anchors(query)
        audit={'anchors':anchors,'triggered':len(anchors)>=2,'rescued_episode_id':None,'matches':[]}
        users={h.episode.user_id for h in list(dense_hits)+list(lexical_hits)}
        if len(users)>1:
            raise ValueError('Lexical rescue cannot mix user scopes')
        if top_k<=0 or not query.strip():
            return [],audit
        if len(anchors)<2:return list(dense_hits[:top_k]),audit
        first={h.episode.episode_id for h in dense_hits[:8]}
        dense_by_id={h.episode.episode_id:h for h in dense_hits}
        for hit in lexical_hits[:100]:
            ep=hit.episode
            if ep.episode_id in first:continue
            ep.validate_provenance(sources)
            matches={a for uid in ep.source_uids for _,_,a in exact_matches(sources[uid].raw_data['text_content'],anchors)}
            if len(matches)<2:continue
            selected=dense_by_id.get(ep.episode_id,hit)
            ranking=[h for h in dense_hits if h.episode.episode_id!=ep.episode_id]
            ranking.insert(min(7,len(ranking)),selected)
            audit.update(rescued_episode_id=ep.episode_id,matches=sorted(matches))
            return ranking[:top_k],audit
        return list(dense_hits[:top_k]),audit
