"""Experimental lexical filtering; no Dense fusion or score boosts."""
from collections import Counter
import math
import re
from .anchors import CodeAnchorExtractor

SAFE_STOPWORDS = frozenset('a an the of to for this that please implementation project code function method change update modify task issue and or is are was were be been being it its in on at as by from with we you they he she our their your i me my do does did can could would should will shall have has had into out up about also there here which what when where how then than these those'.split())
SEMANTIC_PROTECTED = frozenset('not without before after preserve fail failed error exception timeout'.split())


class FilteredCodeBM25:
    def __init__(self, documents, *, static=True, high_df=None, k1=1.5, b=.75):
        if high_df not in (None,.7,.8,.9) or k1 <= 0 or not 0 <= b <= 1:
            raise ValueError('Only the preregistered lexical settings are allowed')
        self.extractor = CodeAnchorExtractor()
        self.tokens = {uid:Counter(self.extractor.tokens(text)) for uid,text in documents.items()}
        self.protected = {uid:self._protected(text) for uid,text in documents.items()}
        self.static, self.high_df, self.k1, self.b = static, high_df, k1, b
        self.last_suppressed = []

    def _protected(self, text):
        protected = set(SEMANTIC_PROTECTED)
        for anchor in self.extractor.extract(text):
            protected.add(anchor.normalized)
            protected.update(anchor.components)
            protected.update(re.findall(r'\b\w+\b', anchor.raw.casefold()))
        return protected

    def search(self, query, *, candidate_uids=None, top_k=10):
        if top_k <= 0 or not query.strip():
            return []
        ids = sorted(self.tokens if candidate_uids is None else set(candidate_uids) & self.tokens.keys())
        if not ids:
            return []
        # Corpus statistics and corpus-wide anchor protection use only eligible
        # histories. Queries may protect their own tokens but never change DF.
        protected = self._protected(query) | set().union(*(self.protected[uid] for uid in ids))
        dfs = Counter(token for uid in ids for token in self.tokens[uid])
        suppressed = (set(SAFE_STOPWORDS) if self.static else set())
        if self.high_df is not None:
            suppressed.update(term for term,count in dfs.items() if count/len(ids) > self.high_df)
        suppressed -= protected
        self.last_suppressed = sorted(suppressed)
        corpus = [{t:c for t,c in self.tokens[uid].items() if t not in suppressed} for uid in ids]
        lengths = [sum(tf.values()) for tf in corpus]
        average = sum(lengths)/len(ids) or 1.
        terms = Counter(t for t in self.extractor.tokens(query) if t not in suppressed)
        scores = []
        for uid,tf,length in zip(ids,corpus,lengths):
            value = sum(count*math.log(1+(len(ids)-dfs[term]+.5)/(dfs[term]+.5))*tf[term]*(self.k1+1)
                        /(tf[term]+self.k1*(1-self.b+self.b*length/average)) for term,count in terms.items() if term in tf)
            if value > 0:
                scores.append((uid,value))
        return sorted(scores,key=lambda pair:(-pair[1],pair[0]))[:top_k]
