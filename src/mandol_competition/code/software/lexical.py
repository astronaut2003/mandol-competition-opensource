"""Code tokenization policy over the common lexical snapshot primitive."""
import re
from mandol_substrate.lexical.snapshot import Bm25Snapshot
from .anchors import CodeAnchorExtractor

class CodeBM25(Bm25Snapshot):
    def __init__(self, documents, *, code_aware=True, k1=1.5, b=.75):
        tokenize = CodeAnchorExtractor().tokens if code_aware else lambda text: re.findall(r"\b\w+\b", text.casefold())
        super().__init__(documents, tokenize=tokenize, k1=k1, b=b)
