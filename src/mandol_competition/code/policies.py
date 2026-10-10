import re

from mandol_substrate.contracts import PublicationPolicy
from mandol_substrate.state import RetrievalProfile
from mandol_substrate.state.models import digest
from .software.anchors import CodeAnchorExtractor

PUBLICATION = PublicationPolicy("code-canonical-lexical-optional-dense-v1", False, False)


class CodeProcessor:
    def __init__(self, code_aware=False):
        self.code_aware = code_aware
        self.fingerprint = digest({"schema": "code-lexical-v1", "code_aware": code_aware})
        self.tokenize = CodeAnchorExtractor().tokens if code_aware else lambda s: re.findall(r"\b\w+\b", s.casefold())

    def document(self, text):
        return " ".join(self.tokenize(text))

    query = document


def retrieval_profile(processor, inference_contract="vllm-http-v1"):
    if inference_contract=="aml-official-eligible-v1":
        from dataclasses import asdict
        from mandol_competition.inference.dashscope import official_embedding_recipes
        dense=official_embedding_recipes()[0]
        return RetrievalProfile(dense.model_id,dense.model_revision,dense.dimension,
            dense.tokenizer_revision,"disabled","disabled",processor.fingerprint,
            bm25_profile="code-lexical-v1",methods=("bm25","cosine"),
            options={"lexical_statistics":"candidate-subset-v1","lexical_minimum_score":0.0,
                "ranking":"engineering-episodes-source-ranked-v1","sparse_channel":"disabled",
                "inference_contract":inference_contract,"dense_recipe":asdict(dense),
                "dense_recipe_hash":dense.hash})
    if inference_contract!="vllm-http-v1":raise ValueError("Unknown Code inference contract")
    return RetrievalProfile("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        1024, "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "disabled", "disabled",
        processor.fingerprint, bm25_profile="code-lexical-v1", methods=("bm25", "cosine"),
        options={"lexical_statistics": "candidate-subset-v1", "lexical_minimum_score": 0.0,
                 "ranking": "engineering-episodes-source-ranked-v1", "sparse_channel": "disabled"})
