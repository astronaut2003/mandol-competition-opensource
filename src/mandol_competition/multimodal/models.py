"""Lazy, frozen inference adapters. No corpus, indexes, or recovery snapshots."""

import json

from mandol_substrate.state.models import ProfileMismatch, RetrievalProfile
from mandol_substrate.text import MandolTextProcessor

from .config import ConfigurationError


def load_profile(config):
    if config.retrieval_profile_path is None:
        from pathlib import Path
        from dataclasses import replace
        filename=("aml-official-v4-v1.json" if config.inference_contract=="aml-official-eligible-v1"
                  else "frozen-fp32-v1.json")
        config = replace(config, retrieval_profile_path=Path(__file__).with_name("profiles") / filename)
    values = json.loads(config.retrieval_profile_path.read_text())
    values["methods"] = tuple(values["methods"])
    profile = RetrievalProfile(**values)
    if config.inference_contract=="aml-official-eligible-v1":
        from mandol_competition.inference.dashscope import MODEL,DIMENSION
        if (profile.options.get("inference_recipe")!="aml-official-eligible-v1"
                or profile.embedding_model!=MODEL or profile.splade_model!=MODEL
                or profile.embedding_dimension!=DIMENSION):
            raise ConfigurationError("Named-v4 contract cannot bind a research model profile")
    from .representation import BASELINE, POLICIES, QUERY_POLICIES
    if profile.options.get("representation_policy", BASELINE) not in POLICIES:
        raise ConfigurationError("unknown versioned representation policy")
    if profile.options.get("query_policy", "mm-query-v1") not in QUERY_POLICIES:
        raise ConfigurationError("unknown versioned query policy")
    methods = tuple("cosine" if m == "cosine_similarity" else m for m in config.retrieval_methods)
    if profile.methods != methods or profile.options.get("rerank_method") != config.rerank_method:
        raise ConfigurationError("retrieval methods/reranker differ from frozen profile")
    if profile.options.get("model_device") != config.model_device:
        raise ConfigurationError("model device differs from frozen profile")
    if profile.fusion != "rrf" or profile.rrf_k < 1:
        raise ConfigurationError("profile requires the supported positive RRF recipe")
    if config.embedding_model_name and (profile.embedding_model != config.embedding_model_name
                                       or profile.embedding_dimension != config.embedding_dim):
        raise ConfigurationError("embedding configuration differs from frozen profile")
    return profile


def build_processor(profile):
    return MandolTextProcessor(mode=profile.options.get("text_mode", "regex"),
                               use_jieba=profile.options.get("jieba", False))



def __getattr__(name):
    if name in {"LocalEncoders", "LocalReranker"}:
        from .compat import frozen_models
        return getattr(frozen_models, name)
    raise AttributeError(name)
