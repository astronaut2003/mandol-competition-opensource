"""Desired versioned recipes; observed serving attestation is reported separately."""
from dataclasses import replace
from pathlib import Path
from .models import InferenceRecipe


def textual_recipes():
    from mandol_competition.textual.core.profile import load_profile
    profile = load_profile(Path(__file__).parents[1] / 'textual/profiles/source_b1_public_v1/profile.json')
    dense, ranker = profile.options['embedding_serving'], profile.options['reranker_serving']
    return (
        InferenceRecipe('textual-qwen-dense-v1', 'embedding', 'qwen3-embedding-0.6b',
            profile.embedding_model, profile.embedding_revision, dense['tokenizer_revision'],
            dimension=1024, dtype=dense['dtype'], normalization='l2', pooling='frozen-vllm-pooling',
            max_length=4096, truncation='vllm-truncate-prompt-tokens-4096', contract='vllm-0.29-http-v1'),
        InferenceRecipe('textual-splade-v3-v1', 'sparse', 'splade-v3', profile.splade_model,
            profile.splade_revision, profile.splade_revision, dimension=30522, dtype='float32',
            pooling='max-log1p-relu', max_length=512, sparse_threshold=0, contract='vllm-0.29-http-v1'),
        InferenceRecipe('textual-querit-4b-v1', 'reranker', 'querit-4b', ranker['model_name'],
            ranker['model_revision'], ranker['tokenizer_revision'], dtype=ranker['dtype'],
            pooling='official-final-cls', max_length=40960, truncation='reject-over-complete-pair-budget-v1',
            query_prompt='official-mteb-default-prompt-final-cls-v1', score_semantics='probs[1]-probs[0]',
            contract='vllm-0.29-http-v1'))


def code_recipe():
    # Same served weights, independently preserved float64 norm -> float32 matrix policy.
    return replace(textual_recipes()[0], recipe_id='code-qwen-dense-v1',
                   normalization='float64-l2-then-float32', contract='code-openai-embeddings-v1')
