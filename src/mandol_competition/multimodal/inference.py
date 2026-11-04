"""HTTP-only MM recipe bindings; frozen and explicit BF16 candidates coexist."""
from dataclasses import replace
from mandol_competition.inference.models import InferenceRecipe
from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPRerankerClient, HTTPEncoders, HTTPJointEncoders


def recipes(profile):
    o = profile.options
    frozen = (
        InferenceRecipe('multimodal-dense-fp32-v1', 'embedding', profile.embedding_model,
            profile.embedding_model, profile.embedding_revision, profile.tokenizer_fingerprint,
            dimension=profile.embedding_dimension, normalization='l2', pooling='sentence-transformer-default',
            query_prompt=o.get('dense_query_prompt', ''), max_length=o.get('dense_max_length', 4096)),
        InferenceRecipe('multimodal-splade-fp32-v1', 'sparse', profile.splade_model,
            profile.splade_model, profile.splade_revision, profile.splade_revision, dimension=30522,
            pooling='max-log1p-relu', max_length=o.get('splade_max_length', 512), sparse_threshold=0.0),
        InferenceRecipe('multimodal-bge-fp32-v1', 'reranker', o.get('reranker_model', 'BAAI/bge-reranker-v2-m3'),
            o.get('reranker_model', 'BAAI/bge-reranker-v2-m3'), o.get('reranker_revision', 'fixture'),
            o.get('reranker_revision', 'fixture'), max_length=o.get('reranker_max_length', 4096),
            pooling='cross-encoder-default', score_semantics='cross-encoder-default-higher-is-relevant'))
    if o.get('inference_recipe') == 'aml-official-eligible-v1':
        from mandol_competition.inference.dashscope import MODEL, DIMENSION, official_embedding_recipes
        if profile.embedding_model != MODEL or profile.splade_model != MODEL or profile.embedding_dimension != DIMENSION:
            raise ValueError('official embedding profile requires the actual text-embedding-v4 model')
        return (*official_embedding_recipes(), replace(frozen[2],
            recipe_id=o.get('reranker_recipe_id', 'multimodal-vllm-bge-bf16-v1'),
            dtype=o.get('reranker_dtype', 'bfloat16'),
            pooling='cls-cross-encoder', truncation='pair-right-max-length',
            score_semantics='sigmoid-higher-is-relevant'))
    if o.get('inference_recipe') in ('multimodal-vllm-bf16-v1', 'multimodal-vllm-deployment-v1'):
        # A new numerical/runtime recipe, never advertised as bit-identical to
        # SentenceTransformers. SPLADE keeps all-token max-log1p-relu pooling.
        result = (
            replace(frozen[0], recipe_id='multimodal-vllm-dense-bf16-v1', dtype='bfloat16',
                pooling='last-token', truncation='right-max-length'),
            replace(frozen[1], recipe_id='multimodal-vllm-splade-bf16-v1', dtype='bfloat16',
                pooling='all-token-max-log1p-relu', truncation='right-max-length'),
            replace(frozen[2], recipe_id='multimodal-vllm-bge-bf16-v1', dtype='bfloat16',
                pooling='cls-cross-encoder', truncation='pair-right-max-length',
                score_semantics='sigmoid-higher-is-relevant'),
        )
        if o.get('inference_recipe') == 'multimodal-vllm-deployment-v1':
            return tuple(replace(r, recipe_id=o['deployment_id'] + '-' + r.channel,
                dtype=o[field]) for r, field in zip(result,
                    ('dense_dtype', 'splade_dtype', 'reranker_dtype'), strict=True))
        return result
    return frozen


def remote_encoders(config, profile):
    dense, sparse, _ = recipes(profile)
    factory = (HTTPJointEncoders if profile.options.get('inference_recipe') == 'aml-official-eligible-v1'
               else ParallelHTTPEncoders if profile.options.get("parallel_http_encoding", False) else HTTPEncoders)
    return factory(HTTPEmbeddingClient(config.embedding_endpoint, dense, batch_size=1, timeout=config.inference_timeout),
                        HTTPSparseEncoderClient(config.sparse_endpoint, sparse, timeout=config.inference_timeout))


def remote_reranker(config, profile):
    from mandol_competition.inference.replicas import replica_endpoints, create_pool
    from .search_profiles import QUERIT, QWEN37, descriptor
    descriptor(profile, config.search_policy)
    endpoints = replica_endpoints('multimodal', config.reranker_endpoint,
                                 config.reranker_pool.get('endpoints'))
    if config.search_policy == QWEN37:
        from mandol_competition.inference.dashscope_rerank import from_environment
        replicas = [from_environment(endpoint, credential_env=config.reranker_credential_env,
                                     timeout=config.inference_timeout) for endpoint in endpoints]
    elif config.search_policy == QUERIT:
        from mandol_competition.textual.core.remote_inference import RemoteQueritClient
        from mandol_competition.textual.core.querit_reference import expected_profile
        replicas = [RemoteQueritClient(endpoint, expected_profile(batch_size=1),
                                      timeout=config.inference_timeout, endpoint_policy='deployment')
                    for endpoint in endpoints]
    else:
        replicas = [HTTPRerankerClient(endpoint, recipes(profile)[2], timeout=config.inference_timeout)
                    for endpoint in endpoints]
    client = create_pool(replicas, config.reranker_pool)
    if 'post_rerank_policy' in profile.options:
        from .reranking import RankBlendedReranker
        try:
            return RankBlendedReranker(client, profile.options['post_rerank_policy'])
        except BaseException:
            client.close()
            raise
    return client


class ParallelHTTPEncoders(HTTPEncoders):
    """Two independent HTTP channels overlap; model math and ordering are unchanged.

    No SQLite work runs in these threads. Both calls finish before publication
    can proceed, including on failure. The pool is bounded and owned by the API.
    """
    def __init__(self, dense, sparse):
        super().__init__(dense, sparse)
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='mm-http-encoding')

    def _pair(self, dense_call, sparse_call):
        from contextvars import copy_context
        from concurrent.futures import wait
        dense = self._pool.submit(copy_context().run, dense_call)
        sparse = self._pool.submit(copy_context().run, sparse_call)
        wait((dense, sparse))
        return dense.result(), sparse.result()

    def encode_documents(self, texts):
        dense, sparse = self._pair(lambda: self.dense.embed(texts),
                                   lambda: self.sparse.encode_documents(texts))
        return list(zip(dense, sparse, strict=True))

    def encode_query(self, text):
        return self._pair(lambda: self.dense.embed([text], query=True)[0],
                          lambda: self.sparse.encode_query(text))

    def close(self):
        self._pool.shutdown(wait=True)
        super().close()
