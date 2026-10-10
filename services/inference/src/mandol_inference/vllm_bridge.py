"""HTTP-only recipe bridge to native vLLM pooling endpoints. No model loaders.

The bridge applies the declared query prompt once and translates wire schemas.
Tokenization, pooling, normalization and model execution remain inside vLLM.
SPLADE uses vLLM's native BertSpladeSparseEmbeddingModel with explicitly null
CLS/SEP removal IDs, retaining special tokens in max-log1p-relu pooling.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mandol_substrate.state import StateUnavailable

from mandol_competition.inference.clients import ordered, finite_vector
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.multimodal.config import get_config
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes

from .multimodal import EncodeRequest, ScoreRequest


def create_app(*, profile=None, dense_http=None, sparse_http=None, reranker_http=None,
               channels=('embedding','sparse','reranker')):
    if not channels or len(set(channels))!=len(channels) or not set(channels)<={'embedding','sparse','reranker'}:
        raise ValueError('explicit supported inference capabilities required')
    profile = profile or load_profile(get_config())
    recipe_name=profile.options.get('inference_recipe')
    if (recipe_name not in ('multimodal-vllm-bf16-v1', 'multimodal-vllm-deployment-v1')
            and not (recipe_name=='aml-official-eligible-v1' and set(channels)=={'reranker'})):
        raise ValueError('the vLLM bridge requires its own versioned recipe')
    dense, sparse, reranker = recipes(profile)
    supplied = dict(embedding=dense_http,sparse=sparse_http,reranker=reranker_http)
    endpoint_env = dict(embedding='MANDOL_VLLM_EMBEDDING_ENDPOINT',
                        sparse='MANDOL_VLLM_SPARSE_ENDPOINT',reranker='MANDOL_VLLM_RERANKER_ENDPOINT')
    clients = {channel:supplied[channel] or HTTPTransport(os.environ[endpoint_env[channel]],timeout=300)
               for channel in channels}
    active_recipes = tuple(r for r in (dense,sparse,reranker) if r.channel in clients)
    expected_version = profile.options.get('vllm_version')
    ready = False

    native = profile.options.get('native_serving', {})

    def binding(recipe):
        from mandol_competition.inference.deployment import bindings
        spec = bindings().get(recipe.served_alias, {}) | native.get(recipe.channel, {})
        if (spec.get('dtype', recipe.dtype) != recipe.dtype
                or spec.get('revision', recipe.model_revision) != recipe.model_revision):
            raise ValueError('serving binding differs from frozen model recipe')
        return spec

    def alias(recipe):
        return binding(recipe).get('served_alias', recipe.served_alias)

    def admit(client, recipe):
        spec = binding(recipe)
        if client.request('GET', '/version').get('version') != spec.get('vllm_version', expected_version):
            raise ValueError('vLLM version mismatch')
        rows = client.request('GET', '/v1/models').get('data', [])
        found = [r for r in rows if r.get('id') == alias(recipe)]
        root_ok = len(found) == 1 and (found[0].get('root') == spec['root'] if 'root' in spec
            else found[0].get('root', '').rstrip('/').endswith('/' + recipe.model_revision))
        if not root_ok:
            raise ValueError('vLLM alias/revision snapshot mismatch')

    @asynccontextmanager
    async def lifespan(app):
        nonlocal ready
        try:
            for recipe in active_recipes:
                admit(clients[recipe.channel], recipe)
            ready = True
            yield
        finally:
            ready = False
            for client in clients.values():
                client.close()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(StateUnavailable)
    async def unavailable(request, exc):
        return JSONResponse(status_code=503, content={'detail': 'inference response unavailable or invalid'})

    def check(body, recipe):
        if recipe.channel not in clients:
            raise HTTPException(404, 'capability not served by this bridge')
        if not ready:
            raise HTTPException(503, 'native serving unavailable')
        if body.model != recipe.served_alias or body.recipe_hash != recipe.hash:
            raise HTTPException(409, 'recipe identity mismatch')

    @app.get('/health')
    def health():
        if not ready:
            raise HTTPException(503, 'native serving unavailable')
        for item in active_recipes:
            clients[item.channel].health()
            admit(clients[item.channel], item)
        return {'status': 'ok'}


    @app.get('/v1/recipe')
    def recipe():
        # Server-declared metadata plus native alias/version probes. Weight and
        # precision proof still requires the deployment command/artifact record.
        for item in active_recipes:
            admit(clients[item.channel], item)
        return {'recipes': [{'recipe': asdict(r), 'hash': r.hash} for r in active_recipes],
                'vllm_version': 'per-channel' if native else expected_version,
                'native_serving': {k:v for k,v in native.items() if k in clients},
                'attestation': 'declared-recipe-with-native-identity-probes'}

    @app.post('/v1/embeddings')
    def embeddings(body: EncodeRequest):
        check(body, dense)
        if body.input_type not in ('query', 'document') or body.encoding_format != 'float':
            raise HTTPException(422, 'unsupported encoding policy')
        prompt = dense.query_prompt if body.input_type == 'query' else dense.document_prompt
        result = clients['embedding'].request('POST', '/v1/embeddings', {
            'model': alias(dense), 'input': [prompt + text for text in body.input],
            'encoding_format': 'float', 'truncate_prompt_tokens': dense.max_length,
            'truncation_side': 'right'})
        values = ordered(result, len(body.input), 'embedding', alias(dense))
        for value in values:
            finite_vector(value, dense.dimension)
        return {'model': dense.served_alias, 'data': [{'index': i, 'embedding': v} for i, v in enumerate(values)]}

    @app.post('/v1/score')
    def score(body: ScoreRequest):
        check(body, reranker)
        if not body.documents:
            return {'model': reranker.served_alias, 'data': []}
        result = clients['reranker'].request('POST', '/score', {
            'model': alias(reranker), 'text_1': body.query, 'text_2': body.documents,
            'truncate_prompt_tokens': reranker.max_length, 'truncation_side': 'right',
            'use_activation': True})
        values = ordered(result, len(body.documents), 'score', alias(reranker))
        finite_vector(values, len(body.documents))
        return {'model': reranker.served_alias, 'data': [{'index': i, 'score': v} for i, v in enumerate(values)]}

    @app.post('/v1/sparse')
    def sparse_encode(body: EncodeRequest):
        check(body, sparse)
        route = binding(sparse).get('path', '/v1/embeddings')
        payload = {'model': alias(sparse), 'input': body.input, 'encoding_format': 'float',
            'truncate_prompt_tokens': sparse.max_length, 'truncation_side': 'right'}
        if route == '/pooling':
            payload.update(task='embed', add_special_tokens=True)
        result = clients['sparse'].request('POST', route, payload)
        values = ordered(result, len(body.input), 'data' if route == '/pooling' else 'embedding', alias(sparse))
        converted = []
        for value in values:
            finite_vector(value, sparse.dimension)
            if any(v < 0 for v in value):
                raise HTTPException(503, 'invalid SPLADE activation')
            converted.append({str(i): v for i, v in enumerate(value) if v > sparse.sparse_threshold})
        return {'model': sparse.served_alias, 'data': [{'index': i, 'sparse': v} for i, v in enumerate(converted)]}
    from .observability import install_bridge_observability
    install_bridge_observability(app, ready=health, transports=lambda: clients)
    return app


def create_reranker_app():
    """BGE capability without accidental Qwen/SPLADE lifecycle dependencies."""
    return create_app(channels=('reranker',))
