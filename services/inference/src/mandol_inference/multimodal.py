"""Serve the original MM FP32 recipe through the common HTTP contract.

This worker deliberately preserves the old numerical implementation. It is not
implicitly replaced by a differently pooled/quantized vLLM model. T/Code's
existing vLLM endpoints remain the primary Dense/reranker deployment path.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict
import math
import os
from pathlib import Path
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes


class EncodeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str
    recipe_hash: str
    input: list[str] = Field(min_length=1, max_length=128)
    input_type: str = 'document'
    encoding_format: str = 'float'


class ScoreRequest(BaseModel):
    model: str
    recipe_hash: str
    query: str
    documents: list[str] = Field(max_length=1024)


def create_app(*, profile=None, encoders=None, reranker=None):
    path = os.getenv('MANDOL_MM_RETRIEVAL_PROFILE')
    profile = profile or load_profile(MultimodalConfig(retrieval_profile_path=Path(path) if path else None))
    recipe_set = recipes(profile)
    lock = threading.Lock()
    holders = {'encoders': encoders, 'reranker': reranker, 'ready': False}

    @asynccontextmanager
    async def lifespan(app):
        if holders['encoders'] is None or holders['reranker'] is None:
            from mandol_competition.multimodal.compat.frozen_models import LocalEncoders, LocalReranker
            holders['encoders'] = holders['encoders'] or LocalEncoders(profile, device=profile.options['model_device'])
            holders['reranker'] = holders['reranker'] or LocalReranker(profile, device=profile.options['model_device'])
        # Immutable cached snapshots only: missing artifacts fail worker startup.
        holders['encoders'].encode_query('memory worker readiness')
        holders['reranker'].score('memory', ['memory worker readiness'])
        holders['ready'] = True
        try:
            yield
        finally:
            holders['ready'] = False
            for obj in (holders['encoders'], holders['reranker']):
                if hasattr(obj, 'close'): obj.close()

    app = FastAPI(lifespan=lifespan)
    def check(body, recipe):
        if body.model != recipe.served_alias or body.recipe_hash != recipe.hash:
            raise HTTPException(409, 'recipe identity mismatch')
        if not holders['ready']:
            raise HTTPException(503, 'worker unavailable')
    def response(recipe, field, values):
        return {'model': recipe.served_alias, 'data': [{'index': i, field: value} for i, value in enumerate(values)]}

    @app.get('/health')
    def health():
        if not holders['ready']: raise HTTPException(503, 'worker unavailable')
        return {'status': 'ok'}

    @app.get('/v1/recipe')
    def recipe():
        return {'recipes': [{'recipe': asdict(r), 'hash': r.hash} for r in recipe_set],
                'runtime': profile.options.get('runtime'), 'semantic_profile_hash': profile.hash}

    @app.post('/v1/embeddings')
    def embedding(body: EncodeRequest):
        check(body, recipe_set[0])
        if body.input_type not in ('query', 'document') or body.encoding_format != 'float':
            raise HTTPException(422, 'unsupported encoding policy')
        with lock:
            values = holders['encoders'].encode_dense(body.input, query=body.input_type == 'query')
        return response(recipe_set[0], 'embedding', values)

    @app.post('/v1/sparse')
    def sparse(body: EncodeRequest):
        check(body, recipe_set[1])
        with lock:
            # The frozen joint encoder returns identical sparse query/document vectors.
            values = holders['encoders'].encode_sparse(body.input)
        return response(recipe_set[1], 'sparse', values)

    @app.post('/v1/score')
    def score(body: ScoreRequest):
        check(body, recipe_set[2])
        with lock:
            values = holders['reranker'].score(body.query, body.documents)
        if len(values) != len(body.documents) or any(not math.isfinite(v) for v in values):
            raise HTTPException(503, 'invalid model output')
        return response(recipe_set[2], 'score', values)
    return app
