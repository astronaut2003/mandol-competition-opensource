"""Independent production ASGI factories; run one factory per single-worker process."""
from dataclasses import replace
import os
from mandol_competition.config import load_config
from .storage import claim_runtime_root


def _deadlines(cfg, old):
    if cfg.runtime.mode == 'legacy':
        return {}
    return {op + '_timeout_seconds': cfg.runtime.limits.get(op + '_timeout', getattr(old, op + '_timeout_seconds'))
            for op in ('add', 'search')}


def _settings(track):
    settings = load_config()
    from mandol_competition.observability.runtime import configure_logging
    configure_logging(settings.log_level)
    cfg = settings.track(track)
    if not cfg.memory_system_key and not cfg.allow_unauthenticated_dev:
        raise ValueError('Memory System Key required outside explicit development mode')
    if track == 'multimodal' and cfg.runtime.mode != 'legacy':
        from mandol_competition.config.unified import secret_from_environment
        from mandol_competition.inference.generation import generation_route
        route = generation_route(cfg.inference.get('vlm'), 'gpt-4o-mini',
                                 credential_env=cfg.runtime.vlm_credential_env)
        if not secret_from_environment(route.credential_env):
            raise ValueError('explicit VLM credential source required for frozen M0')
    claim_runtime_root(cfg.root, track)
    from mandol_competition.inference.deployment import claim_serving_identity
    claim_serving_identity(cfg.root)
    return cfg


def textual():
    from mandol_competition.textual.api.app import create_app
    from mandol_competition.textual.api.config import Settings
    cfg = _settings('textual')
    old = Settings.from_env(bind_run=False) if cfg.runtime.mode == 'legacy' else Settings()
    app = create_app(replace(old, competition_run_root=str(cfg.root), evaluation_namespace=cfg.namespace,
        memory_system_key=cfg.memory_system_key, allow_unauthenticated_dev=cfg.allow_unauthenticated_dev,
        qdrant_mode='external', qdrant_endpoint=cfg.qdrant_endpoint, qdrant_collection=cfg.collection,
        qdrant_api_key=cfg.qdrant_api_key, inference_contract=cfg.inference_contract,
        embedding_url=cfg.inference['embedding'], splade_url=cfg.inference['sparse'], reranker_url=cfg.inference['reranker'],
        reranker_pool=cfg.reranker_pool, experimental_quality_profile=cfg.quality_candidate,
        evidence_profile=cfg.evidence_profile,
        reranker_backend=cfg.reranker_backend, reranker_credential_env=cfg.reranker_credential_env,
        **_deadlines(cfg, old)))
    from mandol_competition.observability.qualification import install_qualification_trace
    return install_qualification_trace(app, 'textual')


def code():
    from mandol_competition.code.api.app import create_app
    from mandol_competition.code.api.config import Settings
    cfg = _settings('code')
    old = Settings.from_env() if cfg.runtime.mode == 'legacy' else Settings()
    app = create_app(replace(old, coding_db_path=str(cfg.root / 'state.sqlite3'), namespace=cfg.namespace,
        memory_system_key=cfg.memory_system_key, qdrant_endpoint=cfg.qdrant_endpoint, qdrant_collection=cfg.collection,
        qdrant_api_key=cfg.qdrant_api_key, coding_memory=True, coding_dense_model_path=None,
        coding_dense_base_url=cfg.inference.get('embedding'),inference_contract=cfg.inference_contract,
        coding_raw_policy='c0-original-v1' if cfg.retrieval_policy == 'baseline' else cfg.retrieval_policy,
        coding_reranker_backend=cfg.reranker_backend,
        coding_reranker_endpoint=cfg.inference.get('reranker', old.coding_reranker_endpoint),
        coding_reranker_credential_env=cfg.reranker_credential_env,
        **_deadlines(cfg, old)))
    from mandol_competition.observability.qualification import install_qualification_trace
    return install_qualification_trace(app, 'code')


def multimodal():
    from mandol_competition.multimodal.api import create_app
    from mandol_competition.multimodal.config import get_config, MultimodalConfig
    cfg = _settings('multimodal')
    old = get_config() if cfg.runtime.mode == 'legacy' else MultimodalConfig()
    topology = {} if cfg.runtime.mode == 'legacy' else dict(vlm_backend='openai_compat',
        vlm_model='gpt-4o-mini-closeai', vlm_request_model='gpt-4o-mini',
        vlm_api_key_env=cfg.runtime.vlm_credential_env)
    adapted = replace(old, data_dir=cfg.root, namespace=cfg.namespace, auth_token=cfg.memory_system_key or None,
        env='dev' if cfg.allow_unauthenticated_dev else 'production',
        vlm_backend=old.vlm_backend if cfg.allow_unauthenticated_dev else 'openai_compat',
        vlm_base_url=cfg.inference.get('vlm', old.vlm_base_url),
        qdrant_url=cfg.qdrant_endpoint, qdrant_collection=cfg.collection, qdrant_api_key=cfg.qdrant_api_key,
        inference_contract=cfg.inference_contract,
        search_policy='m0-original-v1' if cfg.retrieval_policy == 'baseline' else cfg.retrieval_policy,
        evidence_policy=os.getenv('MANDOL_MM_EVIDENCE_POLICY', old.evidence_policy),
        vision_failure_policy=os.getenv('MANDOL_MM_VISION_FAILURE_POLICY', old.vision_failure_policy),
        prompt_version=os.getenv('MANDOL_MM_PROMPT_VERSION', old.prompt_version),
        vlm_output_tokens=int(os.getenv('MANDOL_MM_VLM_OUTPUT_TOKENS', old.vlm_output_tokens)),
        vlm_max_output_tokens=int(os.getenv('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS', old.vlm_max_output_tokens)),
        model_device='external-http' if cfg.inference_contract=='aml-official-eligible-v1' else old.model_device,
        embedding_endpoint=cfg.inference['embedding'], sparse_endpoint=cfg.inference['sparse'], reranker_endpoint=cfg.inference['reranker'],
        reranker_pool=cfg.reranker_pool, reranker_credential_env=cfg.reranker_credential_env,
        **_deadlines(cfg, old))
    return create_app(replace(adapted, **topology))
