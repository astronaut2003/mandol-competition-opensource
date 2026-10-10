"""Independent production ASGI factories; run one factory per single-worker process."""
from dataclasses import replace
from mandol_competition.config import load_config
from .storage import claim_runtime_root


def _settings(track):
    settings = load_config()
    from mandol_competition.observability.runtime import configure_logging
    configure_logging(settings.log_level)
    cfg = settings.track(track)
    if not cfg.memory_system_key and not cfg.allow_unauthenticated_dev:
        raise ValueError('Memory System Key required outside explicit development mode')
    claim_runtime_root(cfg.root, track)
    from mandol_competition.inference.deployment import claim_serving_identity
    claim_serving_identity(cfg.root)
    return cfg


def textual():
    from mandol_competition.textual.api.app import create_app
    from mandol_competition.textual.api.config import Settings
    cfg = _settings('textual')
    old = Settings.from_env(bind_run=False)
    app = create_app(replace(old, competition_run_root=str(cfg.root), evaluation_namespace=cfg.namespace,
        memory_system_key=cfg.memory_system_key, allow_unauthenticated_dev=cfg.allow_unauthenticated_dev,
        qdrant_mode='external', qdrant_endpoint=cfg.qdrant_endpoint, qdrant_collection=cfg.collection,
        qdrant_api_key=cfg.qdrant_api_key, inference_contract=cfg.inference_contract,
        embedding_url=cfg.inference['embedding'], splade_url=cfg.inference['sparse'], reranker_url=cfg.inference['reranker']))
    from mandol_competition.observability.qualification import install_qualification_trace
    return install_qualification_trace(app, 'textual')


def code():
    from mandol_competition.code.api.app import create_app
    from mandol_competition.code.api.config import Settings
    cfg = _settings('code')
    old = Settings.from_env()
    app = create_app(replace(old, coding_db_path=str(cfg.root / 'state.sqlite3'), namespace=cfg.namespace,
        memory_system_key=cfg.memory_system_key, qdrant_endpoint=cfg.qdrant_endpoint, qdrant_collection=cfg.collection,
        qdrant_api_key=cfg.qdrant_api_key, coding_memory=True, coding_dense_model_path=None,
        coding_dense_base_url=cfg.inference.get('embedding'),inference_contract=cfg.inference_contract))
    from mandol_competition.observability.qualification import install_qualification_trace
    return install_qualification_trace(app, 'code')


def multimodal():
    from mandol_competition.multimodal.api import create_app
    from mandol_competition.multimodal.config import get_config
    cfg = _settings('multimodal')
    old = get_config()
    return create_app(replace(old, data_dir=cfg.root, namespace=cfg.namespace, auth_token=cfg.memory_system_key or None,
        env='dev' if cfg.allow_unauthenticated_dev else 'production',
        vlm_backend=old.vlm_backend if cfg.allow_unauthenticated_dev else 'openai_compat',
        vlm_base_url=cfg.inference.get('vlm', old.vlm_base_url),
        qdrant_url=cfg.qdrant_endpoint, qdrant_collection=cfg.collection, qdrant_api_key=cfg.qdrant_api_key,
        inference_contract=cfg.inference_contract,
        model_device='external-http' if cfg.inference_contract=='aml-official-eligible-v1' else old.model_device,
        embedding_endpoint=cfg.inference['embedding'], sparse_endpoint=cfg.inference['sparse'], reranker_endpoint=cfg.inference['reranker']))
