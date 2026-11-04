"""Offline maintenance of a claimed unified runtime; API writer must be stopped.

No legacy DB adoption, collection deletion, model loading or automatic reset.
"""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import shutil
import sqlite3

from mandol_substrate.contracts import PublicationPolicy, ResourceIdentity, SpacePolicy
from mandol_substrate.maintenance import ResourcePurger
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.state import SQLiteStateStore, Scope, ProfileMismatch, StateUnavailable
from mandol_substrate.vector.verification import verify_projection
from mandol_competition.config import load_config
from .storage import claim_runtime_root


def maintain(track, operation, *, user_id=None, projection_batch_size=None):
    if projection_batch_size is not None and (operation != 'prepare-dense'
            or type(projection_batch_size) is not int or not 1 <= projection_batch_size <= 128):
        raise ValueError('projection batch size is a bounded Code prepare-dense option only')
    if operation == 'prepare-dense':
        if track != 'code' or not isinstance(user_id, str) or not user_id:
            raise ValueError('prepare-dense requires Code and an exact existing user identity')
    elif user_id is not None and operation != 'purge':
        # Historically this argument was silently ignored by reconcile/gc.
        # Never turn an intended single-user repair into a whole-corpus write.
        raise ValueError('user identity is only supported by purge and Code prepare-dense')
    cfg = load_config().track(track)
    path = cfg.root / 'state.sqlite3'
    if not path.is_file() or not (cfg.root / 'unified-runtime.json').is_file():
        raise ValueError('maintenance requires an existing claimed unified runtime')
    claim_runtime_root(cfg.root, track)
    # Read bindings without changing the database; the writer lock below is still mandatory.
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        manifest = dict(db.execute('SELECT key,value FROM manifest'))
    space = json.loads(manifest['space_policy'])
    spaces = SpacePolicy(tuple(space['spaces']), tuple(map(tuple, space['edges'])))
    policy = PublicationPolicy(**json.loads(manifest['publication_policy']))
    identity = ResourceIdentity(track, cfg.namespace, cfg.collection)
    projection_policy = QdrantProjectionPolicy()
    projection_factory = QdrantProjectionBackend
    projection_channel = 'dense'
    extension = None
    cleanup = lambda scope: None
    encoder_factory = None
    if track == 'textual':
        from mandol_competition.textual.core.profile import load_profile
        from mandol_competition.textual.core.remote_inference import remote_encoders
        from mandol_substrate.text import MandolTextProcessor
        profile, processor = load_profile(cfg.root / 'profile.json'), MandolTextProcessor()
        if cfg.inference_contract == 'aml-official-eligible-v1':
            from mandol_competition.inference.clients import HTTPEmbeddingClient, HTTPSparseEncoderClient, HTTPJointEncoders
            from mandol_competition.inference.dashscope import official_embedding_recipes
            dense, sparse = official_embedding_recipes()
            encoder_factory = lambda: HTTPJointEncoders(
                HTTPEmbeddingClient(cfg.inference['embedding'], dense),
                HTTPSparseEncoderClient(cfg.inference['sparse'], sparse))
        else:
            encoder_factory = lambda: remote_encoders(profile, cfg.inference['embedding'], cfg.inference['sparse'], endpoint_policy='http')
    elif track == 'code':
        from mandol_competition.code.policies import CodeProcessor, retrieval_profile
        from mandol_competition.code.api.config import Settings
        from mandol_competition.code.api.remote_embeddings import RemoteDenseEncoder, RecipeDenseEncoder, RecipeNativeEncoder
        from mandol_competition.code.projection import DenseEncoders, POLICY
        from mandol_competition.code.native_sparse import BASELINE, POLICY as NATIVE_POLICY, FRONTIER, NativeProjection, descriptor
        settings = Settings.from_env() if cfg.runtime.mode == 'legacy' else Settings()
        processor = CodeProcessor(settings.coding_code_aware)
        profile, projection_policy = retrieval_profile(processor, cfg.inference_contract), POLICY
        if cfg.retrieval_policy == NATIVE_POLICY:
            if cfg.inference_contract != 'aml-official-eligible-v1':
                raise ValueError('Code Native maintenance requires the named-v4 inference contract')
            projection_factory, projection_channel = NativeProjection, FRONTIER
            extension = descriptor(profile)
            encoder_factory = lambda: RecipeNativeEncoder(cfg.inference['embedding'])
        elif cfg.retrieval_policy in {'baseline', BASELINE}:
            encoder = RecipeDenseEncoder if cfg.inference_contract == 'aml-official-eligible-v1' else RemoteDenseEncoder
            encoder_factory = lambda: DenseEncoders(encoder(cfg.inference['embedding']))
        else:
            raise ValueError('Unknown Code maintenance retrieval policy')
    else:
        from mandol_competition.multimodal.config import get_config, MultimodalConfig
        from mandol_competition.multimodal.models import load_profile, build_processor
        from mandol_competition.multimodal.inference import remote_encoders
        old = get_config() if cfg.runtime.mode == 'legacy' else MultimodalConfig()
        config = replace(old, data_dir=cfg.root, namespace=cfg.namespace,
                         inference_contract=cfg.inference_contract,
                         model_device='external-http' if cfg.inference_contract == 'aml-official-eligible-v1' else old.model_device,
                         embedding_endpoint=cfg.inference['embedding'], sparse_endpoint=cfg.inference['sparse'])
        profile = load_profile(config)
        processor = build_processor(profile)
        encoder_factory = lambda: remote_encoders(config, profile)
        def cleanup(scope):
            root = config.user_root(scope.user_id)
            if root.exists(): shutil.rmtree(root)
    state = SQLiteStateStore(path, profile, processor=processor, identity=identity, space_policy=spaces)
    projection = encoders = None
    try:
        if extension is not None:
            state.bind_manifest('code_native_sparse_extension_v1', extension)
        state.integrity_check()
        if operation == 'prepare-dense':
            scope = Scope(cfg.namespace, user_id)
            # Missing identity is an error, not permission to create a tenant.
            state.scope_state(scope)
            state.assert_not_purging(scope)
            scopes = [scope]
        else:
            scopes = list(state.iter_scopes())
        canonical = True
        for scope in scopes:
            state.lexical.verify_scope(scope)
            seq = state.scope_state(scope)
            canonical &= seq['canonical_seq'] == seq['published_seq'] == seq['lexical_seq']
        try:
            projection = projection_factory(cfg.qdrant_endpoint, profile, state=state, storage_id=state.db_uuid,
                identity=identity, collection_name=cfg.collection, api_key=cfg.qdrant_api_key,
                policy=projection_policy, optional_projection=not policy.required_projection)
            projection.explicit_rebuild = operation == 'rebuild'
            projection.ensure_schema(scopes[0] if operation == 'prepare-dense' else Scope(cfg.namespace, '__maintenance__'),
                create=operation in {'rebuild', 'prepare-dense'})
        except ProfileMismatch:
            raise
        except Exception:
            if operation != 'verify' or policy.required_projection:
                raise
            return {'canonical_servable': bool(canonical), 'projection_servable': False,
                    'required_projection': False, 'projection_error': 'Unavailable',
                    'projection_channel': projection_channel}
        if operation == 'prepare-dense':
            if policy.required_projection or not canonical:
                raise StateUnavailable('Code warmup requires acknowledged canonical and lexical state')
            encoders = encoder_factory()
            coordinator = PublicationCoordinator(state, projection, projection.codec, encoders, policy=policy,
                batch_size=projection_batch_size or 128)
            # No global recover(), iter_scopes(), purge, GC or full verifier.
            # project_optional verifies the entire selected scope before its
            # single frontier is ready, and reuses existing immutable vectors.
            was_ready = state.projection_ready(scope, projection_channel)
            available = coordinator.project_optional(scope, channel=projection_channel)
            return {'canonical_servable': True, 'required_projection': False,
                'projection_servable': available, 'scope_only': True,
                'projection_batch_size': coordinator.batch_size,
                'was_ready': was_ready, 'published_seq': state.scope_state(scope)['published_seq'],
                'projection_frontier_ready': state.projection_ready(scope, projection_channel),
                'projection_channel': projection_channel}
        purger = ResourcePurger(state, projection, cleanup=cleanup)
        if operation == 'purge':
            if not user_id: raise ValueError('purge requires an exact user identity')
            scope = Scope(cfg.namespace, user_id)
            state.ensure_scope(scope)
            state.begin_purge(scope, {'purged': True})
            purger.complete(scope)
        elif operation in {'reconcile', 'rebuild'}:
            encoders = encoder_factory()
            coordinator = PublicationCoordinator(state, projection, projection.codec, encoders, policy=policy)
            for scope in scopes: purger.complete(scope)
            coordinator.recover()
            for scope in scopes:
                if policy.required_projection:
                    coordinator.audit_scope(scope)
                elif not coordinator.project_optional(scope, channel=projection_channel):
                    raise StateUnavailable('optional projection repair remains degraded')
                projection.gc_scope(scope)
        elif operation == 'gc':
            for scope in scopes:
                purger.complete(scope)
                projection.gc_scope(scope)
        report = verify_projection(state, projection)
        return {'canonical_servable': not state.pending_count(), 'required_projection': policy.required_projection,
                'projection_servable': report['servable'], 'projection': report,
                'projection_channel': projection_channel}
    finally:
        with ExitStack() as cleanup:
            cleanup.callback(state.close)
            if projection is not None:
                cleanup.callback(projection.close)
            if encoders is not None:
                if hasattr(encoders, 'close'):
                    cleanup.callback(encoders.close)
                elif hasattr(encoders, 'encoder'):
                    cleanup.callback(encoders.encoder.close)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=['textual', 'code', 'multimodal'])
    parser.add_argument('operation', choices=['verify', 'gc', 'reconcile', 'rebuild', 'purge', 'prepare-dense'])
    parser.add_argument('--user-id-file', help='UTF-8 exact user ID; no stripping or logging')
    parser.add_argument('--projection-batch-size',type=int,help='Code prepare-dense only: warm the configured Dense/native-pair policy, at most this many sources per batch (1..128)')
    args = parser.parse_args()
    from pathlib import Path
    from mandol_competition.observability.runtime import configure_logging
    configure_logging('WARNING')
    try:
        extra = {'projection_batch_size': args.projection_batch_size} if args.projection_batch_size is not None else {}
        result = maintain(args.track, args.operation,
                          user_id=Path(args.user_id_file).read_text() if args.user_id_file else None,**extra)
        print(json.dumps(result, sort_keys=True))
        needs_projection = result['required_projection'] or args.operation == 'prepare-dense'
        return 0 if result['canonical_servable'] and (not needs_projection or result['projection_servable']) else 1
    except Exception as exc:
        # DB paths, user identities and upstream bodies never enter the report.
        print(json.dumps({'error_class': type(exc).__name__, 'success': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
