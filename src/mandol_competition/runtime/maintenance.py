"""Offline maintenance of a claimed unified runtime; API writer must be stopped.

No legacy DB adoption, collection deletion, model loading or automatic reset.
"""
import argparse
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


def maintain(track, operation, *, user_id=None):
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
    cleanup = lambda scope: None
    encoder_factory = None
    if track == 'textual':
        from mandol_competition.textual.core.profile import load_profile
        from mandol_competition.textual.core.remote_inference import remote_encoders
        from mandol_substrate.text import MandolTextProcessor
        profile, processor = load_profile(cfg.root / 'profile.json'), MandolTextProcessor()
        encoder_factory = lambda: remote_encoders(profile, cfg.inference['embedding'], cfg.inference['sparse'], endpoint_policy='http')
    elif track == 'code':
        from mandol_competition.code.policies import CodeProcessor, retrieval_profile
        from mandol_competition.code.api.config import Settings
        from mandol_competition.code.api.remote_embeddings import RemoteDenseEncoder
        from mandol_competition.code.projection import DenseEncoders, POLICY
        processor = CodeProcessor(Settings.from_env().coding_code_aware)
        profile, projection_policy = retrieval_profile(processor), POLICY
        encoder_factory = lambda: DenseEncoders(RemoteDenseEncoder(cfg.inference['embedding']))
    else:
        from mandol_competition.multimodal.config import get_config
        from mandol_competition.multimodal.models import load_profile, build_processor
        from mandol_competition.multimodal.inference import remote_encoders
        config = replace(get_config(), data_dir=cfg.root, namespace=cfg.namespace,
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
        state.integrity_check()
        scopes = list(state.iter_scopes())
        canonical = True
        for scope in scopes:
            state.lexical.verify_scope(scope)
            seq = state.scope_state(scope)
            canonical &= seq['canonical_seq'] == seq['published_seq'] == seq['lexical_seq']
        try:
            projection = QdrantProjectionBackend(cfg.qdrant_endpoint, profile, state=state, storage_id=state.db_uuid,
                identity=identity, collection_name=cfg.collection, api_key=cfg.qdrant_api_key,
                policy=projection_policy, optional_projection=not policy.required_projection)
            projection.explicit_rebuild = operation == 'rebuild'
            projection.ensure_schema(Scope(cfg.namespace, '__maintenance__'), create=operation == 'rebuild')
        except ProfileMismatch:
            raise
        except Exception:
            if operation != 'verify' or policy.required_projection:
                raise
            return {'canonical_servable': bool(canonical), 'projection_servable': False,
                    'required_projection': False, 'projection_error': 'Unavailable'}
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
                elif not coordinator.project_optional(scope):
                    raise StateUnavailable('optional projection repair remains degraded')
                projection.gc_scope(scope)
        elif operation == 'gc':
            for scope in scopes:
                purger.complete(scope)
                projection.gc_scope(scope)
        report = verify_projection(state, projection)
        return {'canonical_servable': not state.pending_count(), 'required_projection': policy.required_projection,
                'projection_servable': report['servable'], 'projection': report}
    finally:
        if encoders is not None:
            if hasattr(encoders, 'close'): encoders.close()
            elif hasattr(encoders, 'encoder'): encoders.encoder.close()
        if projection is not None: projection.close()
        state.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('track', choices=['textual', 'code', 'multimodal'])
    parser.add_argument('operation', choices=['verify', 'gc', 'reconcile', 'rebuild', 'purge'])
    parser.add_argument('--user-id-file', help='UTF-8 exact user ID; no stripping or logging')
    args = parser.parse_args()
    from pathlib import Path
    from mandol_competition.observability.runtime import configure_logging
    configure_logging('WARNING')
    try:
        result = maintain(args.track, args.operation,
                          user_id=Path(args.user_id_file).read_text() if args.user_id_file else None)
        print(json.dumps(result, sort_keys=True))
        return 0 if result['canonical_servable'] and (not result['required_projection'] or result['projection_servable']) else 1
    except Exception as exc:
        # DB paths, user identities and upstream bodies never enter the report.
        print(json.dumps({'error_class': type(exc).__name__, 'success': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
