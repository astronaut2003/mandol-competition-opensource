from dataclasses import asdict, replace

import pytest

from mandol_substrate.contracts import DerivedNamespaceBinding, ResourceIdentity
from mandol_substrate.state import ProfileMismatch, Scope, SQLiteStateStore
from mandol_substrate.state.models import digest


def binding(identity, name="derived:explicit"):
    return DerivedNamespaceBinding(name, identity.namespace, identity.hash,
                                   digest("recipe-v1"), "summary")


def test_exact_authorization_restart_and_unchanged_resource_hash(tmp_path, profile, scope):
    identity = ResourceIdentity("textual", scope.namespace, "textual_test")
    frozen = asdict(identity)
    path = tmp_path / "state.sqlite3"
    state = SQLiteStateStore(path, profile, identity=identity)
    derived = Scope("derived:explicit", scope.user_id)
    try:
        with pytest.raises(ProfileMismatch):
            state.ensure_scope(derived)
        state.authorize_derived_namespace(binding(identity))
        state.ensure_scope(scope)
        state.ensure_scope(derived)
        state.ensure_scope(Scope(derived.namespace, "another-user"))
        assert set(state.owned_user_scopes(scope.user_id)) == {scope, derived}
        assert asdict(identity) == frozen and identity.hash == digest(frozen)
        for wrong in ("hlm:" + "a" * 64, "view:" + "b" * 64, "code", "multimodal"):
            with pytest.raises(ProfileMismatch):
                state.ensure_scope(Scope(wrong, scope.user_id))
        with pytest.raises(ProfileMismatch):
            state.authorize_derived_namespace(replace(binding(identity), recipe_hash=digest("different")))
        with pytest.raises(ProfileMismatch):
            state.authorize_derived_namespace(binding(ResourceIdentity("code", "code", "code_test")))
    finally:
        state.close()
    reopened = SQLiteStateStore(path, profile, identity=identity)
    try:
        reopened.ensure_scope(derived)
        assert reopened.profile_hash == profile.hash
    finally:
        reopened.close()


def test_restart_rejects_corrupt_authorization_and_does_not_adopt_existing_scope(tmp_path, profile, scope):
    identity = ResourceIdentity("textual", scope.namespace, "textual_test")
    path = tmp_path / "state.sqlite3"
    state = SQLiteStateStore(path, profile, identity=identity)
    state.authorize_derived_namespace(binding(identity))
    state.ensure_scope(Scope("derived:explicit", scope.user_id))
    with state.transaction() as db:
        db.execute("DELETE FROM manifest WHERE key LIKE 'derived_namespace:%'")
    state.close()
    with pytest.raises(ProfileMismatch):
        SQLiteStateStore(path, profile, identity=identity)


def test_unbound_compatibility_remains_explicit(tmp_path, profile):
    state = SQLiteStateStore(tmp_path / "state.sqlite3", profile)
    try:
        state.ensure_scope(Scope("old-research", "u"))
        with pytest.raises(ProfileMismatch):
            state.authorize_derived_namespace(binding(ResourceIdentity("t", "t", "c")))
    finally:
        state.close()
