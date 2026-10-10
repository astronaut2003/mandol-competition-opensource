from dataclasses import replace

from mandol_competition.textual.core.hlm_production import CanonicalHighLevelMemory, HighLevelMemoryPolicy, ProductionHLMRecipe
from mandol_competition.textual.core.heterogeneous_worker import typed_scope
from mandol_substrate.state import BuildJob, CanonicalMemory, Edge, Mutation
from tests.common.test_publication import coordinator


def test_historical_view_keeps_superseded_evidence_but_hides_retractions_without_rewriting_truth(store, profile, scope, memory, tmp_path):
    c, _ = coordinator(store, profile, tmp_path)
    old = replace(memory, text_content="I live in Beijing.", metadata={"user_id": scope.user_id}, role="user")
    new = replace(old, logical_uid="new-raw", text_content="I have moved to Shanghai.")
    c.apply(scope, "source", "source", Mutation(memories=(old, new)), {})
    recipe = ProductionHLMRecipe(profile.hash, policy=HighLevelMemoryPolicy(profile_enabled=True))
    partition = typed_scope(scope, "profile", recipe)
    roots, derived = [], []
    for raw, uid in ((old, "old-profile"), (new, "new-profile")):
        root = BuildJob.planned(scope, {raw.logical_uid: 1}, recipe.fingerprint)
        roots.append(replace(root, status="completed", progress={"result": {"output_uids": {"profile": [uid]}}}))
        derived.append(CanonicalMemory(uid, raw.text_content, memory_level="L2_Profile", metadata={
            "user_id": scope.user_id, "raw_namespace": scope.namespace, "recipe_hash": recipe.fingerprint,
            "hlm_kind": "profile", "hlm_job_id": root.job_id, "source_versions": {raw.logical_uid: 1},
            "operation": "ADD", "hlm_item": {"profile_item": {"status": "current"}, "evidence": [{
                "source_uid": raw.logical_uid, "quote": raw.text_content}]}}))
    c.apply(scope, "roots", "roots", Mutation(jobs=tuple(roots)), {})
    supersedes = Edge("edge-supersedes", "new-profile", "old-profile", "PROFILE_SUPERSEDES")
    c.apply(partition, "profiles", "profiles", Mutation(memories=tuple(derived), edges=(supersedes,)), {})
    runtime = CanonicalHighLevelMemory(c, recipe, lambda _: None)
    try:
        memories = store.get_memories(partition, [m.logical_uid for m in derived])
        assert set(runtime.eligible(scope, memories)) == {"new-profile"}
        history = runtime.eligible_history(scope, memories)
        assert set(history) == {"old-profile", "new-profile"}
        assert history["old-profile"].metadata["hlm_item"]["profile_item"]["status"] == "historical"
        assert memories["old-profile"].metadata["hlm_item"]["profile_item"]["status"] == "current"
        assert store.get_memories(partition, ["old-profile"])["old-profile"].metadata == memories["old-profile"].metadata
        retracts = Edge("edge-retracts", "new-profile", "old-profile", "PROFILE_RETRACTS")
        c.apply(partition, "retract", "retract", Mutation(edges=(retracts,)), {})
        assert set(runtime.eligible_history(scope, memories)) == {"new-profile"}
    finally:
        runtime.close()
