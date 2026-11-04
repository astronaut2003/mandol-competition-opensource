"""Search-only overlays; the immutable M0 storage/encoding profile is retained.

Changing the reranker never changes raw media, VLM preparation or vectors. The
effective search identity explicitly combines the storage hash and this policy;
it must not be advertised as the historical BGE M0 method.
"""
from mandol_substrate.state.models import digest

BASELINE = "m0-original-v1"
QUERIT = "mm-querit-same-pool-rank-only-v1"
QWEN37 = "mm-qwen37-same-pool-rank-only-v1"


def descriptor(profile, policy, evidence_policy="baseline"):
    if policy not in {BASELINE, QUERIT, QWEN37}:
        raise ValueError("Unknown MM search policy")
    result = dict(schema="multimodal-search-overlay-v1", profile_id=policy,
                  storage_profile_hash=profile.hash, candidate_pool="unchanged-m0",
                  fusion="unchanged-m0", score_filter="none", score_blend="none")
    if policy in {QUERIT, QWEN37}:
        from mandol_competition.textual.core.querit_reference import expected_profile
        if profile.options.get("inference_recipe") != "aml-official-eligible-v1":
            raise ValueError("MM Querit overlay requires the named-v4 M0 index")
        if profile.options.get("post_rerank_policy") or profile.options.get("rerank_method") != "baai":
            raise ValueError("MM Querit overlay requires the original, unblended M0 pool")
        from dataclasses import asdict
        from mandol_competition.inference.dashscope_rerank import recipe
        result.update(reranker=expected_profile(batch_size=1) if policy == QUERIT else asdict(recipe()),
                      scores="probs[1]-probs[0]" if policy == QUERIT else recipe().score_semantics,
                      ordering="score-descending-uid-ties", input="rendered-text-only")
    else:
        result.update(reranker=profile.options.get("reranker_model"))
    if evidence_policy == 'source-grounded-recall-v4':
        import hashlib
        from pathlib import Path
        from . import evidence_review as review
        from . import source_context as context
        result.update(evidence_policy=review.POLICY, selector_model='gpt-4o-mini',
            selector_prompt_sha256=hashlib.sha256(review.PRIMARY_PROMPT.encode()).hexdigest(),
            verifier_prompt_sha256=hashlib.sha256(review.VERIFY_PROMPT.encode()).hexdigest(),
            nonempty_results='baseline-ids-order-scores-with-attributed-source-context',
            empty_requires='complete-fragment-coverage-and-two-full-rejections',
            max_selector_calls=review.MAX_MODEL_CALLS, max_review_batches=review.MAX_BATCHES,
            max_review_fragments=review.MAX_FRAGMENTS, review_images_per_batch=review.MAX_BATCH_IMAGES,
            review_text_bytes_per_batch=review.MAX_BATCH_TEXT_BYTES,
            max_context_sources=context.MAX_CONTEXT_SOURCES,
            context_text_bytes=context.MAX_CONTEXT_TEXT_BYTES,
            media_allocation='reviewed-useful-sources-first-round-robin-v1',
            source_provenance=True, storage_migration=False,
            response_policy_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('evidence_review.py', 'source_context.py', 'grounded_delivery.py')})
    elif evidence_policy in {"source-grounded-atomic-v2", "source-grounded-delivery-v3"}:
        from .atomic_guard import (POLICY, MAX_SOURCES, MAX_REVIEW_IMAGES,
            REVIEW_TEXT_BYTES, REVIEW_URI_BYTES, PRIMARY_PROMPT, VERIFY_PROMPT)
        import hashlib
        result.update(evidence_policy=POLICY, selector_model="gpt-4o-mini",
            selector_prompt_sha256=hashlib.sha256(PRIMARY_PROMPT.encode()).hexdigest(),
            verifier_prompt_sha256=hashlib.sha256(VERIFY_PROMPT.encode()).hexdigest(),
            reviewed_sources=MAX_SOURCES, review_images=MAX_REVIEW_IMAGES,
            review_text_bytes=REVIEW_TEXT_BYTES, review_uri_bytes=REVIEW_URI_BYTES,
            nonempty_results="baseline-ids-order-scores", empty_requires="complete-review-and-two-rejections",
            max_selector_calls=2, source_provenance=True,
            duplicate_image_policy="reference-original-in-response")
        if evidence_policy == "source-grounded-delivery-v3":
            from .evidence_delivery import POLICY as DELIVERY_POLICY, SCHEMA, MAX_NOTE_BYTES
            from pathlib import Path
            result.update(evidence_policy=DELIVERY_POLICY, evidence_gate=POLICY,
                delivery_note_schema=SCHEMA, delivery_note_max_bytes=MAX_NOTE_BYTES,
                delivery_note_source_sha256=hashlib.sha256(
                    Path(__file__).with_name("evidence_delivery.py").read_bytes()).hexdigest(),
                delivery_note_trigger="image-omission-or-recorded-degraded-extraction")
    elif evidence_policy != "baseline":
        from .evidence import POLICY, MAX_CANDIDATES, MAX_EVIDENCE, MAX_SELECTOR_IMAGES, SELECTOR_TEXT_BYTES, SYSTEM
        import hashlib
        if evidence_policy != POLICY:
            raise ValueError("Unknown MM evidence policy")
        result.update(evidence_policy=POLICY, selector_model="gpt-4o-mini",
            selector_prompt_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest(),
            selector_candidates=MAX_CANDIDATES, evidence_limit=MAX_EVIDENCE,
            selector_images=MAX_SELECTOR_IMAGES, selector_text_bytes=SELECTOR_TEXT_BYTES,
            source_provenance=True, duplicate_image_policy="reference-original-in-response")
    return result


def identity(profile, policy, evidence_policy="baseline"):
    return digest(descriptor(profile, policy, evidence_policy))
