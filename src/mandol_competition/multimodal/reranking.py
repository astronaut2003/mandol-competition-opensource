"""Opt-in MM ranking policy; native model transport and recipes stay unchanged."""
import math

from mandol_competition.inference.clients import finite_vector


def blend_scores(scores, *, weight=2., k=60):
    """Combine input fusion order with native relevance ranks.

    Equal native scores share their minimum rank. This lets the retrieval prior
    resolve a model tie rather than inventing relevance from a memory ID.
    """
    values = finite_vector(scores, len(scores))
    ranks = {}
    for index, value in enumerate(sorted(values, reverse=True), 1):
        ranks.setdefault(value, index)
    return [1/(k+index) + weight/(k+ranks[value])
            for index, value in enumerate(values, 1)]


class RankBlendedReranker:
    def __init__(self, client, policy):
        if (not isinstance(policy, dict)
                or set(policy) != {'version', 'weight', 'k'}
                or policy['version'] != 'mm-post-rerank-rrf-v1'
                or policy['k'] != 60
                or type(policy['weight']) not in (int, float)
                or not math.isfinite(policy['weight']) or not 0 < policy['weight'] <= 4):
            raise ValueError('unsupported MM post-rerank policy')
        self.client, self.policy = client, dict(policy)
        self.recipe = client.recipe

    def score(self, query, documents):
        if not documents:
            return []
        scores = self.client.score(query, documents)
        finite_vector(scores, len(documents))
        return blend_scores(scores, weight=self.policy['weight'], k=self.policy['k'])

    def validate_profile(self):
        return self.client.validate_profile()

    def health(self):
        return self.client.health()

    def attestation(self):
        return self.client.attestation() | {'application_post_rerank_policy': self.policy,
            'application_score_semantics': 'weighted reciprocal ranks; higher is relevant; not a probability'}

    def close(self):
        self.client.close()
