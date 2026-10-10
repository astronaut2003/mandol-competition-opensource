from types import SimpleNamespace

import pytest

from mandol_competition.textual.core.evidence_navigation import EvidenceFamilyPacker, NavigationPolicy
from mandol_substrate.state import CanonicalMemory, Scope


@pytest.mark.parametrize("window,requested,expected", [(10,100,10), (50,100,50), (100,100,100), (100,1,1)])
def test_candidate_navigation_obeys_declared_window_and_requested_top_k(window, requested, expected):
    class Context:
        selected_k = window
        def render(self, scope, hits, canonical): return hits
    class Returns:
        final_k = window
        max_context_chars = 24000 if window == 10 else 100000
        def materialize(self, hits, top_k): return hits[:min(self.final_k, top_k)]
    policy = NavigationPolicy(("profile",), version="canonical_raw_navigation_v1"
                              if window == 10 else "canonical_raw_navigation_candidate_v2")
    packer = EvidenceFamilyPacker(policy, Context(), Returns())
    scope = Scope("textual", "u")
    canonical = {f"u{i:03}": CanonicalMemory(f"u{i:03}", "Raw source evidence " + str(i),
                     metadata={"user_id": "u"}, commit_seq=1) for i in range(100)}
    ordered = [(uid, 1. - i / 100) for i, uid in enumerate(canonical)]
    assert len(packer.pack(scope, ordered, canonical, requested_top_k=requested, published_seq=1)) == expected
