import random
import pytest
from mandol_competition.code.api.coding_store import CodingEvidenceStore
from mandol_competition.code.software.lexical import CodeBM25
from mandol_competition.code.projection import CanonicalLexicalChannel

@pytest.mark.parametrize('code_aware', [False, True])
def test_sqlite_bm25_matches_frozen_code_for_filtered_candidates(tmp_path, code_aware):
    rng = random.Random(42)
    texts = [' '.join(rng.choices(['alpha', 'Beta', 'src/x.py', 'APIError', 'foo_bar'], k=rng.randint(1, 30))) for _ in range(45)]
    store = CodingEvidenceStore(tmp_path / 'state.sqlite3', code_aware=code_aware)
    store.initialize()
    store.add('u', 's', 'r', [{'role': 'user', 'content': t} for t in texts])
    docs = {m.logical_uid: m.text_content for batch in store.state.iter_memories(store.scope('u')) for m in batch}
    old = CodeBM25(docs, code_aware=code_aware)
    new = CanonicalLexicalChannel(store, 'u', docs)
    for candidates in [None, [], sorted(docs)[::3], sorted(docs)[::2]]:
        for query in ['alpha Beta', 'src/x.py APIError', 'foo_bar foo_bar', 'absent']:
            expected = old.search(query, top_k=100, candidate_uids=candidates)
            actual = new.search(query, top_k=100, candidate_uids=candidates)
            assert [u for u, _ in actual] == [u for u, _ in expected]
            assert [s for _, s in actual] == pytest.approx([s for _, s in expected], abs=1e-12)
    store.close()
