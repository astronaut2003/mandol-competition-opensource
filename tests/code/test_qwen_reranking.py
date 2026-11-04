"""Real scratch SQLite and fake HTTP scoring; not a real-model qualification."""
from dataclasses import replace
import json

import pytest

from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.config import Settings
from mandol_competition.code.reranking import CodeEpisodeReranker, CANDIDATE_K
from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
from mandol_competition.code.software.experience_retrieval import EpisodeRetriever, EvidencePacker
from mandol_competition.inference.dashscope_rerank import BACKEND
from mandol_substrate.representation import MemoryUnit
from mandol_substrate.state import ProfileMismatch, StateUnavailable
from mandol_competition.inference.errors import InferenceUnavailable, InferenceRateLimited, InferenceDeadlineExceeded


class Scorer:
    def __init__(self, bad=None):
        self.bad, self.calls, self.closed = bad, [], 0
    def score(self, query, docs):
        self.calls.append((query, list(docs)))
        if isinstance(self.bad, Exception):
            raise self.bad
        if self.bad is not None:
            return self.bad
        return [(i + 1) / len(docs) for i in range(len(docs))]
    def close(self):
        self.closed += 1


def candidates(n=40):
    units = [MemoryUnit(uid=f'raw:{i}', raw_data={'text_content':f'  JWT regression evidence {i}\n'},
        metadata=dict(user_id='u', session_id=f's{i}', role='user', sequence=0)) for i in range(n)]
    sources = {u.uid:u for u in units}
    episodes = [EngineeringEpisodeBuilder().build([u], user_id='u', session_id=u.metadata['session_id']) for u in units]
    return sources, EpisodeRetriever(episodes, sources, user_id='u').search('JWT', top_k=CANDIDATE_K)


def test_reranking_scores_all_32_existing_candidates_and_preserves_exact_source_cards():
    sources, hits = candidates();scorer=Scorer();ranker=CodeEpisodeReranker(scorer)
    before = {h.episode.episode_id:EvidencePacker(max_cards=1).pack([h], sources, query='JWT', top_k=1)[0]['content'] for h in hits}
    original_scores = [h.score for h in hits]
    value = ranker.rank('JWT', hits, sources, user_id='u')
    assert len(scorer.calls[0][1]) == len(hits) == 32
    assert [h.episode.episode_id for h in value.hits] == [h.episode.episode_id for h in reversed(hits)]
    assert [h.score for h in hits] == original_scores and value.hits[0].score == 1.
    for h in value.hits:
        assert EvidencePacker(max_cards=1).pack([h], sources, query='JWT', top_k=1)[0]['content'] == before[h.episode.episode_id]
    assert not value.fallback and ranker.stats()['successes'] == 1


@pytest.mark.parametrize('bad', [TimeoutError('provider down'), InferenceUnavailable('http_503',status_code=503),
                               [float('nan')], [-.1], [True], [.2, .4]])
def test_optional_failure_preserves_exact_original_packing_and_does_not_retry(bad):
    sources,hits=candidates(1);scorer=Scorer(bad);ranker=CodeEpisodeReranker(scorer)
    value=ranker.rank('JWT', hits, sources, user_id='u')
    assert value.fallback and len(scorer.calls) == 1
    assert EvidencePacker().pack(value.hits, sources, query='JWT', top_k=100) == EvidencePacker().pack(hits, sources, query='JWT', top_k=100)
    assert ranker.stats()['fallbacks'] == 1


def test_bad_provenance_or_foreign_scope_is_never_sent_to_provider():
    sources,hits=candidates(2);scorer=Scorer();ranker=CodeEpisodeReranker(scorer)
    assert ranker.rank('JWT', [], sources, user_id='u').hits == [] and not scorer.calls
    with pytest.raises(ProfileMismatch):
        ranker.rank('JWT', hits, sources, user_id='other')
    sources[next(iter(sources))].raw_data['text_content']='forged source'
    with pytest.raises(ValueError):
        ranker.rank('JWT', hits, sources, user_id='u')
    assert not scorer.calls


def test_service_add_replay_restart_reranker_failure_recovery_and_rollback(tmp_path, monkeypatch):
    scorer=Scorer()
    import mandol_competition.inference.dashscope_rerank as provider
    monkeypatch.setattr(provider,'from_environment',lambda *a,**k:scorer)
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),inference_contract='aml-official-eligible-v1',
        coding_raw_policy='code-native-sparse-rrf100-v1',coding_reranker_backend=BACKEND)
    service=CodingMemoryService(cfg);service.start()
    def add():
        for i in range(3):service.add_messages('u',f's{i}',f'r{i}',[dict(role='user',content=f'  JWT issuer fixed in auth/file{i}.py\n',timestamp=10+i)])
    try:
        add();add();assert not scorer.calls
        uuid=service.store.state.db_uuid;profile=service.store.profile.hash
        assert len(service.store.snapshot('u')[1]) == 3
        assert service.search('absent','JWT',100) == [] and not scorer.calls
        assert service.search('u','',100) == [] and not scorer.calls
        cards=service.search('u','JWT',100)
        assert len(cards)==3 and len(scorer.calls)==1
        scope=service.store.scope('u')
        scorer.bad=TimeoutError('unavailable')
        fallback=service.search('u','JWT',100)
        assert service.health() and service.stats()['reranker']['fallbacks']==1
        assert service.store.state._rows("SELECT 1 FROM runtime_events WHERE scope=? AND channel='reranker'",(scope.key,))
        scorer.bad=None
        assert service.search('u','JWT',100)==cards
        assert len(service.search('u','JWT',1))==1
        assert service.store.state.pending_count()==0
    finally:service.shutdown()
    assert scorer.closed==1
    rollback=CodingMemoryService(replace(cfg,coding_reranker_backend='profile'));rollback.start()
    try:
        assert rollback.store.state.db_uuid==uuid and rollback.store.profile.hash==profile
        assert rollback.search('u','JWT',100)==fallback
        assert all(json.loads(r['payload'])['content'].startswith('  JWT') for r in rollback.store.snapshot('u')[1])
    finally:rollback.shutdown()
    resumed=CodingMemoryService(cfg);resumed.start()
    try:assert resumed.search('u','JWT',100)==cards
    finally:resumed.shutdown()


def test_dense_and_reranker_can_fail_independently_without_blocking_add(tmp_path, monkeypatch):
    scorer=Scorer(TimeoutError('unavailable'))
    import mandol_competition.inference.dashscope_rerank as provider
    import mandol_competition.code.projection as projection
    monkeypatch.setattr(provider,'from_environment',lambda *a,**k:scorer)
    def unavailable(*args,**kwargs):raise StateUnavailable('fake unavailable projection')
    monkeypatch.setattr(projection,'prepare_channel',unavailable)
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),inference_contract='aml-official-eligible-v1',
        coding_raw_policy='code-native-sparse-rrf100-v1',coding_reranker_backend=BACKEND)
    service=CodingMemoryService(cfg);service.start();service._encoder=object()
    try:
        service.add_messages('u','s','r',[dict(role='user',content='JWT auth/middleware.py regression passed')])
        assert service.search('u','JWT',100) and service.health()
        assert service.stats()['last_retrieval_path']=='LEXICAL_FALLBACK'
        assert service.stats()['reranker']['fallbacks']==1
        scorer.bad=None
        assert service.search('u','JWT',100)
        assert service.stats()['reranker']['successes']==1
    finally:service.shutdown()


def test_concurrent_users_keep_fallback_events_in_the_failed_scope(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import mandol_competition.inference.dashscope_rerank as provider
    barrier=threading.Barrier(2)
    class Parallel(Scorer):
        def score(self, query, docs):
            barrier.wait(timeout=3)
            if 'broken' in query:raise TimeoutError('injected provider error')
            return [1.] * len(docs)
    scorer=Parallel();monkeypatch.setattr(provider,'from_environment',lambda *a,**k:scorer)
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),inference_contract='aml-official-eligible-v1',coding_reranker_backend=BACKEND)
    service=CodingMemoryService(cfg);service.start()
    try:
        for user in ['failed','healthy']:
            service.add_messages(user,'s','r',[dict(role='user',content='JWT issuer evidence '+user)])
        with ThreadPoolExecutor(max_workers=2) as pool:
            left=pool.submit(service.search,'failed','JWT broken',100)
            right=pool.submit(service.search,'healthy','JWT fixed',100)
            assert left.result() and right.result()
        rows=service.store.state._rows("SELECT scope FROM runtime_events WHERE channel='reranker'")
        assert [r['scope'] for r in rows]==[service.store.scope('failed').key]
        assert service.stats()['reranker']['successes']==service.stats()['reranker']['fallbacks']==1
    finally:service.shutdown()


def test_shutdown_drains_inflight_reranking_before_closing_http_and_sqlite(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import mandol_competition.inference.dashscope_rerank as provider
    entered=threading.Event();release=threading.Event()
    class Slow(Scorer):
        def score(self, query, docs):
            entered.set();assert release.wait(3)
            assert self.closed==0
            return [1.]*len(docs)
    scorer=Slow();monkeypatch.setattr(provider,'from_environment',lambda *a,**k:scorer)
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),inference_contract='aml-official-eligible-v1',coding_reranker_backend=BACKEND)
    service=CodingMemoryService(cfg);service.start()
    service.add_messages('u','s','r',[dict(role='user',content='JWT issuer evidence')])
    with ThreadPoolExecutor(max_workers=2) as pool:
        searching=pool.submit(service.search,'u','JWT',100);assert entered.wait(3)
        stopping=pool.submit(service.shutdown)
        try:assert not stopping.done() and scorer.closed==0
        finally:release.set()
        assert searching.result();stopping.result()
    assert scorer.closed==1 and not service.ready


def test_failed_reranker_manifest_binding_closes_its_new_http_client(tmp_path, monkeypatch):
    import mandol_competition.inference.dashscope_rerank as provider
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),inference_contract='aml-official-eligible-v1')
    original=CodingMemoryService(cfg);original.start()
    original.store.state.bind_manifest('code_qwen37_reranker_extension_v1',dict(incompatible='fixture'))
    original.shutdown()
    scorer=Scorer();monkeypatch.setattr(provider,'from_environment',lambda *a,**k:scorer)
    broken=CodingMemoryService(replace(cfg,coding_reranker_backend=BACKEND))
    try:
        with pytest.raises(ProfileMismatch):broken.start()
    finally:broken.shutdown()
    assert scorer.closed==1
    restored=CodingMemoryService(cfg);restored.start();restored.shutdown()


@pytest.mark.parametrize('stage',['rerank','projection','raw_query'])
@pytest.mark.parametrize('failure',[InferenceRateLimited(retry_after=60),InferenceDeadlineExceeded()])
def test_actual_http_code_search_cannot_hide_retry_signals_in_optional_fallback(tmp_path,monkeypatch,stage,failure):
    from fastapi.testclient import TestClient
    from mandol_competition.code.api.app import create_app
    import mandol_competition.inference.dashscope_rerank as provider
    import mandol_competition.code.projection as projection
    scorer=Scorer(failure if stage=='rerank' else None)
    monkeypatch.setattr(provider,'from_environment',lambda *args,**kwargs:scorer)
    cfg=Settings(coding_db_path=str(tmp_path/'state.sqlite3'),memory_system_key='fixture',
        inference_contract='aml-official-eligible-v1',qdrant_collection='code_aml_v4_v1',coding_reranker_backend=BACKEND)
    service=CodingMemoryService(cfg)
    with TestClient(create_app(cfg,memory_service=service),raise_server_exceptions=False) as client:
        import time
        until=time.monotonic()+5
        while client.get('/health').status_code!=200:
            assert time.monotonic()<until
            time.sleep(.01)
        headers={'X-Api-Key':'fixture'}
        assert client.post('/add',headers=headers,json=dict(user_id='artificial',session_id='s',request_id='r',
            messages=[dict(role='user',content='JWT issuer synthetic evidence')])).status_code==200
        if stage=='projection':
            service._encoder=object()
            def unavailable(*args,**kwargs):raise failure
            monkeypatch.setattr(projection,'prepare_channel',unavailable)
        elif stage=='raw_query':
            service._encoder=object()
            class Unavailable:
                def search(self,*args,**kwargs):raise failure
            monkeypatch.setattr(projection,'prepare_channel',lambda *args,**kwargs:Unavailable())
        response=client.post('/search',headers=headers,json=dict(user_id='artificial',query='JWT',top_k=100))
        assert response.status_code==(429 if isinstance(failure,InferenceRateLimited) else 503)
        if isinstance(failure,InferenceRateLimited):assert response.headers['Retry-After']=='60'
        assert service.stats()['reranker']['fallbacks']==0
        assert service.store.state._rows('SELECT count(*) n FROM requests')[0]['n']==1
        assert service.store.state._rows('SELECT count(*) n FROM memories')[0]['n']==1
