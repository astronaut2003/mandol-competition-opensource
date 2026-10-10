import pytest
from mandol_substrate.representation import MemoryUnit
from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
from mandol_competition.code.software.experience_retrieval import EpisodeRetriever,EpisodeRetrievalConfig


def retriever(*,distinctive):
    sources={}
    episodes=[]
    for i,text in enumerate(['ValueError in src/normal.py','ValueError in src/special.py','ValueError in src/other.py']):
        u=MemoryUnit(uid=str(i),raw_data={'text_content':text},metadata={'user_id':'u','session_id':str(i),'role':'user'})
        sources[u.uid]=u
        episodes.append(EngineeringEpisodeBuilder().build([u],user_id='u',session_id=str(i)))
    class Channel:
        def search(self,*args,**kwargs):return [('0',1.),('1',.9),('2',.8)]
    return EpisodeRetriever(episodes,sources,user_id='u',raw_channel=Channel(),
        config=EpisodeRetrievalConfig(anchor_weight=.25,distinctive_anchors=distinctive))


def test_ubiquitous_error_name_adds_no_distinctive_support():
    r=retriever(distinctive=True)
    hits=r.search('ValueError',3)
    assert all(h.features['anchor_overlap']==0 for h in hits)
    assert [h.episode.source_uids[0] for h in hits]==['0','1','2']


def test_rare_exact_source_path_supports_without_inserting_candidates():
    r=retriever(distinctive=True)
    hits=r.search('src/special.py',3)
    assert hits[0].episode.source_uids==['1']
    assert hits[0].features['anchor_overlap']==pytest.approx(.5)
    assert {h.episode.source_uids[0] for h in hits}=={'0','1','2'}
    assert all(0<=h.features['anchor_overlap']<=1 for h in hits)


def test_baseline_default_and_no_anchor_remain_raw_ranked():
    assert EpisodeRetrievalConfig().distinctive_anchors is False
    assert [h.episode.source_uids[0] for h in retriever(distinctive=True).search('general requirement',3)]==['0','1','2']
