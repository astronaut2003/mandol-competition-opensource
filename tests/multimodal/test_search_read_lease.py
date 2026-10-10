from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

from .backend_support import query, request, services
from .test_sqlite_qdrant import image_part
from mandol_competition.multimodal.protocol import SearchRequest
from mandol_competition.multimodal.vlm.base import Extraction


def test_same_user_text_searches_overlap_and_purge_waits_for_both(backend):
    add, search = services(backend)
    add.handle(request())
    entered = Barrier(3)
    release = Event()
    original = backend.search

    def held(*args, **kwargs):
        entered.wait(timeout=5)
        assert release.wait(timeout=5)
        return original(*args, **kwargs)

    backend.search = held
    with ThreadPoolExecutor(max_workers=3) as pool:
        first = pool.submit(search.handle, query())
        second = pool.submit(search.handle, query())
        try:
            entered.wait(timeout=5)  # Both real Search operations hold read leases.
            purge = pool.submit(backend.purge, 'alice')
            assert not purge.done()
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
        assert purge.result(timeout=10)['units_removed'] == 1
    assert backend._operations.active_scopes == 0


def test_image_search_preserves_exclusive_cache_and_single_paid_sequence(backend):
    entered, release = Event(), Event()

    class Vision:
        model_name = 'image-lease-fixture'
        calls = 0

        def extract(self, **kwargs):
            self.calls += 1
            entered.set()
            assert release.wait(timeout=5)
            return Extraction(description='red backpack',model=self.model_name,prompt_version='v1',schema_version='1')

    vision = Vision()
    add, search = services(backend, vision)
    add.handle(request())
    image_query = SearchRequest(query=[image_part()],user_id='alice',top_k=10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(search.handle, image_query)
        assert entered.wait(timeout=5)
        second = pool.submit(search.handle, image_query)
        try:
            assert not second.done()
            assert vision.calls == 1
        finally:
            release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
    assert vision.calls == 1
