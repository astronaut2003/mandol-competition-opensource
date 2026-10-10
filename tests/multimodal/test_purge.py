import base64
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from PIL import Image

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.media_store import UserMediaStore
from mandol_competition.multimodal.persistence import CanonicalStore
from mandol_competition.multimodal.protocol import AddRequest
from mandol_competition.multimodal.purge import purge_user
from mandol_competition.multimodal.service import AddService
from mandol_competition.multimodal.store import UserGraphRegistry
from mandol_competition.multimodal.vlm.base import Extraction
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor


class Graph:
    def __init__(self):
        self.units = {}
        self.closed = False

    def add_unit(self, unit, **kwargs):
        self.units[unit.uid] = unit

    def rebuild_all_indexes(self):
        pass

    def get_multi_retriever(self):
        return self

    def smart_search(self, query, **kwargs):
        return [(unit, 1.0) for unit in self.units.values() if query in unit.text_cached]

    def close(self):
        self.closed = True


def image_uri():
    output = io.BytesIO()
    Image.new("RGB", (3, 3), "blue").save(output, "PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def setup(tmp_path, extractor=None):
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None)
    store = CanonicalStore(tmp_path / "canonical.sqlite3")
    graphs = []

    def factory(user_id):
        graph = Graph()
        graphs.append(graph)
        return graph

    registry = UserGraphRegistry(config, store, graph_factory=factory)
    vision = extractor or FakeVisionExtractor({}, model="mock", prompt_version="v1", schema_version="1")
    add = AddService(config, extractor=vision, registry=registry, store=store)
    return config, store, registry, add, graphs


def add_request(user_id, request_id, uri=None):
    content = [{"type": "text", "text": "blue backpack"}]
    if uri:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return AddRequest.model_validate({
        "user_id": user_id, "request_id": request_id, "session_id": "s1",
        "messages": [{"role": "user", "content": content}],
    })


def test_purge_removes_one_users_graph_units_and_media(tmp_path):
    config, store, registry, add, graphs = setup(tmp_path)
    add.handle(add_request("a/b", "r1", image_uri()))
    add.handle(add_request("a:b", "r2", image_uri()))
    media_id = store.succeeded_units("a:b")[0].raw_data["content_parts"][1]["media_id"]

    report = purge_user(user_id="a/b", config=config, store=store, registry=registry)
    assert (report.graph_evicted, report.units_removed, report.media_dir_removed) == (True, 1, True)
    assert graphs[0].closed
    assert not config.user_root("a/b").exists()
    assert store.succeeded_units("a/b") == []
    assert store.status_of("r1") is None
    assert len(store.succeeded_units("a:b")) == 1
    assert UserMediaStore(config, "a:b").get(media_id) is not None
    registry.close()
    store.close()


def test_purge_is_idempotent_and_old_request_can_be_readded(tmp_path):
    config, store, registry, add, _ = setup(tmp_path)
    request = add_request("alice", "r1")
    add.handle(request)
    purge_user(user_id="alice", config=config, store=store, registry=registry)
    again = purge_user(user_id="alice", config=config, store=store, registry=registry)
    assert again.units_removed == 0 and not again.media_dir_removed
    assert add.handle(request)["success"] is True
    assert len(store.succeeded_units("alice")) == 1
    registry.close()
    store.close()


def test_purge_refuses_active_graph_lease(tmp_path):
    config, store, registry, add, _ = setup(tmp_path)
    add.handle(add_request("alice", "r1"))
    with registry.acquire("alice"):
        with pytest.raises(RuntimeError, match="active graph leases"):
            purge_user(user_id="alice", config=config, store=store, registry=registry)
    assert len(store.succeeded_units("alice")) == 1
    registry.close()
    store.close()


def test_purge_waits_for_in_flight_add_before_deleting(tmp_path):
    entered, release = Event(), Event()

    class BlockingExtractor:
        model_name = "mock"

        def extract(self, *, image_data_uri, context_text, image_position, image_count):
            entered.set()
            assert release.wait(5)
            return Extraction(description="blue backpack", model="mock",
                              prompt_version="v1", schema_version="1")

    config, store, registry, add, _ = setup(tmp_path, BlockingExtractor())
    with ThreadPoolExecutor(max_workers=2) as pool:
        adding = pool.submit(add.handle, add_request("alice", "r1", image_uri()))
        assert entered.wait(5)
        purging = pool.submit(purge_user, user_id="alice", config=config,
                              store=store, registry=registry)
        assert not purging.done()
        release.set()
        assert adding.result(timeout=5)["success"] is True
        assert purging.result(timeout=5).units_removed == 1
    assert store.succeeded_units("alice") == []
    assert not config.user_root("alice").exists()
    registry.close()
    store.close()
