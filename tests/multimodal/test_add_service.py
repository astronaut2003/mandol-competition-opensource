import base64
import io
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from PIL import Image

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.persistence import CanonicalStore
from mandol_competition.multimodal.protocol import AddRequest
from mandol_competition.multimodal.service import AddService, ConflictError
from mandol_competition.multimodal.store import UserGraphRegistry
from mandol_competition.multimodal.vlm.base import Extraction
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor


class Graph:
    def __init__(self, *, fail=False):
        self.units = {}
        self.fail = fail
        self.closed = False

    def add_unit(self, unit, **kwargs):
        if self.fail:
            raise RuntimeError("index failed")
        self.units[unit.uid] = unit

    def rebuild_all_indexes(self):
        pass

    def get_multi_retriever(self):
        return self

    def smart_search(self, query, **kwargs):
        return [(unit, 1.0) for unit in self.units.values() if query in unit.text_cached]

    def close(self):
        self.closed = True


def image_uri(color="red"):
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def payload(request_id="r1", user_id="alice", images=None):
    images = images if images is not None else [image_uri()]
    return AddRequest(
        request_id=request_id,
        user_id=user_id,
        session_id="s1",
        messages=[
            {"role": "user", "timestamp": 1704067200000, "content": [
                {"type": "text", "text": "red bag"},
                *({"type": "image_url", "image_url": {"url": uri}} for uri in images),
                {"type": "text", "text": "on the chair"},
            ]},
            {"role": "assistant", "content": "I see it."},
        ],
    )


def setup(tmp_path, extractor=None, factory=None, **config_options):
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None, **config_options)
    store = CanonicalStore(tmp_path / "canonical.db")
    graphs = []

    def graph_factory(user_id):
        graph = factory(user_id) if factory else Graph()
        graphs.append(graph)
        return graph

    registry = UserGraphRegistry(config, store, graph_factory=graph_factory)
    extractor = extractor or FakeVisionExtractor({}, model="mock", prompt_version="v1", schema_version="1")
    return AddService(config, extractor=extractor, registry=registry, store=store), graphs


def test_add_is_immediately_searchable_and_retry_is_idempotent(tmp_path):
    service, graphs = setup(tmp_path)
    request = payload(images=[image_uri("red"), image_uri("blue")])
    response = service.handle(request)
    assert response == {"success": True, "request_id": "r1", "user_id": "alice", "session_id": "s1"}
    assert service.extractor.calls == 2
    assert len(graphs[0].units) == 2
    first, second = list(graphs[0].units.values())
    assert first.raw_data["content_parts"][0] == {"type": "text", "text": "red bag"}
    assert [part["type"] for part in first.raw_data["content_parts"]] == [
        "text", "image", "image", "text"
    ]
    assert "red bag\non the chair" in first.raw_data["text_content"]
    assert first.metadata["timestamp"] == "2024-01-01 00:00:00"
    assert second.metadata["modality"] == "text"
    with service.registry.acquire("alice") as session:
        assert session.search("red bag", top_k=10)
    assert service.handle(request) == response
    assert service.extractor.calls == 2
    with pytest.raises(ConflictError):
        service.handle(payload(images=[image_uri("green")]))
    service.registry.close()
    service.store.close()


def test_invalid_later_message_never_writes_media_or_request(tmp_path):
    service, graphs = setup(tmp_path)
    request = AddRequest.model_validate({
        "request_id": "r1", "user_id": "alice", "session_id": "s1",
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_uri()}}]},
            {"role": "assistant", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.org/x"}}
            ]},
        ],
    })
    with pytest.raises(ValueError):
        service.handle(request)
    assert service.store.request_count() == 0
    assert graphs == []
    assert not list(tmp_path.rglob("*.bin"))
    service.registry.close()
    service.store.close()


def test_index_failure_does_not_replay_or_leave_ghost_memory(tmp_path):
    failures = iter([True, False])
    service, graphs = setup(tmp_path, factory=lambda user: Graph(fail=next(failures)))
    with pytest.raises(RuntimeError, match="index failed"):
        service.handle(payload())
    assert service.store.status_of("r1") == "prepared"
    assert service.store.succeeded_units("alice") == []
    assert graphs[0].closed
    assert service.handle(payload())["success"] is True
    assert len(service.store.succeeded_units("alice")) == 2
    service.registry.close()
    service.store.close()


def test_extractor_failure_keeps_original_evidence_with_degradation_mark(tmp_path):
    extractor = FakeVisionExtractor({}, strict=True, model="mock", prompt_version="v1", schema_version="1")
    service, graphs = setup(tmp_path, extractor=extractor)
    assert service.handle(payload())["success"] is True
    unit = next(iter(graphs[0].units.values()))
    assert unit.metadata["vlm_degraded"] is True
    assert unit.raw_data["content_parts"][1]["type"] == "image"
    service.registry.close()
    service.store.close()


def test_same_id_parallel_calls_extract_only_once(tmp_path):
    entered, release = Event(), Event()

    class SlowExtractor:
        model_name = "slow"
        calls = 0

        def extract(self, **kwargs):
            self.calls += 1
            entered.set()
            assert release.wait(5)
            return Extraction(description="red bag")

    service, graphs = setup(tmp_path, extractor=SlowExtractor())
    request = payload()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(service.handle, request)
        assert entered.wait(5)
        second = pool.submit(service.handle, request)
        release.set()
        assert first.result(timeout=5) == second.result(timeout=5)
    assert service.extractor.calls == 1
    assert len(graphs[0].units) == 2
    service.registry.close()
    service.store.close()
