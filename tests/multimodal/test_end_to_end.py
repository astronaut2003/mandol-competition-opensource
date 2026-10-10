import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.media_store import UserMediaStore
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
        return [(unit, 1.0) for unit in list(self.units.values())[: kwargs["top_k"]]]

    def close(self):
        self.closed = True


def image_uri():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(stream, "PNG")
    raw = stream.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), len(raw)


def request(uri, user="alice", request_id="r1"):
    return {"request_id": request_id, "user_id": user, "session_id": "s1", "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "a red flower"},
            {"type": "image_url", "image_url": {"url": uri}},
        ]},
    ]}


def query(user="alice", top_k=100):
    return {"query": [{"type": "text", "text": "flower"}], "user_id": user, "top_k": top_k}


def app_for(config, graphs):
    def factory(user):
        graph = Graph()
        graphs[user] = graph
        return graph

    return create_app(
        config,
        extractor=FakeVisionExtractor({}, model="mock", prompt_version="v1", schema_version="1"),
        graph_factory=factory,
    )


def test_restart_rebuilds_only_same_user_evidence(tmp_path):
    uri, _ = image_uri()
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None)
    first_graphs = {}
    with TestClient(app_for(config, first_graphs)) as client:
        assert client.post("/add", json=request(uri)).status_code == 200
        assert len(client.post("/search", json=query()).json()["data"]) == 1
    second_graphs = {}
    with TestClient(app_for(config, second_graphs)) as client:
        response = client.post("/search", json=query())
        assert response.status_code == 200
        assert len(response.json()["data"]) == 1
        assert response.json()["data"][0]["content"][1]["image_url"]["url"] == uri
        assert client.post("/search", json=query("bob")).json() == {"data": []}
        assert client.post("/add", json=request(uri)).json()["success"] is True
        assert len(second_graphs["alice"].units) == 1


def test_whole_add_image_budget_rejects_before_any_write(tmp_path):
    uri, size = image_uri()
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None,
                              max_request_image_bytes=size, image_byte_budget=size)
    graphs = {}
    app = app_for(config, graphs)
    body = request(uri)
    body["messages"].append({"role": "assistant", "content": [
        {"type": "image_url", "image_url": {"url": uri}},
    ]})
    with TestClient(app) as client:
        response = client.post("/add", json=body)
        assert response.status_code == 422
        assert app.state.add_service.store.request_count() == 0
        assert graphs == {}
        assert not list(tmp_path.rglob("*.bin"))


def test_commit_failure_does_not_publish_ghost_or_replay(tmp_path, monkeypatch):
    uri, _ = image_uri()
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None)
    graphs = {}
    app = app_for(config, graphs)
    store = app.state.add_service.store
    original_mark = store.mark_succeeded

    def fail_commit(*args):
        raise OSError("commit failed")

    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(store, "mark_succeeded", fail_commit)
        assert client.post("/add", json=request(uri)).status_code == 500
        assert store.status_of("r1") == "prepared"
        assert store.succeeded_units("alice") == []
        assert graphs["alice"].closed
        monkeypatch.setattr(store, "mark_succeeded", original_mark)
        assert client.post("/search", json=query()).json() == {"data": []}
        assert client.post("/add", json=request(uri)).status_code == 200
        assert len(client.post("/search", json=query()).json()["data"]) == 1


def test_missing_media_keeps_textual_visual_description(tmp_path):
    uri, _ = image_uri()
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None)
    app = app_for(config, {})
    with TestClient(app) as client:
        client.post("/add", json=request(uri))
        media = UserMediaStore(config, "alice")
        media_id = app.state.add_service.store.succeeded_units("alice")[0].raw_data["content_parts"][1]["media_id"]
        media.path_for(media_id).unlink()
        result = client.post("/search", json=query()).json()["data"][0]
        assert [part["type"] for part in result["content"]] == ["text"]
        assert "a red flower" in result["content"][0]["text"]
