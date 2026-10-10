import base64
import io

from PIL import Image

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.media_store import compute_media_id
from mandol_competition.multimodal.persistence import CanonicalStore
from mandol_competition.multimodal.protocol import AddRequest, SearchRequest
from mandol_competition.multimodal.service import AddService, SearchService
from mandol_competition.multimodal.store import UserGraphRegistry
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor


class Graph:
    def __init__(self):
        self.units = {}
        self.queries = []

    def add_unit(self, unit, **kwargs):
        self.units[unit.uid] = unit

    def rebuild_all_indexes(self):
        pass

    def get_multi_retriever(self):
        return self

    def smart_search(self, query, **kwargs):
        self.queries.append(query)
        return [(unit, 1.0 - index * 0.1) for index, unit in enumerate(self.units.values())]

    def close(self):
        pass


def image(color):
    output = io.BytesIO()
    Image.new("RGB", (3, 3), color).save(output, format="PNG")
    raw = output.getvalue()
    return raw, "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def add_request(user, uri, index):
    return AddRequest(
        request_id=f"r{index}-{user}", user_id=user, session_id=f"s{index}",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"photo {index}"},
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": "from my album"},
        ]}],
    )


def search_request(user, parts, top_k=100, options=None):
    return SearchRequest(query=parts, user_id=user, top_k=top_k, options=options)


def setup(tmp_path, fixture, **options):
    config = MultimodalConfig(data_dir=tmp_path, rerank_method=None, **options)
    store = CanonicalStore(tmp_path / "canonical.db")
    graphs = {}

    def factory(user):
        graph = Graph()
        graphs[user] = graph
        return graph

    registry = UserGraphRegistry(config, store, graph_factory=factory)
    extractor = FakeVisionExtractor(fixture, model="mock", prompt_version="v1", schema_version="1")
    add = AddService(config, extractor=extractor, registry=registry, store=store)
    search = SearchService(config, extractor=extractor, registry=registry, store=store)
    return add, search, graphs, extractor


def test_image_query_options_order_and_user_isolation(tmp_path):
    raw_red, red = image("red")
    raw_blue, blue = image("blue")
    raw_green, green = image("green")
    fixture = {
        compute_media_id(raw_red): {"description": "red backpack"},
        compute_media_id(raw_blue): {"description": "blue blanket"},
        compute_media_id(raw_green): {"description": "red backpack"},
    }
    add, search, graphs, extractor = setup(tmp_path, fixture)
    add.handle(add_request("alice", red, 1))
    add.handle(add_request("alice", blue, 2))
    add.handle(add_request("bob", blue, 3))

    request = search_request("alice", [
        {"type": "text", "text": "Which backpack?"},
        {"type": "image_url", "image_url": {"url": green}},
    ], options=["A. red", "B. blue"])
    result = search.handle(request)
    assert len(result["data"]) == 2
    assert "red backpack" in graphs["alice"].queries[-1]
    assert "[Options]\nA. red\nB. blue" in graphs["alice"].queries[-1]
    assert "red backpack" in result["data"][0]["content"][0]["text"]
    assert result["data"][0]["content"][1]["image_url"]["url"] == red
    assert result["data"][0]["score"] > result["data"][1]["score"]
    assert len(search.handle(search_request("alice", [{"type": "text", "text": "backpack"}], 1))["data"]) == 1
    assert len(search.handle(search_request("bob", [{"type": "text", "text": "blanket"}]))["data"]) == 1
    assert search.handle(search_request("nobody", [{"type": "text", "text": "anything"}])) == {"data": []}
    assert not list((search.config.user_root("alice") / "media").rglob(compute_media_id(raw_green)[4:] + ".bin"))
    assert extractor.calls == 4
    search.registry.close()
    search.store.close()


def test_ranked_response_keeps_full_text_when_image_budget_exhausted(tmp_path):
    first_raw, first = image("red")
    second_raw, second = image("blue")
    fixture = {
        compute_media_id(first_raw): {"description": "red backpack"},
        compute_media_id(second_raw): {"description": "blue blanket"},
    }
    add, search, graphs, _ = setup(tmp_path, fixture, max_images=1, image_byte_budget=len(first_raw))
    add.handle(add_request("alice", first, 1))
    add.handle(add_request("alice", second, 2))
    result = search.handle(search_request("alice", [{"type": "text", "text": "photo"}]))
    assert len(result["data"]) == 2
    assert [part["type"] for part in result["data"][0]["content"]] == ["text", "image_url"]
    assert [part["type"] for part in result["data"][1]["content"]] == ["text"]
    assert "blue blanket" in result["data"][1]["content"][0]["text"]
    search.registry.close()
    search.store.close()
