import io

from fastapi.testclient import TestClient

from src.api import create_app
from tests.fakes import FakeEmbedder, make_gallery, make_image
from src.build_index import build
from src.search import SearchEngine


def client(tmp_path):
    images, index = tmp_path / "images", tmp_path / "index"
    make_gallery(images, per_category=10)
    fake = FakeEmbedder()
    build(images, index, embedder=fake)
    engine = SearchEngine.load(index, images, embedder=fake, kind="hnsw")
    return TestClient(create_app(engine=engine, images_root=images))


def png_bytes(category, seed):
    buf = io.BytesIO()
    make_image(category, seed).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_text_search(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/health").json()["gallery_size"] == 40
        r = c.post("/search/text", json={"query": "a photo of a bed", "k": 3})
        body = r.json()
        assert r.status_code == 200 and len(body["results"]) == 3
        assert all(x["category"] == "bed" for x in body["results"])
        assert c.get(body["results"][0]["image_url"]).status_code == 200   # image is served


def test_image_search_and_errors(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/search/image?k=4", files={"file": ("q.png", png_bytes("chair", 5), "image/png")})
        assert r.status_code == 200 and {x["category"] for x in r.json()["results"]} == {"chair"}
        assert c.post("/search/image", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
        assert c.post("/search/text", json={"query": "a"}).status_code == 422
        assert c.post("/search/text", json={"query": "sofa", "k": 999}).status_code == 422
