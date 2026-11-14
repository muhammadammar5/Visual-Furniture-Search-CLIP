import json

import numpy as np

from src.build_index import build
from src.evaluate import evaluate, stratified_sample
from src.retrieval_metrics import hit_at_k, precision_at_k, reciprocal_rank
from src.search import SearchEngine
from tests.fakes import FakeEmbedder, make_gallery, make_image


def build_engine(tmp_path, kind="flat"):
    images, index = tmp_path / "images", tmp_path / "index"
    make_gallery(images, per_category=25)
    fake = FakeEmbedder()
    info = build(images, index, embedder=fake)
    return SearchEngine.load(index, images, embedder=fake, kind=kind), info, index, fake


def test_build_writes_all_artifacts(tmp_path):
    _, info, index, _ = build_engine(tmp_path)
    for name in ("embeddings.npy", "metadata.csv", "flat.faiss", "hnsw.faiss", "info.json"):
        assert (index / name).exists()
    assert info["n_images"] == 100 and info["n_categories"] == 4
    assert np.load(index / "embeddings.npy").shape == (100, 64)


def test_reuse_embeddings_skips_encoding(tmp_path):
    engine, _, index, fake = build_engine(tmp_path)

    class Boom(FakeEmbedder):
        def embed_image_paths(self, *a, **k):
            raise AssertionError("should not re-embed")

    info = build(tmp_path / "images", index, embedder=Boom(), reuse=True)
    assert info["n_images"] == 100


def test_text_and_image_search_return_matching_category(tmp_path):
    engine, _, _, _ = build_engine(tmp_path, kind="hnsw")
    out = engine.search_text("a photo of a sofa", k=5)
    assert [r["category"] for r in out["results"]] == ["sofa"] * 5
    assert out["timings_ms"]["search"] >= 0
    out = engine.search_image(make_image("table", seed=999), k=5)
    assert {r["category"] for r in out["results"]} == {"table"}
    assert [r["rank"] for r in out["results"]] == [1, 2, 3, 4, 5]


def test_exclude_id_removes_query_itself(tmp_path):
    engine, _, index, _ = build_engine(tmp_path)
    vec = np.load(index / "embeddings.npy")[7:8]
    assert engine.search_vector(vec, 3)[0]["id"] == 7
    assert 7 not in [r["id"] for r in engine.search_vector(vec, 3, exclude_id=7)]


def test_ranking_metrics_known_values():
    rel = [False, True, False, True, False]
    assert hit_at_k(rel, 1) == 0 and hit_at_k(rel, 2) == 1
    assert precision_at_k(rel, 5) == 0.4
    assert reciprocal_rank(rel, 10) == 0.5 and reciprocal_rank([False] * 5, 10) == 0


def test_stratified_sample_is_balanced_and_deterministic():
    cats = np.array(["a"] * 50 + ["b"] * 50)
    s1, s2 = stratified_sample(cats, 20, seed=1), stratified_sample(cats, 20, seed=1)
    assert (s1 == s2).all() and len(set(s1)) == 20
    assert sorted(np.bincount(np.searchsorted(["a", "b"], cats[s1]))) == [10, 10]


def test_full_evaluation_with_fake_embedder(tmp_path):
    _, _, index, fake = build_engine(tmp_path)
    custom = [{"query": "a comfy sofa", "relevant_categories": ["sofa"]}]
    report = evaluate(index, fake, n_queries=40, k=5, text_queries=custom)
    assert report["image_to_image_category"]["hit@5"] == 1.0
    assert report["text_to_image_category_prompts"]["precision@5"] == 1.0
    assert report["text_to_image_custom_queries"]["hit@5"] == 1.0
    assert report["zero_shot_classification"]["accuracy"] == 1.0
    assert report["hnsw_recall@10_vs_exact"] > 0.9
    assert set(report["search_latency_ms"]) == {"flat", "hnsw"}
    json.dumps(report)                                   # must be JSON-serialisable
