import numpy as np
import pytest

from src.benchmark import synthetic_vectors
from src.index import VectorIndex


@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivf"])
def test_search_returns_self_first_and_sorted_scores(kind):
    data = synthetic_vectors(2000, 32, seed=0, n_clusters=20)
    idx = VectorIndex(32, kind).build(data)
    scores, ids = idx.search(data[:5], 5)
    assert (ids[:, 0] == np.arange(5)).all()
    assert (np.diff(scores, axis=1) <= 1e-6).all()        # descending similarity


def test_hnsw_recall_is_high_vs_exact():
    data = synthetic_vectors(3000, 64, seed=1, n_clusters=30)
    q = synthetic_vectors(100, 64, seed=2, n_clusters=30)
    _, exact = VectorIndex(64, "flat").build(data).search(q, 10)
    _, approx = VectorIndex(64, "hnsw").build(data).search(q, 10)
    recall = np.mean([len(set(a) & set(e)) / 10 for a, e in zip(approx, exact)])
    assert recall > 0.9


def test_save_and_load_roundtrip(tmp_path):
    data = synthetic_vectors(500, 16, seed=3, n_clusters=10)
    idx = VectorIndex(16, "hnsw").build(data)
    idx.save(tmp_path / "x.faiss")
    loaded = VectorIndex.load(tmp_path / "x.faiss")
    assert loaded.kind == "hnsw" and loaded.size == 500
    assert (loaded.search(data[:3], 3)[1] == idx.search(data[:3], 3)[1]).all()


def test_k_larger_than_gallery_is_clamped():
    idx = VectorIndex(8, "flat").build(synthetic_vectors(4, 8, seed=0, n_clusters=2))
    assert idx.search(np.ones((1, 8), "float32"), 10)[1].shape[1] == 4


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        VectorIndex(8, "bogus")
