import io
import tarfile

import numpy as np
import pandas as pd
import pytest

from src.build_index import build
from src.dataset import load_abo_metadata
from src.evaluate import evaluate
from src.index import VectorIndex
from src.prepare_abo import extract_and_organize
from tests.fakes import COLORS, FakeEmbedder, make_image


def make_abo(root, items_per_split=None, views=3):
    """Create an ABO-style tree:  root/<split>/<label>/<image_id>.jpg + root/metadata.csv"""
    items_per_split = items_per_split or {"train": 5, "val": 2, "test": 3}
    rows = []
    for split, n_items in items_per_split.items():
        for cat in COLORS:
            for item in range(n_items):
                for view in range(views):
                    image_id = f"{split}_{cat}_{item}_{view}"
                    folder = root / split / cat
                    folder.mkdir(parents=True, exist_ok=True)
                    make_image(cat, seed=abs(hash(image_id)) % 10_000).save(folder / f"{image_id}.jpg")
                    rows.append({"image_id": image_id, "item_id": f"{split}_{cat}_{item}", "label": cat,
                                 "split": split, "is_primary": view == 0, "view_index": view})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return len(rows)


def test_load_abo_metadata_builds_paths_and_drops_missing(tmp_path):
    n = make_abo(tmp_path)
    (tmp_path / "train" / "sofa" / "train_sofa_0_0.jpg").unlink()
    meta, missing = load_abo_metadata(tmp_path)
    assert missing == 1 and len(meta) == n - 1
    assert set(meta.columns) == {"path", "category", "split", "item_id"}
    assert meta["path"].iloc[0].count("/") == 2


def test_load_abo_metadata_requires_columns(tmp_path):
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "metadata.csv", index=False)
    with pytest.raises(ValueError):
        load_abo_metadata(tmp_path)


def test_extract_and_organize_handles_nested_archive(tmp_path):
    src = tmp_path / "src"
    make_abo(src, {"train": 2, "val": 1, "test": 1}, views=2)
    archive = tmp_path / "abo.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:                 # layout: abo/<split>/... and abo/metadata.csv
        tar.add(src, arcname="abo")
    out = tmp_path / "images"
    counts = extract_and_organize(archive, out)
    assert (out / "metadata.csv").exists() and (out / "train" / "chair").is_dir()
    assert set(counts) == set(COLORS) and sum(counts.values()) == 4 * 4 * 2


def test_build_in_abo_mode_keeps_split_and_item_columns(tmp_path):
    make_abo(tmp_path / "images")
    build(tmp_path / "images", tmp_path / "index", embedder=FakeEmbedder())
    meta = pd.read_csv(tmp_path / "index" / "metadata.csv")
    assert {"split", "item_id", "category", "path"} <= set(meta.columns)
    assert set(meta["split"]) == {"train", "val", "test"}


def test_max_images_samples_a_subset(tmp_path):
    make_abo(tmp_path / "images")
    info = build(tmp_path / "images", tmp_path / "index", embedder=FakeEmbedder(), max_images=30)
    assert info["n_images"] == 30


def structured_artifacts(tmp_path, dim=64, seed=0):
    """Embeddings where category dominates, each product has its own offset, and views add small noise."""
    rng = np.random.default_rng(seed)
    cat_proto = {c: rng.normal(size=dim) for c in COLORS}
    rows, vecs = [], []
    for split, n_items in {"train": 6, "val": 2, "test": 5}.items():
        for cat in COLORS:
            for item in range(n_items):
                offset = rng.normal(size=dim) * 0.6
                for view in range(3):
                    v = cat_proto[cat] + offset + rng.normal(size=dim) * 0.15
                    vecs.append(v / np.linalg.norm(v))
                    rows.append({"path": f"{split}/{cat}/{item}_{view}.jpg", "category": cat,
                                 "split": split, "item_id": f"{split}_{cat}_{item}"})
    vecs = np.array(vecs, dtype="float32")
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    np.save(index_dir / "embeddings.npy", vecs)
    pd.DataFrame(rows).to_csv(index_dir / "metadata.csv", index=False)
    for kind in ("flat", "hnsw"):
        VectorIndex(dim, kind).build(vecs).save(index_dir / f"{kind}.faiss")
    return index_dir


def test_split_protocol_uses_unseen_test_products_and_same_product_metric(tmp_path):
    index_dir = structured_artifacts(tmp_path)
    report = evaluate(index_dir, FakeEmbedder(), n_queries=40, k=5)
    assert report["protocol"] == "split"
    assert report["n_gallery"] == 4 * 8 * 3 and report["n_query_pool"] == 4 * 5 * 3
    assert report["image_to_image_category"]["hit@5"] == 1.0
    same = report["same_product_retrieval"]
    assert same["hit@1"] > 0.9 and same["products_with_multiple_views_in_test"] == 20


def test_auto_protocol_falls_back_to_leave_one_out_without_split(tmp_path):
    index_dir = structured_artifacts(tmp_path)
    meta = pd.read_csv(index_dir / "metadata.csv").drop(columns=["split", "item_id"])
    meta.to_csv(index_dir / "metadata.csv", index=False)
    report = evaluate(index_dir, FakeEmbedder(), n_queries=20, k=5)
    assert report["protocol"] == "leave-one-out" and "same_product_retrieval" not in report
