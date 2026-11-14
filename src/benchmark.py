"""Speed benchmarks.

    python -m src.benchmark ann                 # exact vs HNSW on SYNTHETIC clustered vectors
    python -m src.benchmark embed               # CLIP encoding latency on your machine

`ann` shows how vector search scales to large galleries (synthetic data, clearly labelled).
`embed` measures the real cost of encoding one text query / one image with CLIP, which
usually dominates end-to-end latency once the index is built.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .config import IMAGES_DIR, INDEX_DIR, RESULTS_DIR
from .index import VectorIndex


def synthetic_vectors(n: int, dim: int, seed: int, n_clusters: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, dim))
    x = centers[rng.integers(0, n_clusters, n)] + 0.35 * rng.normal(size=(n, dim))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype("float32")


def time_queries(index: VectorIndex, queries: np.ndarray, k: int) -> float:
    index.search(queries[:5], k)
    times = []
    for q in queries:
        start = time.perf_counter()
        index.search(q[None, :], k)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def run_ann(sizes, dim=512, n_queries=200, k=10, ef_search=64) -> list:
    rows = []
    for n in sizes:
        data = synthetic_vectors(n, dim, seed=1)
        queries = synthetic_vectors(n_queries, dim, seed=2)
        flat = VectorIndex(dim, "flat").build(data)
        start = time.perf_counter()
        hnsw = VectorIndex(dim, "hnsw", ef_search=ef_search).build(data)
        build_s = time.perf_counter() - start
        _, exact = flat.search(queries, k)
        _, approx = hnsw.search(queries, k)
        recall = float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))
        row = {"gallery_size": n, "ef_search": ef_search, "flat_ms": round(time_queries(flat, queries, k), 3),
               "hnsw_ms": round(time_queries(hnsw, queries, k), 3),
               "hnsw_recall@10": round(recall, 4), "hnsw_build_s": round(build_s, 1)}
        row["speedup"] = round(row["flat_ms"] / max(row["hnsw_ms"], 1e-9), 1)
        rows.append(row)
        print(row)
    return rows


def run_embed(n: int = 30) -> dict:
    from PIL import Image

    from .dataset import scan_images
    from .embedder import CLIPEmbedder

    embedder = CLIPEmbedder()
    meta, _ = scan_images(IMAGES_DIR, verify=False)
    images = []
    for p in meta["path"].head(n):
        with Image.open(IMAGES_DIR / p) as im:
            images.append(im.convert("RGB"))

    def med(fn, items):
        fn(items[:2])  # warm-up
        times = []
        for item in items:
            start = time.perf_counter()
            fn([item])
            times.append((time.perf_counter() - start) * 1000)
        return round(statistics.median(times), 2)

    texts = ["grey 3-seater sofa", "wooden dining table", "king size bed"] * (n // 3)
    return {"device": str(embedder.device), "fp16": embedder.fp16,
            "text_encode_ms_median": med(embedder.embed_texts, texts),
            "image_encode_ms_median": med(embedder.embed_images, images)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    ann = sub.add_parser("ann")
    ann.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    ann.add_argument("--ef-search", type=int, default=64,
                     help="HNSW search depth: higher = better recall, slower queries")
    sub.add_parser("embed")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.cmd == "ann":
        out = {"note": "synthetic clustered vectors, 512-d", "rows": run_ann(args.sizes, ef_search=args.ef_search)}
        (RESULTS_DIR / "benchmark_ann.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        out = run_embed()
        (RESULTS_DIR / "benchmark_embed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
