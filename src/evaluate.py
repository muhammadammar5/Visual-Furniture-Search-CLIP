"""Measure retrieval quality and search speed on YOUR image collection.

    python -m src.evaluate --n-queries 200
    python -m src.evaluate --text-queries eval/text_queries.json     # add your own text queries

Two protocols (chosen automatically):
  * "split" (ABO dataset, has a train/val/test column): queries come from the TEST split and
    are searched against a catalogue made of the train+val products. No product appears on
    both sides, so there is no same-product leakage.
  * "leave-one-out" (plain folder-per-category data): sampled images are queries against the
    whole gallery, excluding the query itself. Near-duplicates can inflate these scores.

What is measured (relevance = result has the same category as the query):
  * image -> image category retrieval
  * same-product retrieval (ABO only): given one photo of a product, find its OTHER photos
    among the test-split photos. A much harder test of fine-grained matching.
  * text -> image: "a photo of a <category>" for each category, plus your own queries
  * zero-shot classification: assign each query image to the category whose text is closest
  * ANN quality: recall of the approximate HNSW index against exact (flat) search
  * search latency per query for both index types (vector search only)
Metrics: hit@k (at least one correct result in the top k), precision@k, MRR@10.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import INDEX_DIR, RESULTS_DIR, SEED
from .index import VectorIndex
from .retrieval_metrics import hit_at_k, precision_at_k, reciprocal_rank

PROMPT = "a photo of a {}"


def stratified_sample(categories: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Pick n indices, spreading them as evenly as possible across categories."""
    rng = np.random.default_rng(seed)
    pools = {c: list(rng.permutation(np.where(categories == c)[0])) for c in np.unique(categories)}
    chosen: List[int] = []
    while len(chosen) < min(n, len(categories)):
        for c in list(pools):
            if pools[c] and len(chosen) < n:
                chosen.append(pools[c].pop())
    return np.array(chosen)


def summarize(rows: List[List[bool]], k: int) -> Dict[str, float]:
    return {
        "n_queries": len(rows),
        f"hit@1": round(float(np.mean([hit_at_k(r, 1) for r in rows])), 4),
        f"hit@{k}": round(float(np.mean([hit_at_k(r, k) for r in rows])), 4),
        f"precision@{k}": round(float(np.mean([precision_at_k(r, k) for r in rows])), 4),
        "mrr@10": round(float(np.mean([reciprocal_rank(r, 10) for r in rows])), 4),
    }


def latency_ms(index: VectorIndex, queries: np.ndarray, k: int) -> Dict[str, float]:
    index.search(queries[:5], k)  # warm-up
    times = []
    for q in queries:
        start = time.perf_counter()
        index.search(q[None, :], k)
        times.append((time.perf_counter() - start) * 1000)
    return {"mean": round(statistics.mean(times), 3), "p50": round(statistics.median(times), 3),
            "p95": round(float(np.percentile(times, 95)), 3)}


def evaluate(index_dir: Path, embedder, n_queries: int = 200, k: int = 5,
             text_queries: Optional[List[dict]] = None, seed: int = SEED,
             protocol: str = "auto") -> Dict[str, object]:
    index_dir = Path(index_dir)
    vectors = np.load(index_dir / "embeddings.npy")
    meta = pd.read_csv(index_dir / "metadata.csv")
    cats = meta["category"].to_numpy()
    dim = vectors.shape[1]

    has_split = "split" in meta.columns and (meta["split"] == "test").any() and (meta["split"] != "test").any()
    if protocol == "auto":
        protocol = "split" if has_split else "leave-one-out"
    if protocol == "split" and not has_split:
        raise ValueError("protocol 'split' needs a metadata 'split' column with test and non-test rows")

    if protocol == "split":
        gal_idx = np.where(meta["split"] != "test")[0]        # catalogue: train + val products
        pool = np.where(meta["split"] == "test")[0]           # queries: unseen test products
    else:
        gal_idx = pool = np.arange(len(meta))
    gal_vecs, gal_cats = vectors[gal_idx], cats[gal_idx]
    flat = VectorIndex(dim, "flat").build(gal_vecs)
    hnsw = VectorIndex(dim, "hnsw").build(gal_vecs)

    report: Dict[str, object] = {
        "protocol": protocol, "n_gallery": int(len(gal_idx)), "n_query_pool": int(len(pool)),
        "n_categories": int(len(np.unique(cats))), "k": k}

    # ---- image -> image category retrieval
    q_global = pool[stratified_sample(cats[pool], n_queries, seed)]
    q_vecs = vectors[q_global]
    pos = {g: i for i, g in enumerate(gal_idx)}               # global id -> gallery position
    _, ids = flat.search(q_vecs, k + 1)
    rows = []
    for q, row in zip(q_global, ids):
        top = [i for i in row if not (protocol == "leave-one-out" and i == pos.get(q))][:k]
        rows.append([gal_cats[i] == cats[q] for i in top])
    report["n_queries"] = len(rows)
    report["image_to_image_category"] = summarize(rows, k)

    # ---- same-product retrieval (needs item ids, uses the test split only)
    if protocol == "split" and "item_id" in meta.columns:
        test_items = meta.loc[pool, "item_id"].to_numpy()
        counts = pd.Series(test_items).value_counts()
        multi = np.where(pd.Series(test_items).map(counts).to_numpy() >= 2)[0]
        if len(multi):
            rng = np.random.default_rng(seed)
            chosen = rng.choice(multi, size=min(n_queries, len(multi)), replace=False)
            sub = VectorIndex(dim, "flat").build(vectors[pool])
            _, ids = sub.search(vectors[pool][chosen], 11)
            rows = [[test_items[i] == test_items[q] for i in [j for j in row if j != q][:10]]
                    for q, row in zip(chosen, ids)]
            result = summarize(rows, k)
            result["products_with_multiple_views_in_test"] = int((counts >= 2).sum())
            report["same_product_retrieval"] = result

    # ---- text -> image
    names = sorted(np.unique(cats).tolist())
    cat_text = embedder.embed_texts([PROMPT.format(c) for c in names])
    _, ids = flat.search(cat_text, max(k, 10))
    rows = [[gal_cats[i] == name for i in row] for name, row in zip(names, ids)]
    report["text_to_image_category_prompts"] = summarize(rows, k)

    if text_queries:
        t_vecs = embedder.embed_texts([q["query"] for q in text_queries])
        _, ids = flat.search(t_vecs, max(k, 10))
        rows = [[gal_cats[i] in {c.lower() for c in q["relevant_categories"]} for i in row]
                for q, row in zip(text_queries, ids)]
        report["text_to_image_custom_queries"] = summarize(rows, k)

    # ---- zero-shot classification of the query pool
    pred = np.array(names)[np.argmax(vectors[pool] @ cat_text.T, axis=1)]
    report["zero_shot_classification"] = {
        "accuracy": round(float(np.mean(pred == cats[pool])), 4),
        "macro_f1": round(float(f1_score(cats[pool], pred, average="macro", zero_division=0)), 4)}

    # ---- ANN quality and speed
    _, exact = flat.search(q_vecs, 10)
    _, approx = hnsw.search(q_vecs, 10)
    recall = float(np.mean([len(set(a) & set(e)) / 10 for a, e in zip(approx, exact)]))
    report["hnsw_recall@10_vs_exact"] = round(recall, 4)
    report["search_latency_ms"] = {"flat": latency_ms(flat, q_vecs, k), "hnsw": latency_ms(hnsw, q_vecs, k)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--text-queries", default=None, help="JSON list of {query, relevant_categories}")
    parser.add_argument("--protocol", choices=["auto", "split", "leave-one-out"], default="auto")
    parser.add_argument("--out", default=str(RESULTS_DIR / "retrieval_metrics.json"))
    args = parser.parse_args()

    from .embedder import CLIPEmbedder
    custom = json.loads(Path(args.text_queries).read_text(encoding="utf-8")) if args.text_queries else None
    report = evaluate(Path(args.index_dir), CLIPEmbedder(), args.n_queries, args.k, custom,
                      protocol=args.protocol)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
