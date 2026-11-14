"""Embed every image once and build the FAISS indexes.

    python -m src.build_index --images data/images
    python -m src.build_index --images data/images --reuse-embeddings    # rebuild indexes only

Writes to index/: embeddings.npy, metadata.csv, flat.faiss, hnsw.faiss, info.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import IMAGES_DIR, INDEX_DIR, MODEL_NAME
from .dataset import load_abo_metadata, scan_images
from .index import VectorIndex


def build(images_dir: Path, index_dir: Path, embedder=None, reuse: bool = False, verify: bool = True,
          max_images: int = None, seed: int = 42) -> dict:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    emb_path, meta_path = index_dir / "embeddings.npy", index_dir / "metadata.csv"

    embed_seconds = 0.0
    if reuse and emb_path.exists() and meta_path.exists():
        import pandas as pd
        meta, vectors, corrupt = pd.read_csv(meta_path), np.load(emb_path), 0
        print(f"Reusing {len(meta)} saved embeddings")
    else:
        if (Path(images_dir) / "metadata.csv").exists():      # ABO layout (see src/prepare_abo.py)
            meta, corrupt = load_abo_metadata(images_dir)
            print(f"ABO layout detected: {len(meta)} images ({corrupt} listed files missing)")
        else:
            meta, corrupt = scan_images(images_dir, verify=verify)
        if max_images and len(meta) > max_images:
            meta = meta.sample(max_images, random_state=seed).sort_index().reset_index(drop=True)
            print(f"--max-images: using a random sample of {len(meta)} images (quick trial)")
        print(f"Found {len(meta)} images in {meta['category'].nunique()} categories")
        if embedder is None:
            from .embedder import CLIPEmbedder
            embedder = CLIPEmbedder()
            print(f"CLIP on {embedder.device} (fp16={embedder.fp16})")
        start = time.perf_counter()
        vectors = embedder.embed_image_paths([Path(images_dir) / p for p in meta["path"]])
        embed_seconds = time.perf_counter() - start
        np.save(emb_path, vectors)
        meta.to_csv(meta_path, index=False)

    timings = {}
    for kind in ("flat", "hnsw"):
        start = time.perf_counter()
        VectorIndex(vectors.shape[1], kind).build(vectors).save(index_dir / f"{kind}.faiss")
        timings[f"{kind}_build_seconds"] = round(time.perf_counter() - start, 3)

    info = {"model": MODEL_NAME, "n_images": int(len(meta)), "dim": int(vectors.shape[1]),
            "n_categories": int(meta["category"].nunique()),
            "categories": meta["category"].value_counts().to_dict(),
            "corrupt_skipped": corrupt, "embed_seconds": round(embed_seconds, 1),
            "images_per_second": round(len(meta) / embed_seconds, 1) if embed_seconds else None, **timings}
    (index_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=str(IMAGES_DIR))
    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--no-verify", action="store_true", help="Skip corrupt-image check (faster)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Embed only a random sample (quick trial before the full run)")
    args = parser.parse_args()
    info = build(Path(args.images), Path(args.index_dir), reuse=args.reuse_embeddings,
                 verify=not args.no_verify, max_images=args.max_images)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
