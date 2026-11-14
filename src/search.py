"""Search engine: text -> images and image -> images through one shared embedding space."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from .index import VectorIndex


class SearchEngine:
    def __init__(self, embedder, index: VectorIndex, meta: pd.DataFrame,
                 images_root: Optional[Path] = None) -> None:
        self.embedder, self.index = embedder, index
        self.meta = meta.reset_index(drop=True)
        self.images_root = Path(images_root) if images_root else None

    @classmethod
    def load(cls, index_dir: Path, images_root: Path, embedder=None, kind: str = "hnsw") -> "SearchEngine":
        index_dir = Path(index_dir)
        if embedder is None:
            from .embedder import CLIPEmbedder
            embedder = CLIPEmbedder()
        meta = pd.read_csv(index_dir / "metadata.csv")
        index = VectorIndex.load(index_dir / f"{kind}.faiss")
        return cls(embedder, index, meta, images_root)

    # ------------------------------------------------------------------ core
    def search_vector(self, vector: np.ndarray, k: int = 5, exclude_id: Optional[int] = None):
        scores, ids = self.index.search(vector, k + (1 if exclude_id is not None else 0))
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx == exclude_id:
                continue
            row = self.meta.iloc[int(idx)]
            results.append({"rank": len(results) + 1, "id": int(idx), "path": row["path"],
                            "category": row["category"], "score": round(float(score), 4)})
            if len(results) == k:
                break
        return results

    def search_text(self, query: str, k: int = 5) -> Dict[str, Any]:
        t0 = time.perf_counter()
        vec = self.embedder.embed_texts([query])
        t1 = time.perf_counter()
        results = self.search_vector(vec, k)
        t2 = time.perf_counter()
        return {"query": query, "results": results,
                "timings_ms": {"embed": round((t1 - t0) * 1000, 2), "search": round((t2 - t1) * 1000, 3)}}

    def search_image(self, image: Image.Image, k: int = 5) -> Dict[str, Any]:
        t0 = time.perf_counter()
        vec = self.embedder.embed_images([image])
        t1 = time.perf_counter()
        results = self.search_vector(vec, k)
        t2 = time.perf_counter()
        return {"results": results,
                "timings_ms": {"embed": round((t1 - t0) * 1000, 2), "search": round((t2 - t1) * 1000, 3)}}
