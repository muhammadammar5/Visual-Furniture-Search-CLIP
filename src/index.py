"""FAISS vector index wrapper. Vectors are L2-normalised, so inner product = cosine similarity.

kinds:
  flat  exact brute-force search (the accuracy reference)
  hnsw  approximate graph index: much faster on large galleries, tiny recall loss
  ivf   approximate inverted-file index (needs training)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, dim: int, kind: str = "flat", hnsw_m: int = 32,
                 ef_construction: int = 200, ef_search: int = 64, nprobe: int = 16) -> None:
        if kind not in ("flat", "hnsw", "ivf"):
            raise ValueError(f"Unknown index kind: {kind}")
        self.dim, self.kind = dim, kind
        self.params = {"hnsw_m": hnsw_m, "ef_construction": ef_construction,
                       "ef_search": ef_search, "nprobe": nprobe}
        self.index = None

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        n = len(vectors)
        p = self.params
        if self.kind == "flat":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, p["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = p["ef_construction"]
            self.index.hnsw.efSearch = p["ef_search"]
        else:
            nlist = max(1, min(int(4 * np.sqrt(n)), n // 10 or 1))
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dim), self.dim, nlist,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.nprobe = min(p["nprobe"], nlist)
        self.index.add(vectors)
        return self

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype="float32")
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(queries, k)
        return scores, ids

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def save(self, path: Path) -> None:
        path = Path(path)
        faiss.write_index(self.index, str(path))
        path.with_suffix(".json").write_text(
            json.dumps({"dim": self.dim, "kind": self.kind, **self.params}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        obj = cls(meta["dim"], meta["kind"], meta["hnsw_m"], meta["ef_construction"],
                  meta["ef_search"], meta["nprobe"])
        obj.index = faiss.read_index(str(path))
        return obj
