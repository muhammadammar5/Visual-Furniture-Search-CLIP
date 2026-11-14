"""A fake CLIP stand-in so the pipeline can be tested without downloading any model.

Images are solid colours (one colour per category plus per-image noise); text that
contains a category name maps near that category's prototype, like real CLIP would.
"""
import hashlib

import numpy as np
from PIL import Image

COLORS = {"sofa": (200, 30, 30), "chair": (30, 200, 30), "table": (30, 30, 200), "bed": (200, 200, 30)}


def make_image(category: str, seed: int, size: int = 16) -> Image.Image:
    img = Image.new("RGB", (size, size), COLORS[category])
    rng = np.random.default_rng(seed)
    for _ in range(5):                                   # per-image variation
        img.putpixel((int(rng.integers(size)), int(rng.integers(size))),
                     tuple(int(v) for v in rng.integers(0, 255, 3)))
    return img


class FakeEmbedder:
    dim = 64

    def __init__(self):
        rng = np.random.default_rng(0)
        self.proto = {c: self._unit(rng.normal(size=self.dim)) for c in COLORS}

    @staticmethod
    def _unit(v):
        return (v / np.linalg.norm(v)).astype("float32")

    def _noise(self, key: bytes, scale=0.12):
        seed = int(hashlib.md5(key).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self.dim) * scale

    def embed_images(self, images):
        out = []
        for im in images:
            arr = np.asarray(im.convert("RGB"), dtype=float)
            mean = np.median(arr.reshape(-1, 3), axis=0)
            cat = min(COLORS, key=lambda c: np.linalg.norm(mean - np.array(COLORS[c])))
            out.append(self._unit(self.proto[cat] + self._noise(arr.tobytes())))
        return np.stack(out).astype("float32")

    def embed_image_paths(self, paths, progress_every=0):
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        return self.embed_images(images)

    def embed_texts(self, texts):
        out = []
        for t in texts:
            hit = next((c for c in COLORS if c in t.lower()), None)
            base = self.proto[hit] if hit else self._unit(self._noise(t.encode(), 1.0))
            out.append(self._unit(base + self._noise(t.encode(), 0.05)))
        return np.stack(out).astype("float32")


def make_gallery(root, per_category: int = 25):
    for cat in COLORS:
        folder = root / cat
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_category):
            make_image(cat, seed=hash((cat, i)) % 10_000).save(folder / f"{i}.png")
    return root
