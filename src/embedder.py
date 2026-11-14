"""CLIP embedder: images and text are mapped into the same 512-d, L2-normalised space."""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .config import MODEL_NAME


class CLIPEmbedder:
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None,
                 fp16: Optional[bool] = None, batch_size: int = 64) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = self.device.type == "cuda" if fp16 is None else (fp16 and self.device.type == "cuda")
        self.batch_size = batch_size
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        if self.fp16:
            self.model.half()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = int(self.model.config.projection_dim)

    def _normalize(self, feats) -> np.ndarray:
        feats = feats.float()
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feats.cpu().numpy().astype("float32")

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = [im.convert("RGB") for im in images[i : i + self.batch_size]]
                pixels = self.processor(images=batch, return_tensors="pt")["pixel_values"].to(self.device)
                if self.fp16:
                    pixels = pixels.half()
                pooled = self.model.vision_model(pixel_values=pixels).pooler_output
                chunks.append(self._normalize(self.model.visual_projection(pooled)))
        return np.concatenate(chunks) if chunks else np.zeros((0, self.dim), dtype="float32")

    def embed_image_paths(self, paths: Sequence[Path], progress_every: int = 500) -> np.ndarray:
        chunks: List[np.ndarray] = []
        start = time.perf_counter()
        for i in range(0, len(paths), self.batch_size):
            images = []
            for p in paths[i : i + self.batch_size]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            chunks.append(self.embed_images(images))
            done = min(i + self.batch_size, len(paths))
            if progress_every and (done % progress_every < self.batch_size or done == len(paths)):
                rate = done / (time.perf_counter() - start)
                print(f"  embedded {done}/{len(paths)} images ({rate:.1f} img/s)")
        return np.concatenate(chunks)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(texts), 256):
                tokens = self.processor(text=list(texts[i : i + 256]), return_tensors="pt",
                                        padding=True, truncation=True, max_length=77).to(self.device)
                pooled = self.model.text_model(input_ids=tokens["input_ids"],
                                               attention_mask=tokens["attention_mask"]).pooler_output
                chunks.append(self._normalize(self.model.text_projection(pooled)))
        return np.concatenate(chunks) if chunks else np.zeros((0, self.dim), dtype="float32")
