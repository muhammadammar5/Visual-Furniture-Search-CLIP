"""REST API.   uvicorn src.api:app --reload      then open http://localhost:8000/docs

GET  /health
POST /search/text    {"query": "grey 3-seater sofa", "k": 5}
POST /search/image   multipart file upload, ?k=5
GET  /images/<path>  serves gallery images referenced in the results
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .config import IMAGES_DIR, INDEX_DIR
from .search import SearchEngine


class TextQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=300)
    k: int = Field(5, ge=1, le=50)


def create_app(engine: Optional[SearchEngine] = None, images_root: Optional[Path] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or SearchEngine.load(
            Path(os.getenv("INDEX_DIR", INDEX_DIR)), Path(os.getenv("IMAGES_DIR", IMAGES_DIR)),
            kind=os.getenv("INDEX_KIND", "hnsw"))
        yield

    app = FastAPI(title="Visual Furniture Search", version="1.0.0", lifespan=lifespan)
    root = Path(images_root or os.getenv("IMAGES_DIR", IMAGES_DIR))
    if root.exists():
        app.mount("/images", StaticFiles(directory=str(root)), name="images")

    def with_urls(payload: dict) -> dict:
        for r in payload["results"]:
            r["image_url"] = f"/images/{r['path']}"
        return payload

    @app.get("/health")
    def health():
        eng: SearchEngine = app.state.engine
        return {"status": "ok", "gallery_size": eng.index.size, "index_kind": eng.index.kind}

    @app.post("/search/text")
    def search_text(body: TextQuery):
        return with_urls(app.state.engine.search_text(body.query, body.k))

    @app.post("/search/image")
    async def search_image(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):
        try:
            image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Upload a valid image file.")
        return with_urls(app.state.engine.search_image(image, k))

    return app


app = create_app()
