"""Project constants. Run every command from the repository root."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
INDEX_DIR = BASE_DIR / "index"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "openai/clip-vit-base-patch32"   # 512-dimensional joint image/text embeddings
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
