"""Scan an image folder laid out as  data/images/<category>/<any depth>/<image>."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image

from .config import IMAGE_EXTS


def scan_images(root: Path, verify: bool = True) -> Tuple[pd.DataFrame, int]:
    """Return (DataFrame[path, category], number_of_corrupt_files_skipped).

    `path` is stored relative to `root` (POSIX style) so the index stays portable.
    The category is the first-level folder name.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    rows, corrupt = [], 0
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTS or not path.is_file():
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue  # images directly in root have no category folder
        if verify:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                corrupt += 1
                continue
        rows.append({"path": rel.as_posix(), "category": rel.parts[0].lower().replace("_", " ")})

    df = pd.DataFrame(rows, columns=["path", "category"])
    if df.empty:
        raise ValueError(f"No images found under {root}. Expected {root}/<category>/<image>.jpg")
    return df, corrupt


def load_abo_metadata(root: Path):
    """Load the ABO furniture layout:  <root>/metadata.csv  +  <root>/<split>/<label>/<image_id>.jpg

    Returns (DataFrame[path, category, split, item_id], number_of_missing_files).
    `item_id` identifies one product (several photos/views share it).
    """
    root = Path(root)
    df = pd.read_csv(root / "metadata.csv")
    needed = {"image_id", "label", "split"}
    if not needed <= set(df.columns):
        raise ValueError(f"metadata.csv must contain {sorted(needed)}; found {list(df.columns)}")

    out = pd.DataFrame({
        "path": df["split"].astype(str) + "/" + df["label"].astype(str) + "/" + df["image_id"].astype(str) + ".jpg",
        "category": df["label"].astype(str).str.lower().str.replace("_", " ", regex=False),
        "split": df["split"].astype(str).str.lower(),
        "item_id": df["item_id"].astype(str) if "item_id" in df.columns else df["image_id"].astype(str),
    })
    exists = out["path"].map(lambda p: (root / p).is_file())
    missing = int((~exists).sum())
    out = out[exists].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"None of the files listed in {root / 'metadata.csv'} were found under {root}")
    return out, missing
