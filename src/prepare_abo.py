"""Download and arrange the ABO furniture dataset (Hugging Face: marco-willi/abo_furniture).

    python -m src.prepare_abo

25,352 product photos (224x224 JPEG) in 6 categories: bed, chair, lamp, sofa, storage, table.
Split 80/10/10 by PRODUCT, so all views of one product stay in the same split.
License: CC BY-NC 4.0 (non-commercial; fine for a portfolio project, cite Collins et al., CVPR 2022).

Result: data/images/<split>/<label>/<image_id>.jpg  and  data/images/metadata.csv
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict

from .config import IMAGES_DIR

REPO_ID, ARCHIVE = "marco-willi/abo_furniture", "abo_furniture.tar.gz"
SPLITS = ("train", "val", "test")


def extract_and_organize(archive: Path, images_dir: Path) -> Dict[str, int]:
    """Unpack the archive and place the split folders and metadata.csv directly under images_dir."""
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=images_dir.parent) as tmp:
        tmp = Path(tmp)
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(tmp, filter="data")      # safer extraction on recent Python versions
            except TypeError:
                tar.extractall(tmp)

        roots = [p.parent for p in tmp.rglob("train") if p.is_dir() and (p.parent / "test").is_dir()]
        if not roots:
            raise RuntimeError("Could not find train/ and test/ folders inside the archive.")
        root = roots[0]
        meta = next(iter(tmp.rglob("metadata.csv")), None)
        if meta is None:
            raise RuntimeError("metadata.csv not found inside the archive.")

        for split in SPLITS:
            if (root / split).is_dir():
                target = images_dir / split
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(root / split), str(target))
        shutil.copy(meta, images_dir / "metadata.csv")

    counts = Counter(p.parent.name for p in images_dir.rglob("*.jpg"))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    print("Downloading ABO furniture subset (about 224 MB) ...")
    archive = hf_hub_download(REPO_ID, ARCHIVE, repo_type="dataset")
    counts = extract_and_organize(Path(archive), Path(args.images_dir))
    print("Images per category:", counts, "| total:", sum(counts.values()))
    print(f"Done. Next: python -m src.build_index --images {args.images_dir}")


if __name__ == "__main__":
    main()
