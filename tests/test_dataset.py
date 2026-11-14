from src.dataset import scan_images
from tests.fakes import make_gallery, make_image


def test_scan_finds_images_and_categories(tmp_path):
    make_gallery(tmp_path, per_category=3)
    df, corrupt = scan_images(tmp_path)
    assert len(df) == 12 and corrupt == 0
    assert set(df["category"]) == {"sofa", "chair", "table", "bed"}
    assert all("/" in p for p in df["path"])             # stored relative, POSIX style


def test_corrupt_files_and_root_images_are_skipped(tmp_path):
    make_gallery(tmp_path, per_category=2)
    (tmp_path / "sofa" / "broken.jpg").write_text("not an image")
    make_image("bed", 1).save(tmp_path / "stray.png")      # no category folder
    df, corrupt = scan_images(tmp_path)
    assert corrupt == 1 and len(df) == 8


def test_category_names_are_normalised(tmp_path):
    folder = tmp_path / "Swivel_Chair"
    folder.mkdir()
    make_image("chair", 2).save(folder / "a.png")
    df, _ = scan_images(tmp_path)
    assert df["category"].iloc[0] == "swivel chair"
