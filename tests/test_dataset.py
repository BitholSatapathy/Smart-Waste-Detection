"""Unit tests for dataset validation, stratified splitting, and preparation."""

import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.dataset import DatasetManager, DatasetValidationError


def _create_dummy_image(path: Path, width: int = 100, height: int = 100) -> None:
    """Helper to write a valid synthetic image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_missing_raw_dir(tmp_path: Path):
    """Verifies that non-existent raw dataset directory raises DatasetValidationError."""
    manager = DatasetManager(raw_dir=tmp_path / "non_existent_raw", output_dir=tmp_path / "processed")
    with pytest.raises(DatasetValidationError, match="Raw dataset directory not found"):
        manager.validate_raw_dataset()


def test_missing_class_folder(tmp_path: Path):
    """Verifies error if any required class folder is missing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create only 5 of the 6 classes
    classes = ["cardboard", "glass", "metal", "paper", "plastic"]
    for c in classes:
        cls_dir = raw_dir / c
        cls_dir.mkdir()
        _create_dummy_image(cls_dir / "img1.jpg")

    manager = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "processed")
    with pytest.raises(DatasetValidationError, match="Missing required class folder"):
        manager.validate_raw_dataset()


def test_empty_class_folder(tmp_path: Path):
    """Verifies error if a class folder exists but contains 0 valid images."""
    raw_dir = tmp_path / "raw"
    for c in DatasetManager.EXPECTED_CLASSES:
        cls_dir = raw_dir / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        if c != "trash":
            _create_dummy_image(cls_dir / "img1.jpg")

    # 'trash' folder exists but is empty
    manager = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "processed")
    with pytest.raises(DatasetValidationError, match="contains 0 valid, readable images"):
        manager.validate_raw_dataset()


def test_unsupported_and_corrupted_files_filtered(tmp_path: Path):
    """Verifies that non-images, empty files, and corrupted files are appropriately handled."""
    raw_dir = tmp_path / "raw"
    for c in DatasetManager.EXPECTED_CLASSES:
        cls_dir = raw_dir / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        _create_dummy_image(cls_dir / "valid_1.jpg")
        _create_dummy_image(cls_dir / "valid_2.png")

    # Add unsupported file to cardboard
    (raw_dir / "cardboard" / "notes.txt").write_text("not an image")
    (raw_dir / "cardboard" / ".DS_Store").write_text("hidden file")

    # Add 0-byte file to plastic
    (raw_dir / "plastic" / "zero_byte.jpg").touch()

    # Add corrupted binary file to glass
    (raw_dir / "glass" / "corrupt.jpg").write_bytes(b"INVALID_IMAGE_BYTES_XYZ")

    # Add undersized image (< 32x32) to metal
    _create_dummy_image(raw_dir / "metal" / "tiny.jpg", width=10, height=10)

    manager = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "processed")
    report = manager.validate_raw_dataset()

    assert report["total_valid"] == 12  # 2 valid per class across 6 classes
    assert len(report["unsupported_files"]) == 1  # notes.txt (.DS_Store starts with dot, skipped)
    assert len(report["corrupted_files"]) == 3  # zero_byte, corrupt.jpg, tiny.jpg
    assert report["counts_by_class"]["cardboard"] == 2
    assert report["counts_by_class"]["plastic"] == 2
    assert report["counts_by_class"]["glass"] == 2
    assert report["counts_by_class"]["metal"] == 2


def test_compute_class_weights():
    """Verifies balanced inverse-frequency class weights formula."""
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    # 5 classes with 100 samples, 1 class (trash) with 20 samples -> total = 520, K = 6
    counts = {
        "cardboard": 100,
        "glass": 100,
        "metal": 100,
        "paper": 100,
        "plastic": 100,
        "trash": 20,
    }

    weights = DatasetManager.compute_class_weights(counts, classes)

    # w_cardboard = 520 / (6 * 100) = 520 / 600 = 0.8667
    # w_trash = 520 / (6 * 20) = 520 / 120 = 4.3333
    assert np.isclose(weights["cardboard"], 0.8667, atol=1e-3)
    assert np.isclose(weights["trash"], 4.3333, atol=1e-3)
    assert weights["trash"] > weights["cardboard"]


def test_stratified_split_proportions_and_no_leakage(tmp_path: Path):
    """Verifies stratified split preserves ratios and guarantees strictly disjoint sets."""
    raw_dir = tmp_path / "raw"
    valid_by_class: dict[str, list[Path]] = {}

    # Create 20 sample images per class
    for c in DatasetManager.EXPECTED_CLASSES:
        cls_dir = raw_dir / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        img_paths = []
        for i in range(20):
            p = cls_dir / f"img_{i:03d}.jpg"
            _create_dummy_image(p)
            img_paths.append(p)
        valid_by_class[c] = img_paths

    manager = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "processed", seed=42)
    splits = manager.split_dataset(valid_by_class, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    assert set(splits.keys()) == {"train", "val", "test"}

    for c in DatasetManager.EXPECTED_CLASSES:
        train_set = set(splits["train"][c])
        val_set = set(splits["val"][c])
        test_set = set(splits["test"][c])

        # Strictly disjoint assertions (Zero Data Leakage)
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set.intersection(test_set)) == 0
        assert len(val_set.intersection(test_set)) == 0

        # Exact total count preservation
        assert len(train_set) + len(val_set) + len(test_set) == 20
        # 70% of 20 = 14, 15% of 20 = 3, 15% of 20 = 3
        assert len(train_set) == 14
        assert len(val_set) == 3
        assert len(test_set) == 3


def test_seed_reproducibility(tmp_path: Path):
    """Verifies that setting identical seeds yields exact same splits."""
    raw_dir = tmp_path / "raw"
    valid_by_class: dict[str, list[Path]] = {}
    for c in DatasetManager.EXPECTED_CLASSES:
        cls_dir = raw_dir / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        img_paths = []
        for i in range(10):
            p = cls_dir / f"img_{i:02d}.jpg"
            _create_dummy_image(p)
            img_paths.append(p)
        valid_by_class[c] = img_paths

    manager_a = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "proc_a", seed=42)
    splits_a = manager_a.split_dataset(valid_by_class)

    manager_b = DatasetManager(raw_dir=raw_dir, output_dir=tmp_path / "proc_b", seed=42)
    splits_b = manager_b.split_dataset(valid_by_class)

    for split in ["train", "val", "test"]:
        for c in DatasetManager.EXPECTED_CLASSES:
            paths_a = [p.name for p in splits_a[split][c]]
            paths_b = [p.name for p in splits_b[split][c]]
            assert paths_a == paths_b


def test_prepare_dataset_splits_end_to_end(tmp_path: Path):
    """Integration test: validates, splits, copies files, and creates manifest JSON."""
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"

    # Create 6 valid images per class
    for c in DatasetManager.EXPECTED_CLASSES:
        cls_dir = raw_dir / c
        for i in range(6):
            _create_dummy_image(cls_dir / f"{c}_{i}.jpg")

    manager = DatasetManager(raw_dir=raw_dir, output_dir=proc_dir, seed=42)
    result = manager.prepare_dataset_splits(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    manifest_path = proc_dir / "dataset_manifest.json"
    assert manifest_path.exists()
    assert result["manifest_path"] == manifest_path

    # Verify manifest contents
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["dataset_name"] == "TrashNet"
    assert manifest["total_raw_valid"] == 36
    assert len(manifest["classes"]) == 6
    assert "balanced_class_weights" in manifest

    # Verify destination directories have actual images
    for split in ["train", "val", "test"]:
        for c in DatasetManager.EXPECTED_CLASSES:
            dest_folder = proc_dir / split / c
            assert dest_folder.exists()
            assert any(dest_folder.iterdir())
