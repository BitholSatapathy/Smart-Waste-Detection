"""Dataset validation, preparation, and stratified splitting pipeline.

Provides structured ingestion, validation, and stratified splitting of the
TrashNet 6-class dataset into disjoint train, validation, and test subsets
while guarding against data leakage and class imbalance.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np

from src.preprocessor import ImagePreprocessor


class DatasetValidationError(Exception):
    """Custom exception raised when raw dataset structure or images fail validation."""
    pass


class DatasetManager:
    """Manages raw dataset validation, stratified splitting, and manifest generation."""

    EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    SUPPORTED_EXTENSIONS = ImagePreprocessor.ALLOWED_EXTENSIONS

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        output_dir: str | Path = "data/processed",
        classes: list[str] | None = None,
        seed: int = 42,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.classes = [c.lower().strip() for c in (classes or self.EXPECTED_CLASSES)]
        self.seed = seed

    def validate_raw_dataset(self) -> dict[str, Any]:
        """Scans raw dataset directory, validating folder structure and image decodability.

        Returns:
            Dictionary with per-class valid paths, corrupted paths, and unsupported files.

        Raises:
            DatasetValidationError: If directory is missing, a class folder is absent,
                                     or any class folder contains 0 valid images.
        """
        if not self.raw_dir.exists() or not self.raw_dir.is_dir():
            raise DatasetValidationError(
                f"Raw dataset directory not found at: '{self.raw_dir.resolve()}'. "
                f"Please create this directory and place the 6 class subfolders inside it."
            )

        # 1. Verify all required class folders exist
        missing_classes = []
        for cls_name in self.classes:
            cls_folder = self.raw_dir / cls_name
            if not cls_folder.exists() or not cls_folder.is_dir():
                missing_classes.append(cls_name)

        if missing_classes:
            raise DatasetValidationError(
                f"Missing required class folder(s) in '{self.raw_dir.resolve()}': "
                f"{', '.join(missing_classes)}. Expected all 6 classes: {', '.join(self.classes)}."
            )

        # 2. Inspect each class folder for valid, corrupted, and unsupported files
        valid_files_by_class: dict[str, list[Path]] = {}
        corrupted_files: list[Path] = []
        unsupported_files: list[Path] = []
        counts_by_class: dict[str, int] = {}

        for cls_name in self.classes:
            cls_folder = self.raw_dir / cls_name
            valid_list: list[Path] = []

            for file_path in sorted(cls_folder.iterdir()):
                if file_path.name.startswith(".") or file_path.is_dir():
                    continue

                suffix = file_path.suffix.lower()
                if suffix not in self.SUPPORTED_EXTENSIONS:
                    unsupported_files.append(file_path)
                    continue

                # Verify file is non-empty and readable via OpenCV
                if file_path.stat().st_size == 0:
                    corrupted_files.append(file_path)
                    continue

                try:
                    with open(file_path, "rb") as f:
                        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is None or img.size == 0:
                        corrupted_files.append(file_path)
                        continue
                    if img.shape[0] < ImagePreprocessor.MIN_DIMENSION or img.shape[1] < ImagePreprocessor.MIN_DIMENSION:
                        corrupted_files.append(file_path)
                        continue
                    valid_list.append(file_path)
                except Exception:
                    corrupted_files.append(file_path)

            if len(valid_list) == 0:
                raise DatasetValidationError(
                    f"Class folder '{cls_name}' contains 0 valid, readable images. "
                    f"At least 1 valid image is required per category."
                )

            valid_files_by_class[cls_name] = valid_list
            counts_by_class[cls_name] = len(valid_list)

        return {
            "valid_files_by_class": valid_files_by_class,
            "counts_by_class": counts_by_class,
            "corrupted_files": corrupted_files,
            "unsupported_files": unsupported_files,
            "total_valid": sum(counts_by_class.values()),
        }

    @staticmethod
    def compute_class_weights(counts_by_class: dict[str, int], class_names: list[str]) -> dict[str, float]:
        """Calculates balanced inverse-frequency class weights for loss computation.

        Formula:
            w_c = total_samples / (num_classes * count_c)

        This ensures rare classes (e.g. 'trash') receive proportionally higher penalty
        during training to counteract class imbalance.
        """
        total_samples = sum(counts_by_class.values())
        num_classes = len(class_names)
        weights: dict[str, float] = {}

        for cls_name in class_names:
            count = counts_by_class.get(cls_name, 0)
            if count > 0:
                w = total_samples / (num_classes * count)
                weights[cls_name] = round(float(w), 4)
            else:
                weights[cls_name] = 1.0

        return weights

    def split_dataset(
        self,
        valid_files_by_class: dict[str, list[Path]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict[str, dict[str, list[Path]]]:
        """Performs stratified random splitting across classes without data duplication.

        Args:
            valid_files_by_class: Valid file paths grouped by class.
            train_ratio: Fraction for training (default 0.70).
            val_ratio: Fraction for validation (default 0.15).
            test_ratio: Fraction for testing (default 0.15).

        Returns:
            Dictionary structured as:
            {
                "train": {cls: [paths...]},
                "val":   {cls: [paths...]},
                "test":  {cls: [paths...]}
            }
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-4):
            raise ValueError(
                f"Split ratios must sum to 1.0, got: {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio:.4f}"
            )

        splits: dict[str, dict[str, list[Path]]] = {
            "train": {cls: [] for cls in self.classes},
            "val": {cls: [] for cls in self.classes},
            "test": {cls: [] for cls in self.classes},
        }

        rng = random.Random(self.seed)

        for cls_name in self.classes:
            file_list = list(valid_files_by_class[cls_name])
            # Deterministic sort before shuffle guarantees identical split given the seed
            file_list.sort(key=lambda p: str(p.name))
            rng.shuffle(file_list)

            n_total = len(file_list)
            n_train = int(round(n_total * train_ratio))
            n_val = int(round(n_total * val_ratio))
            # Test split absorbs remainder to guarantee exact total count preservation
            n_test = n_total - (n_train + n_val)

            # Ensure every split receives at least 1 image if n_total >= 3
            if n_total >= 3:
                if n_train == 0:
                    n_train = 1
                if n_val == 0:
                    n_val = 1
                if n_test == 0:
                    n_test = 1
                # Readjust if sum exceeds
                while (n_train + n_val + n_test) > n_total:
                    if n_train > 1:
                        n_train -= 1
                    elif n_val > 1:
                        n_val -= 1

            train_files = file_list[:n_train]
            val_files = file_list[n_train : n_train + n_val]
            test_files = file_list[n_train + n_val : n_train + n_val + n_test]

            # Strict assertion against data leakage across splits
            train_set = set(train_files)
            val_set = set(val_files)
            test_set = set(test_files)

            assert len(train_set.intersection(val_set)) == 0, f"Data leakage between train and val in {cls_name}!"
            assert len(train_set.intersection(test_set)) == 0, f"Data leakage between train and test in {cls_name}!"
            assert len(val_set.intersection(test_set)) == 0, f"Data leakage between val and test in {cls_name}!"
            assert len(train_files) + len(val_files) + len(test_files) == n_total, f"Count mismatch in {cls_name}!"

            splits["train"][cls_name] = train_files
            splits["val"][cls_name] = val_files
            splits["test"][cls_name] = test_files

        return splits

    def prepare_dataset_splits(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict[str, Any]:
        """Validates raw data, splits stratified subsets, copies files, and creates manifest.

        Destination directories:
            output_dir/train/<class>/
            output_dir/val/<class>/
            output_dir/test/<class>/

        Also generates:
            output_dir/dataset_manifest.json
        """
        # 1. Validate raw data
        val_report = self.validate_raw_dataset()
        valid_by_class = val_report["valid_files_by_class"]
        counts_by_class = val_report["counts_by_class"]

        # 2. Compute class weights
        class_weights = self.compute_class_weights(counts_by_class, self.classes)

        # 3. Stratified split
        splits = self.split_dataset(
            valid_by_class,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        # 4. Copy files to destination partitions
        copied_counts: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}
        all_dest_paths: set[Path] = set()

        for split_name, class_dict in splits.items():
            for cls_name, paths in class_dict.items():
                dest_dir = self.output_dir / split_name / cls_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                copied_counts[split_name][cls_name] = len(paths)

                for src_path in paths:
                    dest_path = dest_dir / src_path.name
                    # Avoid accidental name collisions
                    if dest_path in all_dest_paths:
                        dest_path = dest_dir / f"{src_path.stem}_{cls_name}{src_path.suffix}"
                    all_dest_paths.add(dest_path)
                    shutil.copy2(src_path, dest_path)

        # 5. Write dataset manifest
        manifest = {
            "dataset_name": "TrashNet",
            "classes": self.classes,
            "seed": self.seed,
            "split_ratios": {
                "train": train_ratio,
                "val": val_ratio,
                "test": test_ratio,
            },
            "raw_counts": counts_by_class,
            "total_raw_valid": val_report["total_valid"],
            "split_counts": {
                "train": {
                    "total": sum(copied_counts["train"].values()),
                    "per_class": copied_counts["train"],
                },
                "val": {
                    "total": sum(copied_counts["val"].values()),
                    "per_class": copied_counts["val"],
                },
                "test": {
                    "total": sum(copied_counts["test"].values()),
                    "per_class": copied_counts["test"],
                },
            },
            "balanced_class_weights": class_weights,
            "corrupted_files_found": len(val_report["corrupted_files"]),
            "unsupported_files_found": len(val_report["unsupported_files"]),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return {
            "validation": val_report,
            "splits": splits,
            "manifest": manifest,
            "manifest_path": manifest_path,
        }
