"""Integration tests for the CLI interface."""

from pathlib import Path
import subprocess
import sys
import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Smart Waste Detection & Segregation System CLI" in result.stdout
    assert "predict" in result.stdout
    assert "batch-predict" in result.stdout
    assert "train" in result.stdout
    assert "stats" in result.stdout


def test_cli_train_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "train", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert "--epochs" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--learning-rate" in result.stdout


def test_cli_demo_setup(tmp_path: Path):
    demo_dir = tmp_path / "test_samples"
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "demo-setup", "--output-dir", str(demo_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (demo_dir / "plastic_bottle.jpg").exists()
    assert (demo_dir / "cardboard_box.jpg").exists()
    assert (demo_dir / "metal_can.jpg").exists()


def test_cli_predict_mock(tmp_path: Path):
    demo_dir = tmp_path / "test_samples"
    # Generate samples first
    subprocess.run(
        [sys.executable, "-m", "src.cli", "demo-setup", "--output-dir", str(demo_dir)],
        check=True,
    )

    sample_img = demo_dir / "plastic_bottle.jpg"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "predict",
            "--image",
            str(sample_img),
            "--mock",
            "--no-log",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Predicted Class:" in result.stdout
    assert "Confidence:" in result.stdout
    assert "Semantic Category:" in result.stdout


def test_cli_batch_predict_mock(tmp_path: Path):
    demo_dir = tmp_path / "test_samples"
    report_csv = tmp_path / "batch_out.csv"

    subprocess.run(
        [sys.executable, "-m", "src.cli", "demo-setup", "--output-dir", str(demo_dir)],
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "batch-predict",
            "--input-dir",
            str(demo_dir),
            "--output-report",
            str(report_csv),
            "--mock",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert report_csv.exists()
    assert "Processed: 6 images" in result.stdout


def test_cli_prepare_data_help():
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "prepare-data", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--raw-dir" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--train-ratio" in result.stdout
    assert "--seed" in result.stdout


def test_cli_prepare_data_execution(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"

    # Create dummy raw data for 6 classes
    for c in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        cls_dir = raw_dir / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        # Create 3 tiny test images
        import cv2
        import numpy as np
        for i in range(3):
            img = np.full((50, 50, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(cls_dir / f"img_{i}.jpg"), img)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "prepare-data",
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(proc_dir),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Dataset Preparation & Validation" in result.stdout
    assert (proc_dir / "dataset_manifest.json").exists()

