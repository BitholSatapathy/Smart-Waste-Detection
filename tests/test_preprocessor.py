"""Unit tests for the ImagePreprocessor module."""

import os
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.preprocessor import ImagePreprocessor, ImageValidationError


@pytest.fixture
def preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(target_size=(224, 224))


@pytest.fixture
def temp_image_path(tmp_path: Path) -> Path:
    """Creates a temporary valid test image."""
    img_path = tmp_path / "valid_test.jpg"
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Give it some color patterns
    img[20:80, 20:80] = [0, 255, 128]
    cv2.imwrite(str(img_path), img)
    return img_path


def test_validate_image_path_success(preprocessor: ImagePreprocessor, temp_image_path: Path):
    resolved = preprocessor.validate_image_path(temp_image_path)
    assert resolved.exists()
    assert resolved == temp_image_path


def test_validate_image_path_missing(preprocessor: ImagePreprocessor, tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ImageValidationError, match="does not exist"):
        preprocessor.validate_image_path(non_existent)


def test_validate_image_path_invalid_extension(preprocessor: ImagePreprocessor, tmp_path: Path):
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("not an image")
    with pytest.raises(ImageValidationError, match="Unsupported image extension"):
        preprocessor.validate_image_path(txt_file)


def test_validate_image_path_empty_file(preprocessor: ImagePreprocessor, tmp_path: Path):
    empty_file = tmp_path / "empty.png"
    empty_file.touch()
    with pytest.raises(ImageValidationError, match="empty"):
        preprocessor.validate_image_path(empty_file)


def test_load_and_decode_success(preprocessor: ImagePreprocessor, temp_image_path: Path):
    rgb = preprocessor.load_and_decode(temp_image_path)
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (100, 100, 3)
    # Check RGB channel order (original was BGR [0, 255, 128], so RGB is [128, 255, 0])
    assert rgb[50, 50, 0] == 128
    assert rgb[50, 50, 1] == 255
    assert rgb[50, 50, 2] == 0


def test_load_and_decode_too_small(preprocessor: ImagePreprocessor, tmp_path: Path):
    tiny_img_path = tmp_path / "tiny.png"
    tiny_img = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(tiny_img_path), tiny_img)

    with pytest.raises(ImageValidationError, match="resolution too low"):
        preprocessor.load_and_decode(tiny_img_path)


def test_prepare_for_inference_channel_first(preprocessor: ImagePreprocessor, temp_image_path: Path):
    tensor = preprocessor.prepare_for_inference(temp_image_path, channel_first=True)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == np.float32
    # Standard ImageNet normalization produces values centered around 0 in range approx [-2.5, 3.0]
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


def test_prepare_for_inference_channel_last(preprocessor: ImagePreprocessor, temp_image_path: Path):
    tensor = preprocessor.prepare_for_inference(temp_image_path, channel_first=False)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -3.0
    assert tensor.max() <= 3.0


def test_imagenet_normalization_exact_values(preprocessor: ImagePreprocessor):
    # Pure black image (0, 0, 0)
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    norm_black = preprocessor.preprocess_image(black, channel_first=False)
    # Expected: (0 - mean) / std
    expected_r = (0.0 - 0.485) / 0.229
    expected_g = (0.0 - 0.456) / 0.224
    expected_b = (0.0 - 0.406) / 0.225
    assert np.isclose(norm_black[0, 0, 0], expected_r, atol=1e-4)
    assert np.isclose(norm_black[0, 0, 1], expected_g, atol=1e-4)
    assert np.isclose(norm_black[0, 0, 2], expected_b, atol=1e-4)
