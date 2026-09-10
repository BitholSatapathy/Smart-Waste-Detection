"""Image validation and preprocessing module for Smart Waste Detection & Segregation.

Uses OpenCV and NumPy to safely inspect, validate, decode, resize, and normalize
waste item images before feeding them into the deep learning classifier.
Enforces standard ImageNet normalization to match pre-trained MobileNetV2 weights.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union
import cv2
import numpy as np


class ImageValidationError(Exception):
    """Custom exception raised when an input image fails validation checks."""
    pass


class ImagePreprocessor:
    """Validates and preprocesses image files for neural network inference and training.

    Standard ImageNet normalization constants:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

    Attributes:
        target_size: Desired (width, height) tuple for model input. Default (224, 224).
        allowed_extensions: Set of supported image file extensions.
        min_dimension: Minimum acceptable width or height in pixels.
    """

    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    MIN_DIMENSION = 32

    # Standard ImageNet distribution parameters required by pre-trained MobileNetV2
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def validate_image_path(self, image_path: str | Path) -> Path:
        """Validates that a given file exists, is non-empty, and has an allowed image extension.

        Args:
            image_path: Path to the image file.

        Returns:
            Resolved Path object.

        Raises:
            ImageValidationError: If the file is missing, empty, or not a supported image format.
        """
        path = Path(image_path)
        if not path.exists():
            raise ImageValidationError(f"Image file does not exist at: '{path.resolve()}'")

        if not path.is_file():
            raise ImageValidationError(f"Expected a file path, but found a directory: '{path.resolve()}'")

        if path.stat().st_size == 0:
            raise ImageValidationError(f"Image file is empty (0 bytes): '{path.name}'")

        suffix = path.suffix.lower()
        if suffix not in self.ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported image extension '{suffix}'. "
                f"Allowed extensions: {', '.join(sorted(self.ALLOWED_EXTENSIONS))}"
            )

        return path

    def load_and_decode(self, image_path: str | Path) -> np.ndarray:
        """Reads and decodes an image from disk using OpenCV.

        1. Validates path.
        2. Reads file bytes (safe for Unicode paths on Windows).
        3. Decodes image with cv2.imdecode.
        4. Verifies spatial resolution.
        5. Converts BGR color space to RGB.

        Args:
            image_path: Path to the image.

        Returns:
            NumPy array representing the image in RGB color space.

        Raises:
            ImageValidationError: If the image cannot be decoded or has invalid dimensions.
        """
        valid_path = self.validate_image_path(image_path)

        try:
            with open(valid_path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            bgr_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ImageValidationError(f"Failed to read image file bytes: {e}")

        if bgr_image is None or bgr_image.size == 0:
            raise ImageValidationError(f"Corrupted or invalid image file. Could not decode: '{valid_path.name}'")

        h, w = bgr_image.shape[:2]
        if h < self.MIN_DIMENSION or w < self.MIN_DIMENSION:
            raise ImageValidationError(
                f"Image resolution too low ({w}x{h}). Minimum required dimension is "
                f"{self.MIN_DIMENSION}x{self.MIN_DIMENSION} pixels."
            )

        # Step 2: Convert BGR (OpenCV default) to RGB
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return rgb_image

    def preprocess_image(
        self, image: np.ndarray, channel_first: bool = True
    ) -> np.ndarray:
        """Resizes, scales, and normalizes an RGB image array for MobileNetV2.

        Transformation Pipeline:
        1. Input: RGB image array (H, W, 3).
        2. Resize to target dimensions (224, 224) via bilinear interpolation.
        3. Convert to float32.
        4. Scale pixel values to [0.0, 1.0].
        5. Apply ImageNet normalization: (scaled - mean) / std.
        6. Transpose to channel-first (3, 224, 224) for PyTorch if channel_first is True.

        Args:
            image: Decoded RGB image array with shape (H, W, 3).
            channel_first: If True, transposes to (C, H, W) for PyTorch;
                           If False, leaves as (H, W, C).

        Returns:
            Normalized float32 array standardized for MobileNetV2.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be a NumPy array, got {type(image)}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ImageValidationError(
                f"Expected RGB image with 3 channels, got shape {image.shape}"
            )

        # 1. Resize to target (224, 224)
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

        # 2. Convert to float32
        img_float = resized.astype(np.float32)

        # 3. Scale pixel values to [0.0, 1.0]
        scaled = img_float / 255.0

        # 4. Apply ImageNet normalization: (scaled - mean) / std
        normalized = (scaled - self.IMAGENET_MEAN) / self.IMAGENET_STD

        if channel_first:
            # (H, W, C) -> (C, H, W)
            return np.transpose(normalized, (2, 0, 1))
        return normalized

    def prepare_for_inference(
        self, image_path: str | Path, channel_first: bool = True
    ) -> np.ndarray:
        """Full pipeline: validates, loads, resizes, normalizes, and adds batch dimension.

        Args:
            image_path: Path to the image file.
            channel_first: Shape format (True for PyTorch (1, C, H, W), False for (1, H, W, C)).

        Returns:
            Preprocessed 4D batch tensor of shape (1, 3, 224, 224) or (1, 224, 224, 3).
        """
        rgb = self.load_and_decode(image_path)
        tensor = self.preprocess_image(rgb, channel_first=channel_first)
        # Add batch dimension (axis 0) -> (1, C, H, W)
        batch_tensor = np.expand_dims(tensor, axis=0)
        return batch_tensor
