"""Predictor service coordinating preprocessor, model, and decision engine.

Provides high-level APIs for both single-image and batch directory waste classification.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import SystemConfig, load_config
from src.decision_engine import DecisionEngine, SegregationResult
from src.history_logger import HistoryLogger
from src.model import TORCH_AVAILABLE, WasteMobileNetV2, create_model
from src.preprocessor import ImagePreprocessor, ImageValidationError


class WastePredictor:
    """End-to-end inference engine for waste detection and segregation recommendations.

    Attributes:
        config: System configuration instance.
        preprocessor: OpenCV image preprocessing pipeline with ImageNet normalization.
        decision_engine: Domain rules engine for semantic categories.
        logger: Prediction history logger.
        model: MobileNetV2 classification model.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config: SystemConfig | None = None,
        enable_logging: bool = True,
        mock_mode: bool = False,
    ):
        self.config = config or load_config()
        self.preprocessor = ImagePreprocessor(target_size=self.config.model.image_size)
        self.decision_engine = DecisionEngine(config=self.config)
        self.logger = HistoryLogger(config=self.config) if enable_logging else None
        self.classes = self.config.classes
        self.mock_mode = mock_mode

        if not mock_mode and TORCH_AVAILABLE:
            self.model = create_model(
                num_classes=len(self.classes),
                dropout_rate=self.config.model.dropout_rate,
                pretrained=True,
            )
            # Load weights if a valid path is provided and exists
            weights_file = Path(model_path or self.config.model.saved_model_path)
            if weights_file.exists():
                self.model.load_weights(weights_file)
            self.model.eval()
        else:
            self.model = None

    def predict_image(
        self,
        image_path: str | Path,
        threshold: float | None = None,
    ) -> SegregationResult:
        """Executes full classification pipeline on a single waste item image.

        Args:
            image_path: Path to the image file.
            threshold: Optional confidence threshold override.

        Returns:
            SegregationResult containing predicted category, advice, and confidence.
        """
        path = Path(image_path)

        # 1. Preprocess and validate image (converts BGR->RGB, scales, applies ImageNet mean/std)
        batch_tensor = self.preprocessor.prepare_for_inference(path, channel_first=True)

        # 2. Forward pass through neural network
        if self.mock_mode or self.model is None:
            # Deterministic mock probabilities based on file hash for testing without GPU/weights
            seed = sum(ord(c) for c in str(path.name)) % 1000
            rng = np.random.default_rng(seed)
            raw_logits = rng.uniform(0.1, 1.0, size=len(self.classes))
            # Boost top class to simulate a confident prediction
            raw_logits[seed % len(self.classes)] += 2.0
            probs = np.exp(raw_logits) / np.sum(np.exp(raw_logits))
        else:
            probs = self.model.predict_probabilities(batch_tensor)

        # 3. Derive segregation advice via decision engine
        result = self.decision_engine.evaluate(probs, threshold=threshold)

        # 4. Audit logging
        if self.logger:
            self.logger.log_prediction(path, result)

        return result

    def predict_batch(
        self,
        directory_path: str | Path,
        threshold: float | None = None,
        output_csv: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Processes all valid images in a target directory.

        Args:
            directory_path: Directory containing images.
            threshold: Optional confidence threshold override.
            output_csv: Optional destination path to write batch results.

        Returns:
            List of dictionaries with inference outcomes for each image.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Batch directory not found: '{dir_path.resolve()}'")

        image_files: list[Path] = []
        for ext in ImagePreprocessor.ALLOWED_EXTENSIONS:
            image_files.extend(dir_path.glob(f"*{ext}"))
            image_files.extend(dir_path.glob(f"*{ext.upper()}"))

        image_files = sorted(set(image_files))
        results: list[dict[str, Any]] = []

        for img_p in image_files:
            try:
                seg_res = self.predict_image(img_p, threshold=threshold)
                record = {
                    "image": img_p.name,
                    "status": seg_res.status,
                    "predicted_class": seg_res.predicted_class,
                    "confidence": round(seg_res.confidence, 4),
                    "semantic_category": seg_res.semantic_category,
                    "display_color_hint": seg_res.display_color_hint,
                    "is_recyclable": seg_res.is_recyclable,
                    "action": seg_res.action,
                    "error": None,
                }
            except Exception as e:
                record = {
                    "image": img_p.name,
                    "status": "ERROR",
                    "predicted_class": "N/A",
                    "confidence": 0.0,
                    "semantic_category": "N/A",
                    "display_color_hint": "N/A",
                    "is_recyclable": False,
                    "action": "Inspection failed.",
                    "error": str(e),
                }
            results.append(record)

        # Write to output CSV if requested
        if output_csv and results:
            out_path = Path(output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(results[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

        return results
