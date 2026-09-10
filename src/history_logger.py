"""Prediction history logger and analytics module.

Maintains an audit trail of all waste classifications and computes aggregate
statistics (e.g. recycling rates, class frequencies, uncertainty ratios)
for system auditing and environmental reporting.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import SystemConfig, load_config
from src.decision_engine import SegregationResult


class HistoryLogger:
    """Manages CSV logging and analytical summaries of waste classification events."""

    CSV_HEADERS = [
        "timestamp",
        "image_path",
        "predicted_class",
        "confidence",
        "status",
        "semantic_category",
        "display_color_hint",
        "is_recyclable",
    ]

    def __init__(self, history_file: str | Path | None = None, config: SystemConfig | None = None):
        self.config = config or load_config()
        self.history_file = Path(history_file or self.config.history_csv)
        self._ensure_file_initialized()

    def _ensure_file_initialized(self) -> None:
        """Creates the history CSV file and writes header if it does not yet exist."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists() or self.history_file.stat().st_size == 0:
            with open(self.history_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADERS)

    def log_prediction(
        self,
        image_path: str | Path,
        result: SegregationResult,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Appends a prediction record to the history log.

        Args:
            image_path: Path or filename of the evaluated waste image.
            result: The SegregationResult produced by the decision engine.
            timestamp: Optional ISO timestamp string. Defaults to current UTC time.

        Returns:
            Dictionary representing the logged row.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "timestamp": timestamp,
            "image_path": str(Path(image_path).name),
            "predicted_class": result.predicted_class,
            "confidence": round(result.confidence, 4),
            "status": result.status,
            "semantic_category": result.semantic_category,
            "display_color_hint": result.display_color_hint,
            "is_recyclable": result.is_recyclable,
        }

        with open(self.history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADERS)
            writer.writerow(row)

        return row

    def get_statistics(self) -> dict[str, Any]:
        """Computes summary statistics across all logged predictions.

        Returns:
            Dictionary containing metrics, or an empty summary if no records exist.
        """
        if not self.history_file.exists() or self.history_file.stat().st_size == 0:
            return {"total_predictions": 0, "message": "No prediction history found."}

        try:
            df = pd.read_csv(self.history_file)
        except Exception:
            return {"total_predictions": 0, "message": "Failed to read history CSV."}

        if df.empty:
            return {"total_predictions": 0, "message": "History file is empty."}

        total = len(df)
        class_counts = df["predicted_class"].value_counts().to_dict()
        class_percentages = {cls: round((count / total) * 100, 2) for cls, count in class_counts.items()}

        status_counts = df["status"].value_counts().to_dict()
        recyclable_count = int(df["is_recyclable"].sum()) if "is_recyclable" in df else 0
        avg_confidence = float(df["confidence"].mean()) if "confidence" in df else 0.0

        return {
            "total_predictions": total,
            "average_confidence": round(avg_confidence, 4),
            "recyclable_items": recyclable_count,
            "non_recyclable_items": total - recyclable_count,
            "recyclable_percentage": round((recyclable_count / total) * 100, 2),
            "status_breakdown": status_counts,
            "class_counts": class_counts,
            "class_percentages": class_percentages,
        }
