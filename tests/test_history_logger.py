"""Unit tests for the HistoryLogger module."""

from pathlib import Path
import pytest

from src.decision_engine import SegregationResult
from src.history_logger import HistoryLogger


@pytest.fixture
def temp_logger(tmp_path: Path) -> HistoryLogger:
    csv_file = tmp_path / "test_history.csv"
    return HistoryLogger(history_file=csv_file)


def test_log_and_read_statistics(temp_logger: HistoryLogger):
    # Initial stats should be 0
    stats_empty = temp_logger.get_statistics()
    assert stats_empty["total_predictions"] == 0

    # Log two predictions
    res1 = SegregationResult(
        predicted_class="plastic",
        confidence=0.92,
        status="CONFIDENT",
        semantic_category="Recyclable",
        display_color_hint="Yellow",
        action="Rinse clean",
        is_recyclable=True,
    )
    temp_logger.log_prediction("bottle.jpg", res1)

    res2 = SegregationResult(
        predicted_class="trash",
        confidence=0.88,
        status="CONFIDENT",
        semantic_category="General / Non-recyclable",
        display_color_hint="Black",
        action="Dispose in general waste",
        is_recyclable=False,
    )
    temp_logger.log_prediction("snack_wrapper.jpg", res2)

    # Check updated stats
    stats = temp_logger.get_statistics()
    assert stats["total_predictions"] == 2
    assert stats["recyclable_items"] == 1
    assert stats["non_recyclable_items"] == 1
    assert stats["recyclable_percentage"] == 50.0
    assert stats["average_confidence"] == 0.90
    assert stats["class_counts"]["plastic"] == 1
    assert stats["class_counts"]["trash"] == 1
