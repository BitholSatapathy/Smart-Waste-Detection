"""Unit tests for the DecisionEngine module."""

import numpy as np
import pytest

from src.config import load_config
from src.decision_engine import DecisionEngine, SegregationResult


@pytest.fixture
def engine() -> DecisionEngine:
    config = load_config()
    return DecisionEngine(config=config)


def test_evaluate_confident_plastic(engine: DecisionEngine):
    probs = {
        "cardboard": 0.05,
        "glass": 0.05,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.75,
        "trash": 0.05,
    }
    result = engine.evaluate(probs, threshold=0.60)
    assert isinstance(result, SegregationResult)
    assert result.predicted_class == "plastic"
    assert result.confidence == 0.75
    assert result.status == "CONFIDENT"
    assert result.semantic_category == "Recyclable"
    assert result.is_recyclable is True
    assert result.display_color_hint == "Yellow"
    assert "Empty liquid contents" in result.action
    assert "Recyclable (Yellow)" in result.target_bin


def test_evaluate_confident_trash(engine: DecisionEngine):
    probs = {
        "cardboard": 0.02,
        "glass": 0.03,
        "metal": 0.05,
        "paper": 0.05,
        "plastic": 0.05,
        "trash": 0.80,
    }
    result = engine.evaluate(probs, threshold=0.60)
    assert result.predicted_class == "trash"
    assert result.status == "CONFIDENT"
    assert result.semantic_category == "General / Non-recyclable"
    assert result.is_recyclable is False
    assert result.display_color_hint == "Black"


def test_evaluate_uncertain_below_heuristic_threshold(engine: DecisionEngine):
    probs = {
        "cardboard": 0.20,
        "glass": 0.15,
        "metal": 0.15,
        "paper": 0.35,
        "plastic": 0.10,
        "trash": 0.05,
    }
    result = engine.evaluate(probs, threshold=0.60)
    assert result.predicted_class == "paper"
    assert result.confidence == 0.35
    assert result.status == "UNCERTAIN"
    assert result.semantic_category == "Manual Verification Required"
    assert "below the initial configurable heuristic threshold" in result.action


def test_evaluate_with_list_input(engine: DecisionEngine):
    # order: cardboard, glass, metal, paper, plastic, trash
    probs = [0.05, 0.80, 0.05, 0.04, 0.03, 0.03]
    result = engine.evaluate(probs, threshold=0.50)
    assert result.predicted_class == "glass"
    assert result.confidence == 0.80
    assert result.status == "CONFIDENT"
    assert result.semantic_category == "Recyclable"


def test_evaluate_with_numpy_array(engine: DecisionEngine):
    probs = np.array([0.90, 0.02, 0.02, 0.02, 0.02, 0.02])
    result = engine.evaluate(probs, threshold=0.60)
    assert result.predicted_class == "cardboard"
    assert result.status == "CONFIDENT"
    assert result.semantic_category == "Recyclable"


def test_evaluate_invalid_list_length(engine: DecisionEngine):
    with pytest.raises(ValueError, match="Expected 6 probabilities"):
        engine.evaluate([0.5, 0.5])


def test_evaluate_empty_dict(engine: DecisionEngine):
    with pytest.raises(ValueError, match="Empty probabilities"):
        engine.evaluate({})


def test_result_to_dict(engine: DecisionEngine):
    probs = {"cardboard": 0.1, "glass": 0.1, "metal": 0.7, "paper": 0.05, "plastic": 0.03, "trash": 0.02}
    result = engine.evaluate(probs)
    d = result.to_dict()
    assert d["predicted_class"] == "metal"
    assert d["confidence"] == 0.7
    assert d["status"] == "CONFIDENT"
    assert d["semantic_category"] == "Recyclable"
    assert d["is_recyclable"] is True
    assert "display_color_hint" in d
    assert "probabilities" in d
