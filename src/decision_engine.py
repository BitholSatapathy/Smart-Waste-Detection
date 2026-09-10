"""Waste Segregation Decision Engine.

Translates neural network class probabilities into actionable waste
segregation directives, semantic categories (Recyclable, General / Non-recyclable),
optional display color hints, and handling advice.
Enforces uncertainty gating for low-confidence classifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import numpy as np

from src.config import SegregationRule, SystemConfig, load_config


@dataclass
class SegregationResult:
    """Encapsulates the complete decision and guidance for an evaluated waste item.

    Attributes:
        predicted_class: Predicted waste item material class (e.g., 'plastic').
        confidence: Top-1 predicted probability score in [0.0, 1.0].
        status: 'CONFIDENT' if confidence >= threshold, else 'UNCERTAIN'.
        semantic_category: Primary semantic category ('Recyclable' or 'General / Non-recyclable').
        action: Clear, actionable disposal advice for the user.
        is_recyclable: True if item belongs to recyclable material stream.
        display_color_hint: Optional internal UI color tag (NOT a universal municipal standard).
        probabilities: Full probability distribution across all recognized classes.
    """
    predicted_class: str
    confidence: float
    status: str  # "CONFIDENT" or "UNCERTAIN"
    semantic_category: str
    action: str
    is_recyclable: bool
    display_color_hint: str = ""
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def waste_stream(self) -> str:
        """Alias for semantic_category for backward compatibility."""
        return self.semantic_category

    @property
    def target_bin(self) -> str:
        """Display label combining semantic category and optional color hint."""
        if self.display_color_hint:
            return f"{self.semantic_category} ({self.display_color_hint})"
        return self.semantic_category

    def to_dict(self) -> dict:
        """Converts result to a dictionary suitable for JSON serialization or logging."""
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "semantic_category": self.semantic_category,
            "display_color_hint": self.display_color_hint,
            "target_bin": self.target_bin,
            "action": self.action,
            "is_recyclable": self.is_recyclable,
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
        }


class DecisionEngine:
    """Evaluates classification probabilities against domain segregation rules.

    Primary segregation categories are semantic (Recyclable vs. General / Non-recyclable).
    Color hints are treated strictly as optional internal UI visualization cues, as physical
    bin coloring varies across different municipalities and regional jurisdictions.
    """

    def __init__(self, config: SystemConfig | None = None):
        self.config = config or load_config()
        self.rules = self.config.category_mapping
        self.classes = self.config.classes
        # Initial configurable heuristic threshold
        self.default_threshold = self.config.default_confidence_threshold

    def evaluate(
        self,
        probabilities: dict[str, float] | list[float] | np.ndarray,
        threshold: float | None = None,
    ) -> SegregationResult:
        """Evaluates prediction probabilities and derives segregation recommendation.

        Args:
            probabilities: Probability distribution either as a dictionary of
                           {class_name: score}, or an ordered list/array of scores
                           matching self.classes.
            threshold: Confidence cutoff (0.0 to 1.0). Defaults to the initial
                       configurable heuristic in config.yaml.

        Returns:
            SegregationResult containing semantic category, advice, and confidence.
        """
        cutoff = threshold if threshold is not None else self.default_threshold

        # Standardize probabilities into {class_name: float}
        prob_dict: dict[str, float] = {}
        if isinstance(probabilities, dict):
            prob_dict = {str(k).lower(): float(v) for k, v in probabilities.items()}
        elif isinstance(probabilities, (list, tuple, np.ndarray)):
            probs = [float(p) for p in probabilities]
            if len(probs) != len(self.classes):
                raise ValueError(
                    f"Expected {len(self.classes)} probabilities, but received {len(probs)}"
                )
            prob_dict = {cls_name: prob for cls_name, prob in zip(self.classes, probs)}
        else:
            raise TypeError(f"Unsupported probabilities type: {type(probabilities)}")

        if not prob_dict:
            raise ValueError("Empty probabilities provided to decision engine.")

        # Identify Top-1 predicted class and confidence
        top_class = max(prob_dict, key=prob_dict.get)  # type: ignore[arg-type]
        top_confidence = float(prob_dict[top_class])

        # Retrieve rule for top class (or fallback)
        rule = self.rules.get(
            top_class,
            SegregationRule(
                semantic_category="General / Non-recyclable",
                display_color_hint="Black",
                action="Unrecognized waste category. Dispose safely in general waste.",
                is_recyclable=False,
            ),
        )

        # Apply confidence thresholding using the configurable heuristic
        if top_confidence >= cutoff:
            return SegregationResult(
                predicted_class=top_class,
                confidence=top_confidence,
                status="CONFIDENT",
                semantic_category=rule.semantic_category,
                display_color_hint=rule.display_color_hint,
                action=rule.action,
                is_recyclable=rule.is_recyclable,
                probabilities=prob_dict,
            )
        else:
            # Low confidence heuristic warning gate
            return SegregationResult(
                predicted_class=top_class,
                confidence=top_confidence,
                status="UNCERTAIN",
                semantic_category="Manual Verification Required",
                display_color_hint="Yellow / Warning",
                action=(
                    f"Confidence ({top_confidence:.1%}) is below the initial configurable heuristic "
                    f"threshold ({cutoff:.1%}). Likely '{top_class}', but visual verification "
                    f"is required before segregation."
                ),
                is_recyclable=rule.is_recyclable,
                probabilities=prob_dict,
            )
