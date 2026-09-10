"""Configuration loader module for Smart Waste Detection & Segregation System.

Reads configuration settings from configs/config.yaml and exposes clean,
reusable accessors with safe defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List
import yaml


@dataclass
class DatasetConfig:
    """Configuration for raw dataset ingestion and stratified partitioning."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    use_class_weights: bool = True


@dataclass
class ModelConfig:
    """Hyperparameters and file paths for the MobileNetV2 classifier."""
    architecture: str = "MobileNetV2"
    image_size: tuple[int, int] = (224, 224)
    num_classes: int = 6
    batch_size: int = 32
    learning_rate: float = 0.0005
    epochs: int = 15
    dropout_rate: float = 0.3
    saved_model_path: str = "models/saved_models/mobilenetv2_waste.pth"
    class_indices_path: str = "models/class_indices.json"


@dataclass
class SegregationRule:
    """Domain segregation directive for a waste class.

    Attributes:
        semantic_category: High-level semantic category (e.g., 'Recyclable', 'General / Non-recyclable').
        display_color_hint: Optional internal visualization color hint (NOT a universal municipal standard).
        action: Clear, actionable disposal advice for end-users.
        is_recyclable: Boolean flag indicating recyclable classification.
    """
    semantic_category: str
    display_color_hint: str
    action: str
    is_recyclable: bool


@dataclass
class SystemConfig:
    """Overall system configuration."""
    project_name: str = "Smart Waste Detection & Segregation System"
    version: str = "1.0.0"
    classes: list[str] = field(
        default_factory=lambda: [
            "cardboard",
            "glass",
            "metal",
            "paper",
            "plastic",
            "trash",
        ]
    )
    # Initial configurable heuristic threshold. Actual threshold should be tuned based on deployment criteria.
    default_confidence_threshold: float = 0.60
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    category_mapping: dict[str, SegregationRule] = field(default_factory=dict)
    history_csv: str = "reports/history.csv"
    confusion_matrix_plot: str = "reports/confusion_matrix.png"
    classification_report_json: str = "reports/classification_report.json"
    sample_images_dir: str = "data/sample_test_images"


def get_default_category_mapping() -> dict[str, SegregationRule]:
    """Provides default domain rules with semantic categorization.

    Note: Waste disposal regulations and physical bin colors vary significantly by
    municipality. display_color_hint is an optional internal UI visualization hint only.
    """
    return {
        "cardboard": SegregationRule(
            semantic_category="Recyclable",
            display_color_hint="Blue",
            action="Flatten box to save volume. Ensure dry and free of food/grease contamination.",
            is_recyclable=True,
        ),
        "glass": SegregationRule(
            semantic_category="Recyclable",
            display_color_hint="Teal",
            action="Rinse container clean. Handle with care to prevent shattering. Do not mix ceramics or lightbulbs.",
            is_recyclable=True,
        ),
        "metal": SegregationRule(
            semantic_category="Recyclable",
            display_color_hint="Grey",
            action="Rinse food or beverage cans. Crush aluminum cans if feasible to optimize recycling volume.",
            is_recyclable=True,
        ),
        "paper": SegregationRule(
            semantic_category="Recyclable",
            display_color_hint="Blue",
            action="Keep dry and clean. Shred confidential documents before recycling. Soiled paper goes to General Waste.",
            is_recyclable=True,
        ),
        "plastic": SegregationRule(
            semantic_category="Recyclable",
            display_color_hint="Yellow",
            action="Empty liquid contents, rinse clean, and reattach screw cap. Check local resin code policies.",
            is_recyclable=True,
        ),
        "trash": SegregationRule(
            semantic_category="General / Non-recyclable",
            display_color_hint="Black",
            action="Residual non-recyclable refuse. Wrap or bag securely to maintain sanitary conditions.",
            is_recyclable=False,
        ),
    }


def find_project_root() -> Path:
    """Finds project root directory by searching for marker files."""
    current = Path(__file__).resolve().parent
    for parent in [current, current.parent]:
        if (parent / "configs" / "config.yaml").exists() or (parent / "requirements.txt").exists():
            return parent
    return Path.cwd()


def load_config(config_path: str | Path | None = None) -> SystemConfig:
    """Loads configuration from YAML file, falling back to sensible defaults."""
    if config_path is None:
        root = find_project_root()
        config_path = root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    config = SystemConfig()
    config.category_mapping = get_default_category_mapping()

    if not config_path.exists():
        return config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        # Project metadata
        proj = raw_data.get("project", {})
        config.project_name = proj.get("name", config.project_name)
        config.version = proj.get("version", config.version)

        # Classes
        if "classes" in raw_data and isinstance(raw_data["classes"], list):
            config.classes = [str(c).lower().strip() for c in raw_data["classes"]]

        # Dataset configs
        d_data = raw_data.get("dataset", {})
        config.dataset = DatasetConfig(
            raw_dir=d_data.get("raw_dir", "data/raw"),
            processed_dir=d_data.get("processed_dir", "data/processed"),
            train_ratio=float(d_data.get("train_ratio", 0.70)),
            val_ratio=float(d_data.get("val_ratio", 0.15)),
            test_ratio=float(d_data.get("test_ratio", 0.15)),
            random_seed=int(d_data.get("random_seed", 42)),
            use_class_weights=bool(d_data.get("use_class_weights", True)),
        )

        # Model configs
        m_data = raw_data.get("model", {})
        img_size = m_data.get("image_size", [224, 224])
        config.model = ModelConfig(
            architecture=m_data.get("architecture", "MobileNetV2"),
            image_size=(int(img_size[0]), int(img_size[1])),
            num_classes=int(m_data.get("num_classes", len(config.classes))),
            batch_size=int(m_data.get("batch_size", 32)),
            learning_rate=float(m_data.get("learning_rate", 0.0005)),
            epochs=int(m_data.get("epochs", 15)),
            dropout_rate=float(m_data.get("dropout_rate", 0.3)),
            saved_model_path=m_data.get("saved_model_path", "models/saved_models/mobilenetv2_waste.pth"),
            class_indices_path=m_data.get("class_indices_path", "models/class_indices.json"),
        )

        # Decision engine
        de_data = raw_data.get("decision_engine", {})
        config.default_confidence_threshold = float(
            de_data.get("default_confidence_threshold", config.default_confidence_threshold)
        )
        
        # Load category mapping (supporting both 'category_mapping' and legacy 'bin_mapping')
        mapping_data = de_data.get("category_mapping") or de_data.get("bin_mapping")
        if mapping_data and isinstance(mapping_data, dict):
            mapping = {}
            for label, item in mapping_data.items():
                semantic_cat = item.get("semantic_category") or item.get("waste_stream") or "General / Non-recyclable"
                color_hint = item.get("display_color_hint") or item.get("target_bin", "")
                mapping[label.lower()] = SegregationRule(
                    semantic_category=semantic_cat,
                    display_color_hint=color_hint,
                    action=item.get("action", "Segregate according to local regulations."),
                    is_recyclable=bool(item.get("is_recyclable", False)),
                )
            config.category_mapping = mapping

        # Paths
        p_data = raw_data.get("paths", {})
        config.history_csv = p_data.get("history_csv", config.history_csv)
        config.confusion_matrix_plot = p_data.get("confusion_matrix_plot", config.confusion_matrix_plot)
        config.classification_report_json = p_data.get("classification_report_json", config.classification_report_json)
        config.sample_images_dir = p_data.get("sample_images_dir", config.sample_images_dir)

    except Exception as e:
        print(f"Warning: Failed to parse config file '{config_path}': {e}. Using defaults.")

    return config
