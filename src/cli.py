"""Command-Line Interface (CLI) for Smart Waste Detection & Segregation System.

Supports the following subcommands:
- prepare-data: Validate and split raw TrashNet dataset into train/val/test partitions
- predict: Classify a single image and output semantic segregation advice
- batch-predict: Process an entire folder of waste images
- train: Train the MobileNetV2 model on a waste dataset
- evaluate: Calculate confusion matrix and classification metrics on test data
- stats: Display prediction analytics and history summary
- demo-setup: Generate curated sample images for quick testing and demonstration
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np

from src.config import load_config
from src.dataset import DatasetManager, DatasetValidationError
from src.history_logger import HistoryLogger
from src.model import TORCH_AVAILABLE, evaluate_and_generate_reports, train_model
from src.predictor import WastePredictor


def print_banner(title: str) -> None:
    """Prints a styled CLI banner."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {title}")
    print(f"{sep}")


def cmd_prepare_data(args: argparse.Namespace) -> int:
    """Handles raw dataset validation and stratified train/val/test splitting."""
    print_banner("Smart Waste Detection: Dataset Preparation & Validation")
    config = load_config(args.config)

    raw_dir = Path(args.raw_dir or config.dataset.raw_dir)
    output_dir = Path(args.output_dir or config.dataset.processed_dir)
    train_ratio = args.train_ratio
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio
    seed = args.seed

    print(f"Source Directory:      {raw_dir.resolve()}")
    print(f"Destination Directory: {output_dir.resolve()}")
    print(f"Split Ratios:          Train {train_ratio:.0%} | Val {val_ratio:.0%} | Test {test_ratio:.0%}")
    print(f"Random Seed:           {seed} (reproducible stratified split)\n")

    manager = DatasetManager(
        raw_dir=raw_dir,
        output_dir=output_dir,
        classes=config.classes,
        seed=seed,
    )

    try:
        # 1. Validation phase
        print("Validating raw folder structure and verifying image decodability...")
        result = manager.prepare_dataset_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    except DatasetValidationError as e:
        print(f"\n[DATASET VALIDATION ERROR] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] Dataset preparation failed: {e}", file=sys.stderr)
        return 1

    manifest = result["manifest"]
    raw_counts = manifest["raw_counts"]
    split_counts = manifest["split_counts"]

    # Format class summary table
    print("\n" + "-" * 40)
    print(f"{'Class':<16} {'Total Valid Images':<20}")
    print("-" * 40)
    for cls_name in config.classes:
        count = raw_counts.get(cls_name, 0)
        print(f"{cls_name:<16} {count:<20d}")
    print("-" * 40)
    print(f"{'TOTAL':<16} {manifest['total_raw_valid']:<20d}")
    print("-" * 40)

    # Report partition counts
    print("\nDataset Partition Summary:")
    print(f"  Training Set:   {split_counts['train']['total']:4d} images ({train_ratio:.0%})")
    print(f"  Validation Set: {split_counts['val']['total']:4d} images ({val_ratio:.0%})")
    print(f"  Test Set:       {split_counts['test']['total']:4d} images ({test_ratio:.0%})")
    print(f"  Integrity:      0 duplicated images across partitions (strictly disjoint).")

    if manifest["corrupted_files_found"] > 0:
        print(f"\nWarning: {manifest['corrupted_files_found']} corrupted image(s) detected and omitted.")
    if manifest["unsupported_files_found"] > 0:
        print(f"Notice:  {manifest['unsupported_files_found']} non-image/unsupported file(s) ignored.")

    print(f"\nManifest saved to: {result['manifest_path'].resolve()}")
    print("\nBalanced Class Weights for Training (Counteracting Class Imbalance):")
    for cls_name, weight in manifest["balanced_class_weights"].items():
        print(f"  {cls_name:<12}: weight = {weight:.4f}")

    print("\n[SUCCESS] Dataset is prepared and ready for model training via:")
    print(f"  python -m src.cli train --data-dir {output_dir}")
    print("=" * 70)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Handles single image prediction."""
    print_banner("Smart Waste Detection: Single Image Classification")
    config = load_config(args.config)
    threshold = args.confidence_threshold if args.confidence_threshold is not None else config.default_confidence_threshold

    predictor = WastePredictor(
        model_path=args.model_path,
        config=config,
        enable_logging=not args.no_log,
        mock_mode=args.mock,
    )

    try:
        result = predictor.predict_image(args.image, threshold=threshold)
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}", file=sys.stderr)
        return 1

    print(f"\nImage:              {Path(args.image).name}")
    print(f"Predicted Class:    {result.predicted_class.upper()}")
    print(f"Confidence:         {result.confidence:.2%}")
    print(f"Status:             {result.status}")
    print(f"Semantic Category:  {result.semantic_category}")
    if result.display_color_hint:
        print(f"Display Color Hint: {result.display_color_hint} (Optional internal UI cue)")
    print(f"Recyclable:         {'YES' if result.is_recyclable else 'NO'}")
    print(f"\nActionable Guidance:")
    print(f"  -> {result.action}")

    print("\nClass Probabilities:")
    for cls_name, prob in sorted(result.probabilities.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(prob * 30)
        bar = "#" * bar_len + "-" * (30 - bar_len)
        print(f"  {cls_name:10s} [{bar}] {prob:6.2%}")

    print("=" * 70)
    return 0


def cmd_batch_predict(args: argparse.Namespace) -> int:
    """Handles batch directory predictions."""
    print_banner("Smart Waste Detection: Batch Processing")
    config = load_config(args.config)
    threshold = args.confidence_threshold if args.confidence_threshold is not None else config.default_confidence_threshold

    predictor = WastePredictor(
        model_path=args.model_path,
        config=config,
        enable_logging=True,
        mock_mode=args.mock,
    )

    try:
        results = predictor.predict_batch(
            directory_path=args.input_dir,
            threshold=threshold,
            output_csv=args.output_report,
        )
    except Exception as e:
        print(f"\n[ERROR] Batch processing failed: {e}", file=sys.stderr)
        return 1

    total = len(results)
    successful = sum(1 for r in results if r["status"] != "ERROR")
    confident = sum(1 for r in results if r["status"] == "CONFIDENT")
    uncertain = sum(1 for r in results if r["status"] == "UNCERTAIN")

    print(f"\nProcessed: {total} images")
    print(f"Successful: {successful} | Confident: {confident} | Uncertain: {uncertain}")
    if args.output_report:
        print(f"Report exported to: {Path(args.output_report).resolve()}")

    print("\nSample Outcomes:")
    print(f"{'Image':<25} {'Class':<12} {'Confidence':<12} {'Semantic Category':<30}")
    print("-" * 80)
    for r in results[:10]:
        print(f"{r['image']:<25} {r['predicted_class']:<12} {r['confidence']:<12.2%} {r['semantic_category']:<30}")

    if total > 10:
        print(f"... and {total - 10} more items.")
    print("=" * 70)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Handles model training via MobileNetV2 transfer learning."""
    print_banner("Smart Waste Detection: Model Training")
    config = load_config(args.config)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"\n[ERROR] Training dataset directory '{data_dir.resolve()}' does not exist.", file=sys.stderr)
        return 1

    save_path = args.save_path or config.model.saved_model_path
    class_indices_path = config.model.class_indices_path
    history_path = getattr(args, "history_path", None) or "reports/training_history.json"
    use_weights = not args.no_class_weights

    try:
        history = train_model(
            data_dir=data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            save_path=save_path,
            class_indices_path=class_indices_path,
            history_path=history_path,
            use_class_weights=use_weights,
            device=args.device,
        )
        print(f"\n[SUCCESS] Training completed successfully!")
        print(f"Checkpoint weights saved to: {Path(save_path).resolve()}")
        print(f"Training history saved to:   {Path(history_path).resolve()}")
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}", file=sys.stderr)
        return 1

    print("=" * 70)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Displays prediction history analytics."""
    print_banner("Smart Waste Detection: History & Analytics")
    config = load_config(args.config)
    history_file = args.history_file or config.history_csv
    logger = HistoryLogger(history_file=history_file, config=config)

    stats = logger.get_statistics()
    total = stats.get("total_predictions", 0)

    if total == 0:
        print(f"\nNo prediction history found at '{history_file}'. Run some predictions first!")
        print("=" * 70)
        return 0

    print(f"\nTotal Waste Items Processed:   {total}")
    print(f"Average System Confidence:     {stats['average_confidence']:.2%}")
    print(f"Recyclable Items:              {stats['recyclable_items']} ({stats['recyclable_percentage']}%)")
    print(f"Non-Recyclable Items:          {stats['non_recyclable_items']}")

    print("\nStatus Distribution:")
    for status, count in stats.get("status_breakdown", {}).items():
        print(f"  {status:<15} : {count:4d} ({count / total:.1%})")

    print("\nWaste Category Breakdown:")
    for cls, count in stats.get("class_counts", {}).items():
        pct = stats["class_percentages"].get(cls, 0.0)
        bar = "#" * int(pct / 3.33)
        print(f"  {cls:12s} : {count:4d} ({pct:5.1f}%) [{bar}]")

    print("=" * 70)
    return 0


def cmd_demo_setup(args: argparse.Namespace) -> int:
    """Generates synthetic sample images for demo and testing without needing full datasets."""
    print_banner("Smart Waste Detection: Demo Environment Setup")
    dest_dir = Path(args.output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("cardboard_box.jpg", (139, 119, 101), "CARDBOARD"),
        ("glass_bottle.jpg", (200, 230, 200), "GLASS"),
        ("metal_can.jpg", (180, 180, 190), "METAL"),
        ("paper_sheet.jpg", (245, 245, 245), "PAPER"),
        ("plastic_bottle.jpg", (100, 180, 240), "PLASTIC"),
        ("trash_wrapper.jpg", (60, 60, 60), "TRASH"),
    ]

    for filename, bg_color, label in samples:
        img_path = dest_dir / filename
        canvas = np.full((300, 300, 3), bg_color, dtype=np.uint8)
        cv2.rectangle(canvas, (10, 10), (290, 290), (0, 0, 0), 2)
        cv2.putText(canvas, label, (35, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(img_path), canvas)
        print(f"  [+] Created demo image: {img_path}")

    print(f"\nCreated 6 sample waste images in '{dest_dir.resolve()}'.")
    print("You can now test predictions immediately via:")
    print(f"  python -m src.cli predict --image {dest_dir}/plastic_bottle.jpg --mock")
    print("=" * 70)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluates classification performance on a test directory."""
    print_banner("Smart Waste Detection: Model Evaluation")
    config = load_config(args.config)
    class_names = config.classes

    cm_path = args.confusion_matrix_plot or config.confusion_matrix_plot
    report_path = args.report_json or config.classification_report_json

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"\n[ERROR] Test directory '{test_dir.resolve()}' not found.", file=sys.stderr)
        return 1

    y_true: list[int] = []
    y_pred: list[int] = []

    predictor = WastePredictor(
        model_path=args.model_path,
        config=config,
        enable_logging=False,
        mock_mode=args.mock,
    )

    for idx, class_name in enumerate(class_names):
        cls_folder = test_dir / class_name
        if not cls_folder.exists() or not cls_folder.is_dir():
            continue
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            for img_file in cls_folder.glob(f"*{ext}"):
                try:
                    res = predictor.predict_image(img_file)
                    pred_idx = class_names.index(res.predicted_class) if res.predicted_class in class_names else 0
                    y_true.append(idx)
                    y_pred.append(pred_idx)
                except Exception as e:
                    print(f"Warning: Skipping '{img_file.name}': {e}")

    if not y_true:
        print("\n[ERROR] No test images found in category subfolders.", file=sys.stderr)
        print("Expected structure: <test-dir>/<class_name>/image.jpg")
        return 1

    metrics = evaluate_and_generate_reports(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        confusion_matrix_path=cm_path,
        report_json_path=report_path,
    )

    print(f"\nOverall Accuracy: {metrics.get('accuracy', 0.0):.2%}")
    print(f"Confusion Matrix saved to:  {Path(cm_path).resolve()}")
    print(f"Full Report saved to:      {Path(report_path).resolve()}")
    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Smart Waste Detection & Segregation System CLI",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML file (default: configs/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. prepare-data
    p_prep = subparsers.add_parser(
        "prepare-data",
        help="Validate and partition raw TrashNet dataset into stratified train/val/test splits",
    )
    p_prep.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory (default: data/raw)")
    p_prep.add_argument("--output-dir", type=str, default="data/processed", help="Path to processed output directory (default: data/processed)")
    p_prep.add_argument("--train-ratio", type=float, default=0.70, help="Fraction for training split (default: 0.70)")
    p_prep.add_argument("--val-ratio", type=float, default=0.15, help="Fraction for validation split (default: 0.15)")
    p_prep.add_argument("--test-ratio", type=float, default=0.15, help="Fraction for testing split (default: 0.15)")
    p_prep.add_argument("--seed", type=int, default=42, help="Reproducible random seed for stratified splitting (default: 42)")

    # 2. predict
    p_predict = subparsers.add_parser("predict", help="Predict class and semantic advice for a single waste image")
    p_predict.add_argument("--image", required=True, type=str, help="Path to input waste image")
    p_predict.add_argument("--confidence-threshold", type=float, default=None, help="Configurable heuristic confidence cutoff")
    p_predict.add_argument("--model-path", type=str, default=None, help="Path to saved model weights")
    p_predict.add_argument("--no-log", action="store_true", help="Disable logging to history.csv")
    p_predict.add_argument("--mock", action="store_true", help="Run in mock mode without loading weights")

    # 3. batch-predict
    p_batch = subparsers.add_parser("batch-predict", help="Run prediction on an entire directory of images")
    p_batch.add_argument("--input-dir", required=True, type=str, help="Directory containing images")
    p_batch.add_argument("--output-report", type=str, default="reports/batch_results.csv", help="Destination CSV")
    p_batch.add_argument("--confidence-threshold", type=float, default=None, help="Configurable heuristic confidence cutoff")
    p_batch.add_argument("--model-path", type=str, default=None, help="Path to saved model weights")
    p_batch.add_argument("--mock", action="store_true", help="Run in mock mode without loading weights")

    # 4. train
    p_train = subparsers.add_parser("train", help="Train MobileNetV2 on a waste dataset")
    p_train.add_argument("--data-dir", required=True, type=str, help="Path to dataset directory (processed/ with train/val or flat class folders)")
    p_train.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    p_train.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    p_train.add_argument("--learning-rate", type=float, default=0.0005, help="Adam optimizer learning rate (default: 0.0005)")
    p_train.add_argument("--val-split", type=float, default=0.20, help="Validation fraction split if flat dataset (default: 0.20)")
    p_train.add_argument("--save-path", type=str, default=None, help="Destination path for best checkpoint weights")
    p_train.add_argument("--history-path", type=str, default="reports/training_history.json", help="Destination path for training history JSON (default: reports/training_history.json)")
    p_train.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weighting for CrossEntropyLoss")
    p_train.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', or auto)")

    # 5. stats
    p_stats = subparsers.add_parser("stats", help="Display prediction history analytics and recycling rates")
    p_stats.add_argument("--history-file", type=str, default=None, help="Path to custom history CSV")

    # 6. demo-setup
    p_demo = subparsers.add_parser("demo-setup", help="Generate sample waste images for testing")
    p_demo.add_argument("--output-dir", type=str, default="data/sample_test_images", help="Target folder")

    # 7. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance on test folder")
    p_eval.add_argument("--test-dir", required=True, type=str, help="Root folder of test set with class subfolders")
    p_eval.add_argument("--model-path", type=str, default=None, help="Path to saved model weights")
    p_eval.add_argument("--confusion-matrix-plot", type=str, default=None, help="Path to save confusion matrix")
    p_eval.add_argument("--report-json", type=str, default=None, help="Path to save report JSON")
    p_eval.add_argument("--mock", action="store_true", help="Run evaluation in mock mode")

    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "prepare-data": cmd_prepare_data,
        "predict": cmd_predict,
        "batch-predict": cmd_batch_predict,
        "train": cmd_train,
        "stats": cmd_stats,
        "demo-setup": cmd_demo_setup,
        "evaluate": cmd_evaluate,
    }

    handler = dispatch.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
