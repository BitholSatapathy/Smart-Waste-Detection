"""Deep learning model module using MobileNetV2 for waste classification.

Implements transfer learning with a pre-trained MobileNetV2 feature extractor,
custom classification head, clean PyTorch training and validation pipelines,
checkpoint management, class-imbalance weighting, and evaluation reporting.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless CLI / server execution
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

from src.config import ModelConfig, load_config
from src.preprocessor import ImagePreprocessor


if TORCH_AVAILABLE:
    class WasteMobileNetV2(nn.Module):
        """MobileNetV2 transfer learning model adapted for 6-class waste classification."""

        def __init__(
            self,
            num_classes: int = 6,
            dropout_rate: float = 0.3,
            pretrained: bool = True,
        ):
            super().__init__()
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = mobilenet_v2(weights=weights)

            # Freeze backbone feature extraction layers for transfer learning
            for param in self.backbone.features.parameters():
                param.requires_grad = False

            # Replace original 1000-class ImageNet head with custom classification head
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_classes),
            )
            self.num_classes = num_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.backbone(x)

        def unfreeze_top_layers(self, num_blocks: int = 2) -> None:
            """Selectively unfreezes the top inverted residual blocks for fine-tuning."""
            features = list(self.backbone.features.children())
            for block in features[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        def predict_probabilities(self, tensor_or_array: Any) -> np.ndarray:
            """Generates normalized softmax probability distribution for input tensor.

            Args:
                tensor_or_array: Shape (1, 3, 224, 224) as PyTorch Tensor or NumPy array.

            Returns:
                NumPy 1D array of shape (num_classes,) containing probabilities summing to 1.0.
            """
            self.eval()
            if isinstance(tensor_or_array, np.ndarray):
                tensor = torch.from_numpy(tensor_or_array).float()
            else:
                tensor = tensor_or_array.float()

            device = next(self.parameters()).device
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = self.forward(tensor)
                probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy().squeeze(0)

        def save_weights(self, path: str | Path) -> None:
            """Saves model state dictionary to disk."""
            target_path = Path(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), target_path)

        def load_weights(self, path: str | Path, map_location: str = "cpu") -> None:
            """Loads model state dictionary from disk."""
            target_path = Path(path)
            if not target_path.exists():
                raise FileNotFoundError(f"Model weights file not found: '{target_path.resolve()}'")
            state_dict = torch.load(target_path, map_location=map_location, weights_only=True)
            self.load_state_dict(state_dict)

else:
    class WasteMobileNetV2:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            raise ImportError(
                "PyTorch is required for WasteMobileNetV2. Run: pip install torch torchvision"
            )


def create_model(
    num_classes: int = 6,
    dropout_rate: float = 0.3,
    pretrained: bool = True,
) -> WasteMobileNetV2:
    """Factory helper to instantiate the WasteMobileNetV2 architecture."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed. Cannot create MobileNetV2 model.")
    return WasteMobileNetV2(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        pretrained=pretrained,
    )


def get_training_transforms() -> tuple[Any, Any]:
    """Returns PyTorch transforms standardized with ImageNet distribution parameters.

    Ensures training and validation/test transforms match the inference pipeline in ImagePreprocessor:
    - RGB color space
    - Resized to 224x224
    - Scaled to [0.0, 1.0]
    - Normalized with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

    Training includes data augmentation (flips and small rotations).
    Validation/Test strictly avoids random augmentations.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not installed.")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


def train_model(
    data_dir: str | Path,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.0005,
    val_split: float = 0.20,
    save_path: str | Path | None = None,
    class_indices_path: str | Path | None = None,
    history_path: str | Path | None = "reports/training_history.json",
    use_class_weights: bool = True,
    device: str | None = None,
    progress_callback: Optional[Callable[[int, int, float, float, float, float], None]] = None,
) -> dict[str, list[float]]:
    """Executes PyTorch transfer learning training pipeline using MobileNetV2.

    Supports either:
    1. A partitioned directory structure: data_dir/train/<classes> and data_dir/val/<classes>.
    2. A flat directory of class subfolders: data_dir/<classes>, which is split using val_split.

    Handles class imbalance via class-weighted CrossEntropyLoss:
        w_c = total_samples / (num_classes * count_c)
    This penalizes misclassifications on underrepresented classes (such as 'trash')
    proportionally higher to prevent model bias toward majority classes.

    Args:
        data_dir: Directory containing either train/val subdirectories or class folders.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam optimizer learning rate.
        val_split: Validation fraction if flat directory is supplied (default 0.20).
        save_path: Filepath where best model checkpoint is saved.
        class_indices_path: Destination path for class-to-index JSON mapping.
        history_path: Filepath where full training history JSON is saved.
        use_class_weights: If True, applies inverse-frequency class weights to loss function.
        device: 'cuda', 'cpu', or None (auto-detect).
        progress_callback: Optional hook for epoch progress reporting.

    Returns:
        Dictionary recording training and validation loss and accuracy history.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for model training.")

    root_dir = Path(data_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Training dataset directory not found: '{root_dir.resolve()}'")

    # Select compute device
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    dev = torch.device(device_str)

    train_transform, val_transform = get_training_transforms()

    # Determine whether root_dir is partitioned into train/val or flat
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"

    if train_dir.exists() and train_dir.is_dir() and val_dir.exists() and val_dir.is_dir():
        train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
        val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)
        class_names = train_dataset.classes
    else:
        full_dataset = datasets.ImageFolder(root=str(root_dir), transform=train_transform)
        class_names = full_dataset.classes
        total_samples = len(full_dataset)
        val_size = int(total_samples * val_split)
        train_size = total_samples - val_size
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)  # type: ignore

    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError(f"No class folders found in dataset directory: '{root_dir.resolve()}'")

    # Save class-to-index mapping
    if class_indices_path:
        indices_path = Path(class_indices_path)
        indices_path.parent.mkdir(parents=True, exist_ok=True)
        with open(indices_path, "w", encoding="utf-8") as f:
            json.dump({cls: idx for idx, cls in enumerate(class_names)}, f, indent=2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Calculate class weights for loss function to address TrashNet class imbalance
    if use_class_weights:
        # Extract target labels from training dataset
        if hasattr(train_dataset, "targets"):
            train_targets = train_dataset.targets
        elif hasattr(train_dataset, "dataset") and hasattr(train_dataset.dataset, "targets"):
            train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]  # type: ignore
        else:
            train_targets = [y for _, y in train_dataset]

        counts = np.bincount(train_targets, minlength=num_classes)
        total_samples = len(train_targets)
        weights = total_samples / (num_classes * np.maximum(counts, 1))
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(dev)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print(f"Applying class-weighted CrossEntropyLoss to address data imbalance across {num_classes} classes.")
    else:
        criterion = nn.CrossEntropyLoss()

    # Model setup (transfer learning: frozen feature backbone, trainable head)
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(dev)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    checkpoint_file = Path(save_path or "models/saved_models/mobilenetv2_waste.pth")

    print(f"Starting MobileNetV2 training on {dev.type.upper()}...")
    print(f"Total training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Epoch iteration
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(dev), targets.to(dev)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == targets.data).item()
            total_train += inputs.size(0)

        epoch_train_loss = running_loss / max(1, total_train)
        epoch_train_acc = correct_train / max(1, total_train)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(dev), targets.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == targets.data).item()
                total_val += inputs.size(0)

        epoch_val_loss = val_loss / max(1, total_val)
        epoch_val_acc = correct_val / max(1, total_val)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d}\n"
            f"  Train Loss:          {epoch_train_loss:.4f}\n"
            f"  Train Accuracy:      {epoch_train_acc:.2%}\n"
            f"  Validation Loss:     {epoch_val_loss:.4f}\n"
            f"  Validation Accuracy: {epoch_val_acc:.2%}"
        )

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc)

        # Save checkpoint if validation accuracy improves
        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            model.save_weights(checkpoint_file)
            print(f"  --> Saved new best checkpoint to {checkpoint_file} (Val Acc: {best_val_acc:.2%})\n")

    # Save complete training history to reports/training_history.json
    if history_path:
        hist_file = Path(history_path)
        hist_file.parent.mkdir(parents=True, exist_ok=True)
        epoch_records = []
        for i in range(len(history["train_loss"])):
            epoch_records.append({
                "epoch": i + 1,
                "train_loss": round(float(history["train_loss"][i]), 5),
                "train_accuracy": round(float(history["train_acc"][i]), 5),
                "validation_loss": round(float(history["val_loss"][i]), 5),
                "validation_accuracy": round(float(history["val_acc"][i]), 5),
            })
        best_epoch = int(np.argmax(history["val_acc"])) + 1 if history["val_acc"] else 0
        payload = {
            "model_architecture": "MobileNetV2",
            "epochs_completed": len(history["train_loss"]),
            "best_validation_accuracy": round(float(best_val_acc), 5),
            "best_validation_epoch": best_epoch,
            "final_train_loss": round(float(history["train_loss"][-1]), 5) if history["train_loss"] else None,
            "final_train_accuracy": round(float(history["train_acc"][-1]), 5) if history["train_acc"] else None,
            "final_val_loss": round(float(history["val_loss"][-1]), 5) if history["val_loss"] else None,
            "final_val_accuracy": round(float(history["val_acc"][-1]), 5) if history["val_acc"] else None,
            "epoch_history": epoch_records,
            "history": history,
        }
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[INFO] Complete training history saved to: {hist_file.resolve()}")

    return history


def evaluate_and_generate_reports(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
    confusion_matrix_path: str | Path | None = None,
    report_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Computes confusion matrix, precision, recall, and F1-score reports.

    Args:
        y_true: Ground truth class integer indices.
        y_pred: Predicted class integer indices.
        class_names: List of string class names in index order.
        confusion_matrix_path: Path where confusion matrix plot will be saved.
        report_json_path: Path where JSON classification report will be saved.

    Returns:
        Dictionary of classification metrics.
    """
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Save JSON report if requested
    if report_json_path:
        json_path = Path(report_json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

    # Generate and save Confusion Matrix plot
    if confusion_matrix_path:
        cm_path = Path(confusion_matrix_path)
        cm_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

        fig.colorbar(cax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="left")
        ax.set_yticklabels(class_names)

        # Annotate cells with counts
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11, fontweight="bold",
                )

        plt.xlabel("Predicted Class", fontweight="bold")
        plt.ylabel("True Class", fontweight="bold")
        plt.title("Confusion Matrix - Smart Waste Segregation", pad=20, fontweight="bold")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=200)
        plt.close(fig)

    return report_dict
