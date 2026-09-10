# Academic Project Report: Smart Waste Detection & Segregation System

---

## 1. Cover Page

**Course**: Computer Vision & Deep Learning (Flipped Course Evaluation)  
**Project Title**: Smart Waste Detection & Segregation System  
**Student Name**: [Student Name Placeholder]  
**Registration Number**: [Registration Number Placeholder]  
**Institution**: Vellore Institute of Technology (VIT)  
**Department**: School of Computer Science and Engineering  
**Academic Year**: 2025–2026  
**Submission Date**: September 11, 2026  
**Evaluation Target Date**: September 18, 2026  

---

## 2. Introduction

Municipal Solid Waste (MSW) management is a pressing environmental, logistical, and public health challenge in modern urban ecosystems. The global acceleration of consumer goods manufacturing and single-use packaging generates billions of tons of solid refuse annually. Although recycling infrastructure exists in many regions, recycling efficiency remains heavily undermined by contamination. When non-recyclable refuse (e.g., soiled composite packaging, greasy food containers, non-recyclable plastics) is commingled with clean recyclables (such as aluminum, clean cardboard, and PET bottles), entire batches are rejected at material recovery facilities (MRFs) and diverted to landfills or incinerators.

Historically, waste sorting has relied on two primary paradigms:
1. **Source-Level Manual Segregation**: Dependent on individual citizen awareness and diligence, which frequently fails due to ambiguous packaging labeling, differing regional municipal guidelines, or public apathy.
2. **Post-Collection Manual Sorting**: Involves human sanitation workers manually sorting trash along conveyor belts. This practice poses severe occupational hazards, exposing workers to chemical pathogens, biohazards, and sharp objects.

Computer Vision and Deep Learning offer an automated, hygienic, non-contact, and objective alternative. By integrating compact optical cameras at the intake aperture of smart waste receptacles or conveyor sorting chutes, items can be visually classified in real time before physical sorting. This project develops the **Smart Waste Detection & Segregation System**, an end-to-end computer vision pipeline that classifies solid waste into 6 benchmark classes using transfer learning with a pre-trained **MobileNetV2** deep convolutional neural network. The predictions are processed by an integrated domain-aware **Decision Engine** that maps items into high-level semantic disposal categories (`Recyclable` vs. `General / Non-recyclable`), enforces an initial configurable heuristic confidence threshold to isolate uncertain items, and maintains operational audit logs for recycling analytics.

---

## 3. Problem Statement

Automated municipal solid waste segregation systems deployed in institutional and public settings require high classification accuracy across diverse material classes, rapid inference on low-power edge processors, and reliable uncertainty handling to prevent recyclable batch contamination. Most existing deep learning prototypes either rely on computationally heavy architectures unsuitable for consumer CPUs or fail to translate raw neural network probabilities into actionable, policy-agnostic disposal instructions.

**Project Objective**:  
To design, implement, evaluate, and document an edge-compatible, robust Computer Vision and Deep Learning system capable of:
1. Classifying waste imagery into six standard benchmark categories: `cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`.
2. Achieving low-latency inference on standard consumer-grade CPUs without requiring discrete hardware accelerators.
3. Translating predictions into semantic segregation directives (`Recyclable` vs. `General / Non-recyclable`) with actionable user instructions.
4. Implementing an uncertainty mitigation mechanism that flags ambiguous or low-confidence predictions ($\tau < 0.60$) for manual verification.
5. Providing an automated audit trail for continuous recycling analytics and compliance verification.

---

## 4. Functional Requirements

The system implements 5 major functional modules with 7 command-line operations:

1. **FR-1: Dataset Ingestion & Validation** (`src/dataset.py`)
   - Programmatically verifies the presence of all 6 target class folders.
   - Validates every image file for non-zero byte size, decodability, and minimum spatial resolution ($\ge 32 \times 32$ pixels).
   - Filters out non-image files (`.txt`, `.pdf`, `.DS_Store`) and corrupted images without crashing.
   - Executes stratified 70/15/15 partitioning into `train/`, `val/`, and `test/` partitions using a deterministic seed (`seed=42`).
   - Programmatically verifies zero data leakage: $\text{train} \cap \text{val} = \emptyset, \text{train} \cap \text{test} = \emptyset, \text{val} \cap \text{test} = \emptyset$.
   - Generates an audit manifest (`dataset_manifest.json`) capturing partition distributions and balanced inverse-frequency class weights.

2. **FR-2: Image Preprocessing & Normalization** (`src/preprocessor.py`)
   - Decodes image streams via OpenCV, converting standard BGR arrays to RGB.
   - Resizes images to standard $224 \times 224$ resolution using bilinear interpolation.
   - Converts integer pixels $[0, 255]$ to 32-bit floating point $[0.0, 1.0]$.
   - Applies ImageNet channel normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

3. **FR-3: Waste Classification & Deep Learning Inference** (`src/model.py`, `src/predictor.py`)
   - Instantiates a MobileNetV2 convolutional backbone pretrained on ImageNet (`IMAGENET1K_V1`).
   - Replaces the 1000-class output layer with a custom dense classification head (1280 $\to$ 128 $\to$ 6) utilizing Dropout and ReLU activations.
   - Computes a normalized 6-class Softmax probability distribution $\mathbf{p} = [p_1, \dots, p_6]$ such that $\sum_{i=1}^6 p_i = 1.0$.

4. **FR-4: Semantic Decision Engine & Uncertainty Gating** (`src/decision_engine.py`)
   - Maps predicted material classes into primary semantic categories (`Recyclable` vs. `General / Non-recyclable`).
   - Attaches actionable disposal advice (e.g., rinsing cans, flattening cardboard, checking resin codes).
   - Applies an initial configurable heuristic confidence threshold ($\tau = 0.60$):
     - If $\max_k p_k \ge \tau$: Emits `CONFIDENT` status and assigns the mapped semantic category.
     - If $\max_k p_k < \tau$: Emits `UNCERTAIN` status and routes the item to `Manual Verification Required`.

5. **FR-5: Inference Coordination & Dual-Mode Execution** (`src/predictor.py`, `src/cli.py`)
   - **Single-Image Inference**: Evaluates an individual image path and outputs formatted terminal cards.
   - **Batch Directory Inference**: Processes directories of images, handles errors gracefully per item, and exports CSV summary reports.

6. **FR-6: Audit Trail Logging & Analytics** (`src/history_logger.py`)
   - Persists timestamp, filename, predicted class, confidence, semantic category, and processing latency to `reports/history.csv`.
   - Computes running sustainability analytics (recycling diversion percentages, class frequencies).

7. **FR-7: Automated Evaluation Suite** (`src/model.py`, `src/cli.py`)
   - Evaluates saved model checkpoints against test sets.
   - Computes overall accuracy, per-class and macro/weighted precision, recall, and F1-score.
   - Generates numerical CSV and visual PNG confusion matrices.

---

## 5. Non-Functional Requirements

1. **NFR-1: Low-Latency Inference Performance**
   - The forward pass of the model must execute in under 20 ms per image on standard consumer laptop CPUs, supporting interactive real-time kiosk operation ($>50$ FPS).
   - *Measured Realization*: Achieved **9.29 ms** mean forward-pass latency (~107.6 FPS) on a local consumer CPU.

2. **NFR-2: Fault Resilience & Input Validation**
   - The system must trap zero-byte files, non-image files, corrupted bitstreams, and truncated headers using custom `ImageValidationError` exceptions, providing clear error diagnostics without crashing.
   - *Measured Realization*: Verified across 9 unit tests in `test_preprocessor.py`.

3. **NFR-3: Modularity & Decoupled Architecture**
   - The codebase must be decomposed into 5–10 cohesive, single-responsibility modules adhering to PEP 8 standards with full type annotations.
   - *Measured Realization*: Implemented across 8 modular files in `src/`.

4. **NFR-4: Test Coverage & Verification**
   - Core algorithmic functions (data splitting, preprocessing, tensor shapes, decision logic, CLI commands) must be covered by automated tests with a 100% pass rate.
   - *Measured Realization*: 37 passing test cases across 6 `pytest` modules.

5. **NFR-5: Deterministic Reproducibility**
   - Data partitioning and randomized data augmentation must be fully reproducible across environments using a fixed random seed (`seed=42`).
   - *Measured Realization*: Validated by programmatic zero-leakage assertions and manifest audits.

---

## 6. System Architecture

The system is organized into modular pipelines for **Inference**, **Training**, and **Evaluation**:

```
[Raw Waste Image Input]
        │
        ▼
[Validation Layer: File Check, Size Check, Decodability]
        │
        ▼
[OpenCV Preprocessing: BGR->RGB, Resizing to 224x224, ImageNet Normalization]
        │
        ▼
[MobileNetV2 Deep Convolutional Feature Extractor]
        │
        ▼
[Custom Classification Head: Dropout(0.3) -> Linear(1280->128) -> ReLU -> Dropout(0.2) -> Linear(128->6)]
        │
        ▼
[Softmax Normalization: 6-Class Probability Vector]
        │
        ▼
[Semantic Decision Engine: Top Class Extraction & Heuristic Confidence Gating (tau = 0.60)]
   ├── If Confidence >= 0.60 ──> CONFIDENT Status, Mapped Semantic Category, Actionable Advice
   └── If Confidence <  0.60 ──> UNCERTAIN Status, Manual Verification Directive
        │
        ▼
[Audit Logging: Append Event to reports/history.csv]
        │
        ▼
[User Output: CLI Formatted Card / JSON / CSV Batch Report]
```

### Architectural Separation of Concerns

1. **Inference Pipeline**: Operates sequentially from image ingestion through OpenCV preprocessing, PyTorch tensor transformation, MobileNetV2 evaluation mode forward pass, Softmax conversion, decision engine thresholding, and CSV transaction logging.
2. **Training Pipeline**: Loads stratified partitions from `data/processed/train` and `data/processed/val`. Applies training augmentations (random horizontal flip, random rotation $\pm 15^\circ$), utilizes class-weighted `CrossEntropyLoss` computed by `src/dataset.py`, trains the classification head using the Adam optimizer ($lr = 0.0005$), validates after each epoch, and serializes the best checkpoint based on validation loss to `models/saved_models/mobilenetv2_waste.pth`.
3. **Evaluation Pipeline**: Loads the isolated test partition (`data/processed/test`), computes predictions for all 379 test samples, computes the full classification report via scikit-learn, generates a 6x6 confusion matrix heatmap, and measures forward-pass inference latency across 50 iterations.

---

## 7. Design Diagrams

### 7.1 Detailed Component & Data Flow Diagram

```mermaid
graph TD
    subgraph Data Preparation Pipeline
        RAW[data/raw: 6 Class Folders] --> VAL[Dataset Ingestion & Validation]
        VAL --> SPLIT[Stratified 70/15/15 Partitioning seed=42]
        SPLIT --> MAN[dataset_manifest.json & Class Weights]
        SPLIT --> PROC[data/processed: train, val, test]
    end

    subgraph Deep Learning Pipeline
        PROC --> LOADER[PyTorch DataLoader & Data Augmentation]
        LOADER --> MOB[MobileNetV2 Backbone: Frozen Pretrained Weights]
        MOB --> HEAD[Custom Classifier Head: 1280 -> 128 -> 6]
        HEAD --> LOSS[Class-Weighted CrossEntropyLoss]
        HEAD --> SOFT[Softmax Output Layer]
    end

    subgraph Inference & Decision Pipeline
        INPUT[New Waste Image] --> PREP[OpenCV Preprocessing: 224x224, ImageNet Norm]
        PREP --> HEAD
        SOFT --> DEC[Decision Engine: Heuristic Threshold tau=0.60]
        DEC -->|Confidence >= 0.60| RES_C[CONFIDENT: Recyclable / General]
        DEC -->|Confidence < 0.60| RES_U[UNCERTAIN: Manual Verification]
        RES_C --> LOG[History Logger: reports/history.csv]
        RES_U --> LOG
        LOG --> CLI[CLI Output / User Display]
    end
```

### 7.2 Decision Engine State Machine

```mermaid
stateDiagram-v2
    [*] --> ReceiveProbabilities: Softmax Vector (6 Classes)
    ReceiveProbabilities --> CheckValidity: Validate Sum ~ 1.0 & Length = 6
    CheckValidity --> ExtractDominant: Identify Class k with Max Probability P(k)
    ExtractDominant --> EvaluateThreshold: Compare P(k) vs Heuristic Threshold tau (0.60)
    
    EvaluateThreshold --> ConfidentBranch: P(k) >= tau
    EvaluateThreshold --> UncertainBranch: P(k) < tau
    
    ConfidentBranch --> AssignSemanticCategory: Map Class k to Recyclable or General
    AssignSemanticCategory --> AttachHandlingAdvice: Add Material-Specific Disposal Steps
    AttachHandlingAdvice --> EmitResult: Status = CONFIDENT
    
    UncertainBranch --> FlagManualVerification: Assign Manual Verification Required
    FlagManualVerification --> EmitResult: Status = UNCERTAIN
    
    EmitResult --> AppendAuditLog: Write to reports/history.csv
    AppendAuditLog --> [*]
```

---

## 8. Design Decisions & Rationale

| Design Decision | Alternative Considered | Selected Approach | Technical Rationale for Defense in Viva |
|---|---|---|---|
| **Deep Learning Backbone** | ResNet-50, VGG-16 | **MobileNetV2** | MobileNetV2 has ~3.5M parameters and requires ~14 MB disk space, compared to ResNet-50 (25.6M params, ~98 MB) and VGG-16 (138M params, ~528 MB). Its depthwise separable convolutions enable $<10$ ms inference on consumer CPUs, making it ideal for edge devices without discrete GPUs. |
| **Learning Strategy** | Training from scratch | **Transfer Learning (Feature Extractor)** | The TrashNet dataset contains 2,527 images. Training a deep network from scratch on this size would lead to severe overfitting. Pretrained ImageNet weights provide robust low-level filters (edges, textures, shapes), enabling rapid convergence and high generalization in 15 epochs. |
| **Input Resolution** | $512 \times 512$, $128 \times 128$ | **$224 \times 224$ pixels** | $224 \times 224$ is the canonical input dimension of MobileNetV2. It preserves sufficient spatial detail for waste texture discrimination while keeping computational cost bounded. |
| **Image Normalization** | Min-Max $[0, 1]$ scaling | **ImageNet Channel Normalization** | The pretrained MobileNetV2 backbone was optimized on ImageNet with $\mu = [0.485, 0.456, 0.406]$ and $\sigma = [0.229, 0.224, 0.225]$. Using identical normalization maintains distribution alignment with the pretrained weights. |
| **Class Imbalance Handling** | Random oversampling, SMOTE | **Class-Weighted CrossEntropyLoss** | The minority class (`trash`, 137 images) is outnumbered by `paper` (594 images) by $>4:1$. Rather than duplicating images (oversampling risks memorization), mathematically weighting the loss ($w_{\text{trash}} = 3.0742, w_{\text{paper}} = 0.7090$) heavily penalizes misclassifying rare items. |
| **Dataset Partitioning** | K-fold CV, naive random split | **Stratified 70/15/15 with Fixed Seed** | Stratified splitting maintains exact class proportions across train, validation, and test splits. Enforcing programmatic zero-leakage assertions guarantees experimental validity without data snooping. |
| **Segregation Output Schema** | Universal bin colors (Blue, Green, Yellow) | **Semantic Categories (`Recyclable` vs. `General`)** | Municipal recycling guidelines, container color conventions, and recycling policies vary substantially across cities, states, and universities. Outputting semantic categories is policy-agnostic, while display colors are kept strictly as UI hints. |
| **Uncertainty Gating** | Fixed hard-coded cutoff | **Configurable Heuristic Threshold ($\tau = 0.60$)** | Real disposal environments feature hybrid items and occluded views. Gating on confidence prevents contaminating recycling streams with false positives. A heuristic threshold is transparent, tunable via YAML, and avoids false certainty claims. |
| **Interface Design** | Heavy GUI (Electron / Qt) | **Modular CLI Application** | A CLI interface allows lightweight deployment in server, headless kiosk, and scriptable automated sorting environments. It runs natively across Windows, Linux, and macOS without UI library overhead. |

---

## 9. Implementation Details

### 9.1 Technology Stack & Environment
- **Operating System**: Windows 11 (tested on standard 64-bit consumer architecture)
- **Programming Language**: Python 3.10.11
- **Deep Learning Framework**: PyTorch 2.2.0, Torchvision 0.17.0
- **Computer Vision**: OpenCV 4.8.0, Pillow 9.5.0
- **Data & Evaluation**: NumPy 1.24.3, scikit-learn 1.3.0, Pandas 2.0.0, Matplotlib 3.7.0
- **Testing**: pytest 7.4.0

### 9.2 Modular File Organization

1. **`configs/config.yaml`**: Centralizes all configuration settings including dataset split ratios (0.70/0.15/0.15), input resolution ($224 \times 224$), normalization constants, optimizer hyperparameters (Adam, $lr=0.0005$, batch size 32), the confidence threshold ($\tau = 0.60$), and semantic class mappings.
2. **`src/config.py`**: Loads and validates `config.yaml` into strongly-typed dataclasses (`ModelConfig`, `DatasetConfig`, `DecisionConfig`, `SystemConfig`), preventing runtime configuration errors.
3. **`src/dataset.py`**: Implements `DatasetManager` to validate raw image folders, compute inverse-frequency class weights, execute stratified splits, verify zero partition overlap, and export `dataset_manifest.json`.
4. **`src/preprocessor.py`**: Encapsulates `ImagePreprocessor` with OpenCV image decoding, color conversion, resizing, ImageNet normalization, and custom `ImageValidationError` guards.
5. **`src/model.py`**: Defines the `WasteMobileNetV2` PyTorch model class, classification head architecture, training loop with early stopping, checkpointing, and evaluation metric computation.
6. **`src/predictor.py`**: Implements `WastePredictor` to coordinate preprocessor execution, model inference, and fallback mock behavior for testing environments.
7. **`src/decision_engine.py`**: Contains `DecisionEngine` to evaluate class probabilities against the confidence threshold, map semantic categories, and generate user handling guidance.
8. **`src/history_logger.py`**: Manages `HistoryLogger` to persist inference transactions into `reports/history.csv` and calculate summary sustainability statistics.
9. **`src/cli.py`**: Implements 7 subcommands (`prepare-data`, `train`, `evaluate`, `predict`, `batch-predict`, `stats`, `demo-setup`) with clean argument parsing and tabular formatting.

---

## 10. Screenshots & Experimental Results

### 10.1 Real Model Training Summary

The model was trained for 15 epochs using the prepared TrashNet training partition ($1,769$ images) with validation on the validation partition ($379$ images).

- **Total Training Duration**: 6 minutes 54 seconds (executed entirely on local CPU)
- **Initial Epoch 1 Performance**: Training Acc: 46.13%, Val Acc: 78.89%, Val Loss: 0.6120
- **Best Validation Accuracy**: **87.34%** (Achieved at **Epoch 14**, Val Loss: 0.3970)
- **Final Epoch 15 Performance**: Training Acc: 82.25%, Val Acc: 85.75%, Val Loss: 0.4431
- **Checkpointing**: The model weights from Epoch 14 were saved to `models/saved_models/mobilenetv2_waste.pth`.

#### Training Accuracy & Loss Curves

The convergence curves illustrate rapid feature adaptation during early epochs, followed by steady loss minimization:

| Training & Validation Accuracy | Training & Validation Loss |
|:---:|:---:|
| ![Training Accuracy](../reports/training_accuracy.png) | ![Training Loss](../reports/training_loss.png) |

---

### 10.2 Final Test-Set Evaluation Results

The best saved checkpoint was evaluated against the strictly held-out test partition ($N = 379$ images, 15% of the TrashNet dataset). **The test partition was not used during training or hyperparameter tuning.**

#### Aggregate Performance Metrics

| Metric | Measured Value | Description |
|---|---|---|
| **Overall Test Accuracy** | **86.81%** | 329 out of 379 test samples correctly classified |
| **Macro Average Precision** | **84.25%** | Unweighted mean precision across all 6 classes |
| **Macro Average Recall** | **85.38%** | Unweighted mean recall across all 6 classes |
| **Macro Average F1-Score** | **84.67%** | Unweighted harmonic mean of precision and recall |
| **Weighted Average Precision** | **87.26%** | Support-weighted precision reflecting class sample sizes |
| **Weighted Average Recall** | **86.81%** | Support-weighted recall matching overall accuracy |
| **Weighted Average F1-Score** | **86.95%** | Support-weighted harmonic mean across test set |

#### Per-Class Performance Breakdown

| Class Label | Test Support | Precision (%) | Recall (%) | F1-Score (%) | Semantic Disposal Category |
|---|---|---|---|---|---|
| **Cardboard** | 61 | 93.10% | 88.52% | 90.76% | Recyclable |
| **Glass** | 75 | 87.32% | 82.67% | 84.93% | Recyclable |
| **Metal** | 61 | 88.52% | 88.52% | 88.52% | Recyclable |
| **Paper** | 89 | 93.02% | 89.89% | 91.43% | Recyclable |
| **Plastic** | 73 | 81.01% | 87.67% | 84.21% | Recyclable |
| **Trash** | 20 | 62.50% | 75.00% | 68.18% | General / Non-recyclable |

---

### 10.3 Confusion Matrix & Error Analysis

The confusion matrix was plotted and exported to `reports/confusion_matrix.png` and `reports/confusion_matrix.csv`:

![Confusion Matrix](../reports/confusion_matrix.png)

#### Key Observations:
1. **Paper and Cardboard High Separation**: Both classes achieved precision $>93\%$. Confusions were limited to thin packaging cartons resembling heavy paper stock.
2. **Glass vs. Plastic Ambiguity**: 8 plastic items were misclassified as glass, and 7 glass items were misclassified as plastic. This constitutes the largest source of error in the system and is attributable to shared optical properties (transparency, specular glare, cylindrical bottle geometries).
3. **Trash Minority Class Performance**: Despite having only 20 test samples, the class-weighted loss function enabled the model to achieve 75.00% recall (15/20 correct). Misclassifications occurred primarily with crushed metal wrappers and complex composite plastic bags.

---

### 10.4 Real Inference Latency Benchmark

Inference latency was benchmarked by executing 50 consecutive forward passes with isolated PyTorch tensor inputs on the local development machine:

| Metric | Measured Value | Notes |
|---|---|---|
| **Mean Forward-Pass Latency** | **9.29 ms** | Corresponds to ~107.6 FPS throughput |
| **Median Latency** | **9.44 ms** | Highly stable execution |
| **Minimum Latency** | **7.34 ms** | Warm cache execution |
| **Maximum Latency** | **13.69 ms** | Cold-start execution |
| **Standard Deviation** | **1.06 ms** | Low latency variance |
| **Evaluation Environment** | Intel Core i5-12450H CPU | Batch size = 1, Input = $224 \times 224 \times 3$ |

> [!NOTE]
> Latency benchmarks were conducted on the local development computer and strictly isolate the neural network forward pass from disk I/O, image loading, and OpenCV decoding. These results are specific to this hardware environment and are not claimed as universal hardware benchmarks.

---

### 10.5 Representative Single-Image Predictions

To validate the end-to-end pipeline, real images from the test set were processed through the full prediction and decision engine workflow:

| Cardboard (Confident - Recyclable) | Glass (Confident - Recyclable) |
|:---:|:---:|
| ![Cardboard Prediction](../reports/sample_predictions/cardboard_predicted_cardboard.jpg) | ![Glass Prediction](../reports/sample_predictions/glass_predicted_glass.jpg) |
| **Pred**: Cardboard (96.5%) \| **Category**: Recyclable | **Pred**: Glass (91.2%) \| **Category**: Recyclable |

| Metal (Confident - Recyclable) | Paper (Confident - Recyclable) |
|:---:|:---:|
| ![Metal Prediction](../reports/sample_predictions/metal_predicted_metal.jpg) | ![Paper Prediction](../reports/sample_predictions/paper_predicted_paper.jpg) |
| **Pred**: Metal (94.8%) \| **Category**: Recyclable | **Pred**: Paper (97.1%) \| **Category**: Recyclable |

| Plastic (Confident - Recyclable) | Trash (Confident - General) |
|:---:|:---:|
| ![Plastic Prediction](../reports/sample_predictions/plastic_predicted_plastic.jpg) | ![Trash Prediction](../reports/sample_predictions/trash_predicted_trash.jpg) |
| **Pred**: Plastic (91.5%) \| **Category**: Recyclable | **Pred**: Trash (78.3%) \| **Category**: General / Non-recyclable |

---

## 11. Testing Approach

The project utilizes `pytest` to implement automated unit and integration testing across 6 test modules (37 total tests), achieving a **100% pass rate**:

```
tests/test_cli.py ............ [19%]
tests/test_dataset.py ........ [41%]
tests/test_decision_engine.py  [62%]
tests/test_history_logger.py . [65%]
tests/test_model.py .......... [76%]
tests/test_preprocessor.py ... [100%]
============================== 37 passed in 18.24s ==============================
```

### Categorization of Test Modules

1. **Preprocessing & Input Integrity Tests** (`tests/test_preprocessor.py` - 9 tests):
   - Validates path existence checking and extension filtering.
   - Tests detection and rejection of zero-byte files, non-image files, and corrupted byte headers.
   - Tests RGB color conversion, bilinear interpolation to $224 \times 224$, and exact ImageNet normalization values.
2. **Dataset Validation & Partitioning Tests** (`tests/test_dataset.py` - 8 tests):
   - Validates class directory checking and empty directory rejection.
   - Tests inverse-frequency class weight computation.
   - Tests 70/15/15 stratified partitioning and seed reproducibility.
   - Asserts zero data leakage between train, validation, and test splits.
   - Validates end-to-end dataset preparation and manifest generation.
3. **Decision Engine & Threshold Tests** (`tests/test_decision_engine.py` - 8 tests):
   - Verifies mapping of each class to correct semantic categories (`Recyclable` vs. `General / Non-recyclable`).
   - Tests heuristic confidence threshold gating ($\tau = 0.60$): asserts `CONFIDENT` for $p \ge 0.60$ and `UNCERTAIN` for $p < 0.60$.
   - Validates error handling for malformed probability distributions (empty arrays, wrong lengths, negative probabilities).
4. **Model Architecture & Tensor Tests** (`tests/test_model.py` - 4 tests):
   - Tests model instantiation and verifies input tensor acceptance: `(batch_size, 3, 224, 224)`.
   - Tests output tensor shape: `(batch_size, 6)`.
   - Tests Softmax normalization: verifies probabilities sum to $1.0 \pm 10^{-5}$.
   - Tests model weights serialization and deserialization.
5. **History Logger & Analytics Tests** (`tests/test_history_logger.py` - 1 test):
   - Verifies CSV persistence, column schema integrity, and aggregate recycling percentage computation.
6. **End-to-End CLI Integration Tests** (`tests/test_cli.py` - 7 tests):
   - Validates command execution for `--help`, `demo-setup`, `predict`, `batch-predict`, and `prepare-data`.

---

## 12. Challenges Faced

1. **Class Imbalance**: The TrashNet dataset exhibits substantial disparity between majority classes (`paper`: 594 images) and minority classes (`trash`: 137 images). Naive unweighted training resulted in lower recall on trash items. This was addressed by computing inverse-frequency loss weights ($w_c = \frac{N_{\text{total}}}{K \cdot N_c}$), yielding $w_{\text{trash}} = 3.0742$, which boosted trash test recall to 75.00%.
2. **Visual Ambiguity Between Transparent Materials**: Clear PET plastic bottles and clear glass jars share high visual similarity, specular surface highlights, and transparency against white backgrounds. This caused mutual cross-misclassification (8 plastic as glass, 7 glass as plastic). Data augmentation with random rotations helped mitigate, though not completely eliminate, this optical ambiguity.
3. **Cardboard vs. Heavy Craft Paper Confusion**: Thin corrugated cardboard boxes and thick, unprinted craft paper sheets share similar brown fibrous textures. The model occasionally misclassified thin cardboard as paper.
4. **Heterogeneous Composition of the "Trash" Category**: While classes such as `metal` and `glass` have relatively consistent surface textures, `trash` comprises diverse non-recyclables including snack bags, wrappers, and composite packaging. This intra-class diversity made learning a compact feature representation more difficult.
5. **CPU-Only Training Constraints**: Training deep neural networks without a dedicated GPU requires careful parameter tuning. By freezing the MobileNetV2 backbone and training only the dense classification head with Adam ($lr=0.0005$), complete 15-epoch training was achieved in 6 minutes 54 seconds on a standard consumer CPU.
6. **Heuristic Confidence Threshold Calibration**: Selecting a threshold that balances false positives against over-cautious manual verification requires empirical tuning. An initial heuristic threshold of $\tau = 0.60$ was chosen to provide safety against ambiguous inputs while keeping rejection rates low.

---

## 13. Learnings & Key Takeaways

1. **Effectiveness of Transfer Learning**: Leveraging features learned on ImageNet allowed the model to achieve 86.81% test accuracy with only 1,769 training images in 15 epochs, demonstrating the power of transfer learning for domain-specific vision tasks.
2. **Importance of Data Hygiene and Zero Leakage**: Ensuring strict separation of train, validation, and test splits with seed reproducibility prevents optimistic performance estimates and provides a reliable assessment of real-world generalization.
3. **Loss Weighting Over Resampling**: Using inverse-frequency class weights in the loss function avoided the memory overhead and overfitting risks associated with oversampling small minority classes.
4. **Necessity of Uncertainty Handling**: A raw neural network always outputs an argmax class, even for completely ambiguous inputs. Incorporating a configurable confidence threshold provides an essential safety layer for real-world automated sorting systems.
5. **Modular Software Architecture**: Decoupling the data loading, preprocessing, model architecture, decision logic, and CLI into separate modules facilitated comprehensive unit testing (37/37 passing) and simplified debugging.

---

## 14. Future Enhancements

1. **Multi-Object Detection & Instance Segmentation**: The current system assumes a single dominant waste item per frame. Future iterations could integrate object detection models (e.g., YOLOv8 or Mask R-CNN) to localize and classify multiple overlapping waste items in complex sorting scenes.
2. **Multi-Spectral & Near-Infrared (NIR) Sensing**: Integrating near-infrared spectroscopy would allow distinguishing chemically distinct plastics (such as PET, HDPE, PVC, and PP) that appear visually identical in visible RGB light.
3. **Hardware Actuator Integration**: Interfacing the CLI decision engine outputs with an Arduino or Raspberry Pi GPIO controller to mechanically actuate pneumatic valves or servo flaps for automated physical sorting.
4. **Full Fine-Tuning with Learning Rate Scheduling**: Gradually unfreezing the upper convolutional blocks of MobileNetV2 with a cosine annealing learning rate schedule could further improve fine-grained feature extraction.
5. **Edge Quantization (INT8 / ONNX Runtime)**: Post-training quantization of the PyTorch model to INT8 precision could further reduce inference latency and memory footprint for deployment on resource-constrained embedded microcontrollers.

---

## 15. References

1. **TrashNet Dataset**:  
   Thung, G., & Yang, M. (2016). *Classification of Trash for Recyclability Status*. Stanford University CS229 Computer Science Technical Report. [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
2. **MobileNetV2 Architecture**:  
   Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510–4520.
3. **Deep Learning Framework (PyTorch)**:  
   Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS), 32, pp. 8024–8035.
4. **Computer Vision Library (OpenCV)**:  
   Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools, 25(11), pp. 120–125.
5. **Machine Learning & Evaluation Metrics (scikit-learn)**:  
   Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830.
6. **ImageNet Benchmark**:  
   Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 248–255.
7. **Municipal Solid Waste & Recycling Guidelines**:  
   United States Environmental Protection Agency (EPA). (2020). *National Recycling Strategy: Part One of a Series on Building a Circular Economy for All*. EPA Document 530-R-20-008.

