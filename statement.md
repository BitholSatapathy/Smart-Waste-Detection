# Project Problem Statement & Scope Specification

**Course**: VITyarthi Flipped Course Evaluation  
**Subject**: Computer Vision  
**Project Title**: Smart Waste Detection & Segregation System  
**Academic Submission Deadline**: September 18, 2026  

---

## 1. Problem Statement

Municipal and industrial solid waste management is one of the most critical environmental challenges of modern urbanization. Rapid consumerism generates millions of tons of municipal solid waste (MSW) daily. Currently, waste sorting relies heavily on:
1. **Manual sorting at dump yards**: Laborers sort through mixed garbage under hazardous conditions, exposing themselves to toxic fumes, sharp glass, biohazards, and pathogens.
2. **Citizen self-segregation**: Highly inconsistent; lack of awareness causes recyclable materials (such as clean paper, PET bottles, and cardboard) to be contaminated by wet food waste and toxic trash, rendering them unrecyclable and diverting them to landfills.

Automating the waste segregation process at source (e.g., smart recycling kiosks in universities, residential communities, and public transit hubs) using Computer Vision and Deep Learning provides a non-invasive, hygienic, high-throughput solution.

---

## 2. Project Objectives

1. **Multi-Class Solid Waste Image Classification**: Design and deploy a Computer Vision model capable of categorizing waste items from standard RGB images into six foundational benchmark categories:
   - `cardboard`
   - `glass`
   - `metal`
   - `paper`
   - `plastic`
   - `trash`
2. **Transfer Learning Optimization**: Utilize MobileNetV2 with pre-trained ImageNet weights to achieve high classification accuracy while maintaining lightweight computational footprint suitable for resource-constrained edge devices (CPU / embedded microcomputers).
3. **Automated Segregation Recommendation Engine**: Formulate and enforce a rule-based decision engine that translates model classifications and confidence scores into primary semantic segregation categories (`Recyclable` vs. `General / Non-recyclable`), providing actionable handling guidelines (e.g., empty liquids, flatten cardboard). Optional display color hints may be provided as internal UI cues, acknowledging that physical bin coloring rules vary by municipality.
4. **Heuristic Confidence Thresholding & Quality Safeguards**: Incorporate uncertainty gating using an initial configurable heuristic threshold (default 60%). If classification confidence falls below this heuristic, the system triggers an `UNCERTAIN` warning and flags the item for manual visual verification, preventing recycling stream contamination.
5. **Auditing & Analytics**: Track all prediction events, calculating historical class distribution and recycling rates to support sustainability monitoring.
6. **Academic Rigor & Production-Grade Engineering**: Implement modular Python architecture, comprehensive automated test suites using `pytest`, and rigorous documentation aligned with VITyarthi project rubrics.

---

## 3. Functional Requirements (FRs)

- **FR-1 (Image Ingestion & Format Validation)**: The system shall accept standard RGB image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) and validate file presence, non-zero byte size, and minimal spatial resolution ($\ge 32 \times 32$).
- **FR-2 (Preprocessing & ImageNet Normalization)**: The system shall decode images using OpenCV, convert BGR color channels to RGB, resize inputs to $224 \times 224$ pixels, scale pixel values to $[0.0, 1.0]$, and apply standard ImageNet distribution normalization ($\text{mean}=[0.485, 0.456, 0.406]$, $\text{std}=[0.229, 0.224, 0.225]$).
- **FR-3 (Deep Learning Classification)**: The system shall compute a 6-class softmax probability distribution using the MobileNetV2 transfer learning architecture.
- **FR-4 (Semantic Segregation Directives Mapping)**: The decision engine shall map predicted labels to semantic categories (`Recyclable`, `General / Non-recyclable`) with optional UI display color hints and item-specific disposal guidance (e.g., "Empty liquids and rinse before disposal").
- **FR-5 (Configurable Confidence Analysis & Gating)**: The system shall compare top prediction probability against an initial configurable heuristic threshold (default $0.60$), flagging items below the threshold as `UNCERTAIN` and recommending manual inspection.
- **FR-6 (Batch Inference)**: The system shall support directory-level batch prediction, generating structured summary CSV reports.
- **FR-7 (Audit Trail Logging)**: The system shall record timestamps, filenames, predicted categories, confidence levels, and semantic classifications into `reports/history.csv`.
- **FR-8 (Analytics Reporting)**: The system shall compute and display cumulative metrics including total items processed, recycling percentage, and category frequency breakdowns.
- **FR-9 (Dataset Ingestion, Validation & Stratified Splitting)**: The system shall validate raw TrashNet dataset folders, verify image decodability, filter corruptions, compute balanced class weights to address class imbalance, and execute a deterministic stratified train/val/test split (70/15/15) with zero data leakage across partitions.

---

## 4. Non-Functional Requirements (NFRs)

- **NFR-1 (Performance & Latency)**: The system should provide practical single-image inference performance on a standard consumer computer, with latency measured and reported during evaluation.
- **NFR-2 (Reliability & Fault Tolerance)**: The system shall gracefully handle corrupted images, missing files, or non-image inputs with explicit, user-friendly error diagnostics without crashing or terminating batch execution.
- **NFR-3 (Modularity & Maintainability)**: The codebase shall strictly follow separation of concerns across 8 distinct modules (`config`, `dataset`, `preprocessor`, `model`, `predictor`, `decision_engine`, `history_logger`, `cli`), following PEP 8 style standards and explicit type hinting.
- **NFR-4 (Testability & Code Verification)**: Core validation, preprocessing, decision engine, model architecture, dataset management, and CLI pathways shall achieve high test coverage with automated `pytest` unit and integration tests (37 passing tests).
- **NFR-5 (Portability)**: The system shall operate cross-platform on Windows, macOS, and Linux using a standard Python environment and specified `requirements.txt`.

---

## 5. Dataset Specification & Benchmark (TrashNet)

The system is calibrated on the standard **TrashNet** dataset:
- **Creators**: Gary Thung and Mindy Yang (Stanford University CS229).
- **Benchmark Classes**:
  1. `cardboard` (403 images)
  2. `glass` (501 images)
  3. `metal` (410 images)
  4. `paper` (594 images)
  5. `plastic` (482 images)
  6. `trash` (137 images)
- **Total Valid Samples**: 2,527 RGB images.
- **Source Locations**:
  - Official GitHub: [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet)
  - Kaggle: [https://www.kaggle.com/datasets/feyzazk/trashnet](https://www.kaggle.com/datasets/feyzazk/trashnet)
- **Class Imbalance Strategy**: The dataset exhibits noticeable class imbalance (`paper` has 594 images whereas `trash` has only 137). To prevent majority-class bias, inverse-frequency class weights $w_c = \frac{N_{\text{total}}}{K \cdot N_c}$ are computed and integrated into the loss function during model training.
- **Data Ingestion Protocol**: Raw dataset is placed in `data/raw/<class_name>/` and processed into stratified partitions `data/processed/{train,val,test}` using `python -m src.cli prepare-data`.

---

## 6. Model Selection & Strategy Rationale

### Why MobileNetV2?
- **Inverted Residual Blocks & Linear Bottlenecks**: Drastically minimizes parameter memory footprint to **~3.5 Million parameters** compared to ResNet-50 (~25.6M) and VGG-16 (~138M).
- **Edge Device Suitability**: Real-world smart recycling bins operate on low-power hardware (such as Raspberry Pi 4 or NVIDIA Jetson Nano). MobileNetV2 is purpose-built for low-latency CPU/edge inference.
- **Transfer Learning Synergy**: Pre-training on 1.4 million ImageNet images equips the convolutional filters with fundamental low-level features (textures, reflective surfaces, geometries) essential for differentiating materials like transparent glass, crumpled paper, and glossy plastic.

---

## 6. Academic Mapping to Computer Vision Curriculum

| Curriculum Concept | Implementation in Project |
|---|---|
| **Image Acquisition & I/O** | `src/preprocessor.py` (OpenCV file reading, decoding, byte-level checks) |
| **Color Space Transformations** | BGR to RGB color conversion via `cv2.cvtColor` |
| **Spatial Scaling & Interpolation** | Bilinear interpolation image resizing to $224 \times 224$ via `cv2.resize` |
| **Statistical Normalization** | Zero-mean unit-variance scaling via standard ImageNet distribution constants |
| **Feature Extraction** | Pre-trained deep convolutional neural networks (MobileNetV2 feature extractor) |
| **Transfer Learning** | Freezing convolutional backbone; fine-tuning custom classification head |
| **Softmax Classification** | Multi-class probability distribution across 6 waste categories |
| **Model Evaluation** | Confusion Matrix, Precision, Recall, and F1-Score analysis via `scikit-learn` (executed upon model training) |
