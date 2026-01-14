# Brain Tumor Classification — Methodological Comparison (Case Study)

## Overview
This repository presents a **three-phase** case study in MRI-based brain tumor detection and classification:

1. **Binary Classification (Tumor vs. No Tumor)** using the **Br35H** dataset — compares classical ML models, a simple neural network, and a small CNN.
2. **Multiclass Classification (4 classes)** on a four-class MRI dataset *(glioma, meningioma, pituitary tumor, no tumor)* — includes augmentation, baseline CNN, transfer learning, and custom CNN variants with attention-style improvements.
3. **Hybrid Phase (Feature/Model Hybridisation)** — explores hybrid approaches that combine strengths of architectures/features (e.g., attention-enhanced CNN + transfer-learning embeddings) to improve multiclass performance and robustness.

> Notebooks are designed to be run sequentially, but each phase can be executed independently.

---

## Repository Structure
- `1_BinaryClassification.ipynb` — Phase 1 (Binary classification)
- `2_MultiClass.ipynb` — Phase 2 (Multiclass baseline + transfer learning)
- `3_hybrid.ipynb` — Phase 3 (Hybrid methods / combined models)

---

## Data Sources

### Br35H (Binary) Dataset
- ~1,500 tumorous and ~1,500 non-tumorous MRI scans (+ unlabeled images for prediction)
- Used to evaluate Logistic Regression, SVM, Random Forest, XGBoost, and neural models

### Multiclass MRI Dataset
- 2,870 images (64×64), 4 classes: glioma, meningioma, pituitary, no tumor
- Split: 75% train / 15% val / 10% test

### Reference Paper (Benchmark)
Khaliki & Başarslan, *Scientific Reports* (01 Feb 2024) — DOI: 10.1038/s41598-024-52823-9

---

## Methodology

### Phase 1 — Binary Classification Pipeline
**Goal:** Detect tumor presence (tumor vs. no tumor)

- **Preprocessing:** grayscale normalisation, flattening, feature scaling
- **Models:**
  - Logistic Regression (with CV)
  - Linear SVM
  - Random Forest
  - XGBoost
  - Simple Feed-Forward Neural Network
  - Small CNN
- **Evaluation:** classification report, confusion matrix, ROC-AUC

---

### Phase 2 — Multiclass Classification Pipeline
**Goal:** Classify MRI into 4 classes (glioma / meningioma / pituitary / no tumor)

- **Loading:** directory-based class import
- **EDA:** class distribution + sample visualization
- **Augmentation:** rotations, flips, shifts, zoom
- **Baselines:**
  - CNN from scratch
  - Transfer learning with VGG16 (ImageNet)
- **Training:** Adam, early stopping, batch size=32, LR scheduling
- **Fine-tuning:** unfreeze selected VGG16 layers with smaller LR

---

### Phase 3 — Hybrid Phase (Hybrid / Combined Approaches)
**Goal:** Improve multiclass performance using hybridisation strategies.

Examples explored in `3_hybrid.ipynb` may include:
- **SE/attention-style enhancements** (channel-wise emphasis)
- **Feature fusion** (custom CNN feature maps + transfer-learning embeddings)
- **Hybrid heads** (combining representations before classification)

> If you add more hybrid variants later, list them here so readers know what’s “new” vs. Phase 2.

---

## Requirements
- Python 3.8+
- TensorFlow 2.x *or* PyTorch 1.x
- scikit-learn, OpenCV
- NumPy, pandas, matplotlib


---

## Usage

### 1) Binary Classification
Run:
- `1_BinaryClassification.ipynb`

### 2) Multiclass Classification
Run:
- `2_MultiClass.ipynb`

### 3) Hybrid Phase
Run:
- `3_hybrid.ipynb`

---

## License
MIT
