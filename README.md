Brain Tumor Classification Methodological Comparison Case Study
=====================================

Overview
--------

This repository presents a two-stage case study in MRI-based brain tumor detection and classification:

1.  **Binary Classification** using the Br35H dataset (tumor vs. no tumor), comparing traditional machine-learning models, a simple neural network, and a small CNN.

2.  **Multiclass Classification** on a four-class MRI dataset (glioma, meningioma, pituitary tumor, no tumor), leveraging data augmentation, baseline and transfer-learning models, and a novel custom CNN with Squeeze-and-Excitation (SE) blocks. Results are benchmarked against Khaliki & Başarslan (2024).

Data Sources
------------

-   **Br35H (Binary) Dataset**

    -   1,500 tumorous and 1,500 non-tumorous MRI scans; 60 unlabeled images for prediction.

    -   Used for evaluating logistic regression, SVM, Random Forest and a simple feed-forward network.

-   **Multiclass MRI Dataset**

    -   2,870 images (64 × 64 pixels), split into four classes: glioma, meningioma, pituitary tumor, and no tumor.

    -   Training/validation/testing split: 75%/15%/10%.

-   **Reference Paper**\
    Khaliki & Başarslan, *Scientific Reports* (01 Feb 2024).\
    DOI: [10.1038/s41598-024-52823-9](https://www.nature.com/articles/s41598-024-52823-9)\
    Benchmarks a 3-layer CNN against InceptionV3, EfficientNetB4, VGG16, and VGG19.

Methodology
-----------

### 1\. Binary Classification Pipeline

-   **Data Preparation**: Grayscale normalization, flattening, feature scaling.

-   **Models**:

    -   Logistic Regression (with cross-validation)

    -   Support Vector Machine (linear kernel)

    -   Random Forest

    -   XGBoost

    -   Simple Feed-Forward Neural Network

    -   Small Convolutional Neural Network

-   **Evaluation**: Classification reports, confusion matrices, ROC curves.

### 2\. Multiclass Classification Pipeline

-   **Data Loading**: Directory-based import of four classes.

-   **Exploratory Analysis**: Class distributions and sample visualizations.

-   **Augmentation**: Random rotations, flips, shifts, and zooms.

-   **Baseline Models**:

    -   3-Layer CNN from scratch

    -   Transfer learning with VGG16 (ImageNet weights)

-   **Custom Architectures**:

    -   **SE-Hybrid CNN**: Integrates Squeeze-and-Excitation blocks for channel-wise attention.

    -   **VGG16-Enhanced Hybrid CNN**: Combines SE-Hybrid feature maps with fine-tuned VGG16 embeddings.

-   **Training**: Adam optimizer, early stopping, batch size = 32, learning-rate scheduling.

-   **Fine-Tuning**: Unfreeze selected VGG16 layers with a reduced learning rate.

Requirements
------------

-   Python 3.8+

-   TensorFlow 2.x or PyTorch 1.x

-   scikit-learn

-   OpenCV

-   NumPy, pandas, matplotlib

Usage
-----

1.  **Binary Classification**\
    Open and run `1_BinaryClassification.ipynb` to reproduce feature-based and neural-network experiments.

2.  **Multiclass Classification**

    -   Run `2_MultiClass.ipynb` for the baseline CNN and VGG16 transfer-learning results.

    -   Run `hybrid.ipynb` to train and evaluate the SE-Hybrid and VGG16-Enhanced Hybrid models.


License
-----
MIT license
