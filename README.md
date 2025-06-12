# Face Verification using MLP on CelebA Dataset

This project explores face verification using a Multi-Layer Perceptron (MLP) model trained on the CelebA dataset. The task involves analyzing the impact of various training factors (dataset size, learning rate, number of epochs) on the performance of a face identity verification system. We also investigate robustness to image perturbations.

---

## 📌 Tasks Overview

### ✅ Task 1: Dataset Size Impact

Train an MLP on randomly selected subsets of **10, 100, 500, 1000, and 5000** image pairs.
- Evaluate on a fixed **disjoint test set of 200 pairs**
- Ensure no image from training set appears in test
- Metrics used: **accuracy, precision, recall, F1-score**

📊 **Goal**: Analyze how performance improves with increasing training data.

---

### ✅ Task 2: Learning Rate Influence

Using a fixed training size of **1000 image pairs** and fixed number of epochs:
- Evaluate model performance with 5 learning rates (e.g. `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2`)

📊 **Goal**: Determine optimal learning rate for convergence and generalization.

---

### ✅ Task 3: Epoch Count Influence

With training size = **1000 pairs**, and best learning rate from Task 2:
- Evaluate performance using **5 epoch counts** (e.g. `5`, `10`, `20`, `30`, `50`)

📊 **Goal**: Understand underfitting/overfitting behavior over epochs.

---

### ✅ Task 4: Parameter Interaction and Optimization

- Analyze interaction between dataset size, learning rate, and epochs
- Identify the best performing combination
- Explore improvements:
  - **Adaptive LR schedules** (e.g. `ReduceLROnPlateau`)
  - **Early stopping** based on validation loss

📊 **Goal**: Establish best practice for face verification MLP training.

---

### 🌟 Bonus Task: Robustness to Image Perturbations

- Add perturbations (e.g. **Gaussian noise, blur, brightness shift**) to test set
- Measure drop in identity verification performance
- Propose and evaluate mitigation strategies:
  - **Training data augmentation**
  - **Preprocessing pipelines**
  - **Loss function adaptations**
  - **Architecture changes (e.g. dropout, batchnorm)**

---

## 🛠️ Technical Details

### 🔍 Theoretical Basis
The project uses pairwise learning and binary classification (same identity or not) based on image embeddings.

### ⚙️ Libraries Used

- `PyTorch` – model training and optimization
- `TorchVision` – CelebA dataset, transforms
- `scikit-learn` – metrics
- `NumPy`, `Matplotlib` – analysis and visualization
- `Pillow` – image augmentations

---

## 📑 Report Contents

Each experiment is described with:
- Objective
- Theoretical background
- Hyperparameter settings
- Code modifications
- Tabular & graphical results
- Interpretation

Final section includes:
- Summary of implementation challenges
- Lessons learned and best practices

---

## 🚀 Running Experiments

**Install dependencies:**
```bash
pip install -r requirements.txt
