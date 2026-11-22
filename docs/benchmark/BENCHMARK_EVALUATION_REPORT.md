# Benchmark Evaluation Report: Historical Broadsheet Ad vs. Non-Ad Classification

**Project:** IndicDLP Live Layout Studio  
**Date:** November 19, 2026  
**Dataset:** 100 Broadsheet Newspaper Scans (*Jugantar*, Jan 1946) — Batches 1 to 10  
**Target Task:** Binary Classification (`Ad` vs. `Non-Ad / Editorial Content`)

---

## 1. Executive Summary & Experimental Results

To address the evaluation benchmark requested by the professor, we trained and evaluated two representative convolutional architectures:
1. **LeNet-5 (Modified):** Classical shallow convolutional baseline (2 conv layers, max pooling, dense linear head).
2. **ResNet-18 (Fine-Tuned):** Deep residual baseline with residual connections and Global Average Pooling (GAP).

Both models were trained on the exact same stratified training split ($N = 5,764$) with weighted Cross-Entropy Loss to handle the 1:3.45 class imbalance, validated on the validation split ($N = 1,235$) for early model selection, and evaluated **strictly once on the unseen test set ($N = 1,237$)**.

### Comprehensive Test Set Evaluation ($N = 1,237$)

| Architecture | Model Class | Test Accuracy | Precision (Ad) | Recall (Ad) | Ad-F1 Score | Macro-F1 | ROC-AUC | False Negatives (Missed Ads) | False Positives (Editorial as Ad) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LeNet-5 (Modified)** | Shallow Baseline | 96.69% | 89.63% | 96.40% | **92.89%** | 95.37% | 0.9927 | 10 / 278 | 31 / 959 |
| **ResNet-18** | Deep Residual | **98.71%** | **95.17%** | **99.28%** | **97.18%** | **98.17%** | **0.9985** | **2 / 278** | **14 / 959** |

---

## 2. Confusion Matrix Breakdown

### Model 1: LeNet-5
```
                      Predicted Non-Ad     Predicted Ad
Actual Non-Ad (959)         928                 31
Actual Ad     (278)          10                268
```
- **True Positives (Ad):** 268
- **True Negatives (Non-Ad):** 928
- **Missed Advertisements:** 10 (3.60% miss rate)

### Model 2: ResNet-18
```
                      Predicted Non-Ad     Predicted Ad
Actual Non-Ad (959)         945                 14
Actual Ad     (278)           2                276
```
- **True Positives (Ad):** 276
- **True Negatives (Non-Ad):** 945
- **Missed Advertisements:** Only 2 out of 278 (0.72% miss rate, **99.28% Recall**)

---

## 3. Addressing the Professor's Key Research Question

> **Professor's Question:** *"How will you manage and train with bounding boxes of different sizes?"*

### Implemented Methodology: Aspect-Ratio Preserving Letterboxing ($224 \times 224$)
1. **The Problem:** Direct naive resizing (e.g. stretching a vertical $120 \times 800$ ad or squashing a horizontal banner into $224 \times 224$) distorts character aspect ratio, stroke thickness, line height, and high-frequency typographic textures.
2. **The Solution:** 
   $$\text{scale} = \frac{224}{\max(W, H)}$$
   The crop is scaled along its longest edge while strictly preserving its natural geometric aspect ratio. The scaled image is then centered onto a uniform neutral white canvas ($224 \times 224$, pixel fill = 255).
3. **Outcome:** Both architectures converged smoothly without spatial distortion artifacts, enabling robust classification across arbitrary bounding box dimensions (from small $80 \times 90$ classified snippets to $1200 \times 600$ half-page display ads).

---

## 4. Handling Class Imbalance

The natural broadsheet distribution contains significantly more editorial paragraphs than commercial ads:
- **Total Dataset:** 1,849 Ads (22.45%) vs. 6,387 Non-Ads (77.55%)
- **Stratified Split:**
  - **Train (70%):** 1,294 Ads, 4,470 Non-Ads ($N = 5,764$)
  - **Val (15%):** 277 Ads, 958 Non-Ads ($N = 1,235$)
  - **Test (15%):** 278 Ads, 959 Non-Ads ($N = 1,237$)

### Inverse Class Frequency Loss Formulation:
$$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$
- $w_{\text{non\_ad}} = \mathbf{0.6447}$
- $w_{\text{ad}} = \mathbf{2.2272}$

Penalizing ad misclassifications by $3.45\times$ prevented the networks from defaulting to majority-class guessing, yielding **96.40% recall for LeNet-5** and **99.28% recall for ResNet-18**.

---

## 5. Model Checkpoints & Artifact Locations

- **LeNet-5 Best Weights:** `checkpoints/best_lenet.pt`
- **ResNet-18 Best Weights:** `checkpoints/best_resnet18.pt`
- **Dataset Directory:** `C:\Users\shiny\Desktop\Ad_Dataset`
- **Dataset Manifest:** `C:\Users\shiny\Desktop\Ad_Dataset\dataset_manifest.json`
- **Benchmark Scripts:**
  - Data Pipeline: `dataset_builder.py` & `ad_dataset.py`
  - Model Definitions: `models_benchmark.py`
  - Training & Evaluation Engine: `train_benchmark.py`
