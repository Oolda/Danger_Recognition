# Spatial-Temporal Weapon Anomaly Detection in CCTV

## Overview

This project presents a spatial-temporal deep learning framework for weapon anomaly detection in CCTV surveillance footage. Instead of performing direct object-level weapon detection, the proposed pipeline analyzes reconstruction behavior over temporal frame sequences using CNN-based spatial embeddings and autoencoder-based anomaly detection models.

The framework combines:

* CNN-based spatial feature extraction
* Temporal sequence construction
* LSTM / GRU / Attention-based autoencoders
* Dense non-temporal reconstruction baseline
* Reconstruction-based anomaly scoring
* XML-driven sequence-level evaluation
* Ablation study across multiple architectures

The project was developed as a deep learning research project focused on anomaly detection in surveillance environments.

---

## Why Anomaly Detection?

Traditional weapon detection systems mainly rely on supervised object detection pipelines trained using bounding-box annotations. However, dangerous events in surveillance environments are often rare, difficult to annotate at scale, visually ambiguous, and highly dependent on temporal context.

This project investigates whether reconstruction-based anomaly detection can identify weapon-related activity without directly learning weapon classes during training.

Instead of predicting weapon bounding boxes, the proposed framework models normal surveillance behavior and detects deviations through spatial-temporal reconstruction patterns.

This approach is especially relevant for CCTV scenarios because:

* Weapon visibility is often low-resolution and partially occluded.
* Dangerous activity may appear only briefly across consecutive frames.
* Fully supervised object detection requires expensive bounding-box annotations.
* Sequence-level reconstruction can capture deviations in both visual appearance and temporal behavior.
* XML annotations can be used only for evaluation rather than direct supervised training.

---

## Project Pipeline

```text
CCTV Frames
→ Grayscale + Resize (64×64)
→ CNN Spatial Encoder
→ 128D Spatial Embeddings
→ Sliding Temporal Windows
→ LSTM / GRU / Attention Autoencoders
→ Reconstruction Error
→ Anomaly Detection
```

---

## Dataset

This project uses the mock CCTV subset from:

**Real-time gun detection in CCTV: An open problem**

Dataset characteristics:

* 3 CCTV cameras
* Corridor and entrance surveillance scenes
* Low-resolution weapon visibility
* Cluttered backgrounds
* Realistic surveillance conditions

### Camera Distribution

| Camera | Frames |
| ------ | -----: |
| Cam1   |    607 |
| Cam5   |   1031 |
| Cam7   |   3511 |
| Total  |   5149 |

### Citation

```bibtex
@article{SalazarGonzalez2020,
  title = {Real-time gun detection in CCTV: An open problem},
  author = {Salazar Gonz{\'a}lez, Jose L. and Zaccaro, Carlos and {\'A}lvarez-Garc{\'i}a, Juan A. and Soria-Morillo, Luis M. and Sancho Caparrini, Fernando},
  journal = {Neural Networks},
  year = {2020},
  doi = {10.1016/j.neunet.2020.09.013}
}
```

---

## Difference from Original Dataset Paper

The original dataset paper focuses on supervised real-time gun detection, while this project reformulates the task as reconstruction-based anomaly detection over spatial-temporal CCTV sequences.

### Original Dataset Paper

* Real-time supervised gun detection
* Faster R-CNN + FPN + ResNet-50
* Bounding-box weapon localization
* Frame-level object detection
* Individual image-frame processing
* No explicit temporal sequence modeling
* Bounding-box annotations used during training
* Output: weapon bounding boxes and object detections

### This Project

* Spatial-temporal anomaly detection
* CNN embeddings + autoencoder-based reconstruction
* Sequence-based temporal modeling
* Reconstruction-based anomaly scoring
* Temporal frame-sequence processing
* LSTM / GRU / Attention / Dense architectures
* XML labels used only for evaluation
* Output: reconstruction-based anomaly scores

---

## Repository Structure

```text
Danger_Recognition/
│
├── notebooks/
│   ├── 01_spatial_feature_extraction.ipynb
│   ├── 02_xml_label_generation.ipynb
│   ├── 03_lstm_autoencoder_baseline.ipynb
│   ├── 04_gru_autoencoder_ablation.ipynb
│   ├── 05_lstm_attention_ablation.ipynb
│   └── 06_dense_no_temporal_baseline.ipynb
│
├── data/
│   ├── processed/
│   └── README.md
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reconstruction_errors/
│
├── models/
├── docs/
├── src/
├── requirements.txt
└── README.md
```

---

## Preprocessing

All CCTV frames are:

* Converted to grayscale
* Resized to 64×64
* Processed through a custom CNN encoder

The CNN encoder transforms each frame into a compact 128-dimensional spatial embedding.

These embeddings are then grouped into fixed-length temporal windows using a sliding-window strategy. Each sequence is reconstructed by the selected autoencoder model, and reconstruction error is used as the anomaly score.

### Generated Feature Shapes

| Feature                   | Shape           |
| ------------------------- | --------------- |
| Training Spatial Features | (3615, 128)     |
| Testing Spatial Features  | (5149, 128)     |
| Training Sequences        | (3588, 10, 128) |
| Testing Sequences         | (5122, 10, 128) |

---

## Proposed Experimental Architecture

The project evaluates multiple reconstruction-based architectures to understand the contribution of spatial pretraining, recurrent temporal modeling, attention mechanisms, and non-temporal dense reconstruction.

### Experiment 1 — Baseline LSTM Autoencoder

**Purpose:**

Evaluate sequence reconstruction performance using a standard recurrent encoder-decoder architecture.

**Architecture:**

* CNN spatial encoder
* 10-frame temporal sequences
* LSTM encoder-decoder autoencoder
* Mean squared error reconstruction loss
* Reconstruction error used as anomaly score

**Observation:**

The baseline LSTM autoencoder produced stable anomaly detection performance after CNN spatial pretraining.

---

### Experiment 2 — GRU Variant

**Purpose:**

Evaluate whether a lighter recurrent architecture can achieve comparable temporal reconstruction performance with fewer parameters.

**Architecture:**

* CNN spatial encoder
* 10-frame temporal sequences
* GRU encoder-decoder autoencoder
* Reduced recurrent parameter count
* Reconstruction-based anomaly scoring

**Observation:**

The GRU variant achieved competitive ROC-AUC performance while reducing parameter count compared to the baseline LSTM model.

---

### Experiment 3 — Attention-Based Variant

**Purpose:**

Investigate whether temporal attention improves the representation of important sequence regions.

**Architecture:**

* CNN spatial encoder
* LSTM-based temporal modeling
* Attention mechanism over temporal sequence representations
* Sequence reconstruction objective
* Reconstruction error used for anomaly detection

**Observation:**

The attention-based model produced stable performance and competitive anomaly classification behavior.

---

### Experiment 4 — Dense Non-Temporal Baseline

**Purpose:**

Measure how much anomaly information is already contained in CNN spatial embeddings without recurrent temporal modeling.

**Architecture:**

* CNN spatial encoder
* Dense reconstruction network
* No explicit recurrent temporal modeling
* Embedding-level reconstruction error
* Reconstruction-based anomaly scoring

**Observation:**

The dense baseline achieved surprisingly strong ROC-AUC performance, highlighting the discriminative power of CNN spatial features.

---

## Models

### Baseline LSTM Autoencoder

* Temporal sequence encoder-decoder
* Sequence-to-sequence reconstruction
* MSE reconstruction loss
* Captures temporal dependencies across frame embeddings

### GRU Variant

* Reduced parameter count
* Faster recurrent alternative
* Competitive sequence reconstruction performance
* Useful for lightweight temporal modeling

### Attention-Based Variant

* Temporal attention weighting
* Enhanced sequence representation
* Helps emphasize important regions within a sequence
* Produces interpretable temporal behavior

### Dense Non-Temporal Baseline

* No recurrent temporal modeling
* Pure embedding reconstruction baseline
* Measures the standalone discriminative strength of CNN spatial features

---

## Experimental Results

### Ablation Study

| Model Variant                     | ROC-AUC | Accuracy | Normal F1 | Anomaly F1 |
| --------------------------------- | ------: | -------: | --------: | ---------: |
| Untrained CNN + LSTM              |  0.4434 |     0.70 |      0.21 |       0.82 |
| Trained CNN + LSTM                |  0.8850 |     0.84 |      0.83 |       0.85 |
| Trained CNN + GRU                 |  0.9044 |     0.84 |      0.84 |       0.85 |
| Trained CNN + LSTM + Attention    |  0.9017 |     0.84 |      0.84 |       0.85 |
| Trained CNN + Dense (No Temporal) |  0.9211 |     0.84 |      0.83 |       0.84 |

---

## Updated Key Findings

* CNN spatial pretraining produced the largest performance improvement across all experiments.
* GRU achieved comparable performance to LSTM with fewer parameters.
* Attention-based temporal modeling produced stable anomaly detection behavior.
* The dense non-temporal baseline achieved surprisingly strong ROC-AUC performance, suggesting that CNN spatial embeddings already contain highly discriminative information.
* Temporal modeling improved interpretability and sequence-level analysis of CCTV activity.
* Reconstruction-based anomaly scoring provides an alternative to direct weapon-class prediction.
* XML annotations were useful for evaluation without requiring direct supervised object detection training.

---

## Evaluation Strategy

Ground-truth labels are generated from XML annotations.

* Weapon classes: handgun, short rifle, knife
* Sequence labels are produced using 10-frame sliding windows
* Reconstruction errors are used as anomaly scores
* XML labels are used only for evaluation, not as direct supervised training labels for weapon classification

A sequence is considered anomalous if weapon-related activity appears within the corresponding temporal window.

---

## Expanded Evaluation Metrics

### ROC-AUC

ROC-AUC measures the separability between normal and weapon-visible sequences across threshold values.

This metric is especially useful because anomaly detection systems often rely on reconstruction-error thresholds rather than fixed class predictions.

### F1-Score

F1-score balances precision and recall and is especially important for anomaly detection tasks where class imbalance can affect accuracy.

Both normal and anomaly F1-scores are reported to better understand class-level performance.

### Confusion Matrix

The confusion matrix provides class-level reconstruction-based anomaly classification behavior.

It helps evaluate whether the model is better at identifying normal surveillance sequences, anomalous weapon-visible sequences, or both.

### Additional Metrics

Additional reported metrics include:

* Accuracy
* Precision
* Recall
* Normal F1-score
* Anomaly F1-score

---

## Figures

The repository includes:

* Confusion matrices
* Ablation study visualizations
* Reconstruction error timelines
* Temporal anomaly score plots

All figures are available under:

```text
results/figures/
```

---

## Installation

```bash
git clone https://github.com/Oolda/Danger_Recognition.git
cd Danger_Recognition
pip install -r requirements.txt
```

---

## Contributors

* Beyza Nur Yazıcı
* Fatemeh Nasirian
* Adel Maddi
* Omar I. S. Diab
* Mohamed Kassas

---

## License

This project is intended for academic and research purposes only.
