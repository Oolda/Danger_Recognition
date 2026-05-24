# Spatial-Temporal Weapon Anomaly Detection in CCTV

## Overview

This project presents a spatial-temporal deep learning framework for weapon anomaly detection in CCTV surveillance footage. Instead of performing direct object-level weapon detection, the proposed pipeline analyzes temporal reconstruction behavior using CNN-based spatial embeddings and recurrent autoencoder architectures.

The framework combines:

* CNN-based spatial feature extraction
* Temporal sequence construction
* LSTM / GRU / Attention-based autoencoders
* Reconstruction-based anomaly scoring
* XML-driven sequence-level evaluation
* Ablation study across multiple architectures

The project was developed as a deep learning research project focused on anomaly detection in surveillance environments.

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

The encoder transforms each frame into a compact 128-dimensional spatial embedding.

### Generated Feature Shapes

| Feature                   | Shape           |
| ------------------------- | --------------- |
| Training Spatial Features | (3615, 128)     |
| Testing Spatial Features  | (5149, 128)     |
| Training Sequences        | (3588, 10, 128) |
| Testing Sequences         | (5122, 10, 128) |

---

## Models

### Baseline LSTM Autoencoder

* Temporal sequence encoder-decoder
* Sequence-to-sequence reconstruction
* MSE reconstruction loss

### GRU Variant

* Reduced parameter count
* Faster recurrent alternative

### Attention-Based Variant

* Temporal attention weighting
* Enhanced sequence representation

### Dense Non-Temporal Baseline

* No recurrent temporal modeling
* Pure embedding reconstruction baseline

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

### Key Findings

* CNN spatial pretraining significantly improves anomaly detection performance.
* GRU achieves strong performance with fewer parameters.
* Attention provides competitive temporal modeling.
* Dense reconstruction performs surprisingly well, showing the strength of CNN spatial embeddings.

---

## Evaluation Strategy

Ground-truth labels are generated from XML annotations.

* Weapon classes: handgun, short rifle, knife
* Sequence labels are produced using 10-frame sliding windows
* Reconstruction errors are used as anomaly scores

Evaluation metrics include:

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

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
