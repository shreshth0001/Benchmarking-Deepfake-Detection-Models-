# Deepfake Detection Benchmarking Project

## Overview

This project provides a **unified, reproducible benchmark** for evaluating deepfake detection models focused on face-swapped manipulations. It implements and compares four popular CNN-based detectors—**Xception**, **Patch-ResNet**, **EfficientNetB0**, and **MesoNet**—using a consistent dataset, preprocessing pipeline, and evaluation protocol.

We aim to assess both the **effectiveness** (AUC, accuracy) and **efficiency** (model size, inference latency) of each model.

---

## Table of Contents

- [Background](#background)  
- [Dataset & Preprocessing](#dataset--preprocessing)  
- [Model Architectures](#model-architectures)  
- [Training & Evaluation](#training--evaluation)  
- [Results](#results)  
- [How to Run](#how-to-run)  
- [Project Structure](#project-structure)  
- [Limitations & Future Work](#limitations--future-work)  
- [References](#references)  

---

## Background

**Deepfakes** are AI-generated videos/images that replace or modify faces using autoencoders or GANs. While they have benign applications, they can also cause misinformation, privacy violations, and security threats.

---

## Dataset & Preprocessing

**Dataset:**
- **UADFV**: 98 videos (49 real, 49 fake; autoencoder-generated)

**Preprocessing:**
- Sample up to 10 frames per video  
- Detect faces using Haar cascade (OpenCV)  
- Crop and resize faces:  
  - `299×299` for Xception, Patch-ResNet, EfficientNetB0  
  - `256×256` for MesoNet  
- Normalize pixel values  
- Use 10 videos/class for training/testing for memory efficiency  

---

## Model Architectures

| Model         | Description                                            | Parameters (M) | Input Size     |
|---------------|--------------------------------------------------------|----------------|----------------|
| **Xception**      | Depthwise separable convs, ImageNet pretrained         | 20.86          | 299×299×3       |
| **Patch-ResNet**  | ResNet50 backbone; early conv features for texture     | 0.23           | 299×299×3       |
| **EfficientNetB0**| Compound scaling; optimized for accuracy/efficiency    | 4.05           | 299×299×3       |
| **MesoNet**       | Custom shallow CNN for fast, light inference           | 0.075          | 256×256×3       |

---

## Training & Evaluation

**Training Setup:**
- Epochs: `2`
- Batch Size: `16`
- Optimizer: `Adam (lr = 2e-4)`
- Loss: `Binary Cross-Entropy`
- Split: `80/20` stratified train/test, with 10% validation from training set

**Evaluation Metrics:**
- Frame-level AUC (ROC area per frame)
- Frame-level Accuracy
- Video-level AUC (mean of frame probabilities)
- Model size (trainable parameters)
- Inference time (ms/frame)

---

## Results

| Model         | Frame AUC | Frame Acc | Video AUC | Inference (ms) | Params (M) |
|---------------|-----------|-----------|-----------|----------------|------------|
| **Xception**      | 1.000     | 1.000     | 1.000     | 638.84         | 20.86      |
| **Patch-ResNet**  | 0.911     | 0.842     | 0.925     | 162.90         | 0.23       |
| **EfficientNetB0**| 0.447     | 0.474     | —         | 221.93         | 4.05       |
| **MesoNet**       | 0.858     | 0.526     | —         | 120.46         | 0.075      |

**Insights:**
- **Xception**: Best performance but heavy and slow
- **Patch-ResNet**: Great balance of speed and accuracy
- **MesoNet**: Light and fast, with solid AUC
- **EfficientNetB0**: Underperformed in this setup

---

## How to Run

1. **Clone** the repo and install dependencies (TensorFlow, Keras, OpenCV, scikit-learn).
2. **Download UADFV** dataset and organize into `data/real/` and `data/fake/`.
3. **Preprocess**: Extract face frames as described above.
4. **Train & Evaluate**: Use `ee656_code.ipynb` to run models.
5. **Analyze**: View AUC, accuracy, confusion matrices, ROC curves, and latency metrics.

---

## Project Structure

```
├── ee656_code.ipynb         # Main code (training, evaluation)  
├── EE656_Report.pdf         # Full report  
├── EE656_Deepfake_Detection_Presentation.pdf  # Slides  
├── data/  
│   ├── real/                # Real videos  
│   └── fake/                # Deepfake videos  
└── README.md                # This file
```



---

## Limitations & Future Work

**Limitations:**
- Small training size (10 videos/class)
- Only autoencoder-based deepfakes tested
- Basic face detection (Haar cascade)
- Only 2 epochs training
- No robustness or segment-level metrics

**Future Work:**
- Include GAN-based and more diverse datasets
- Use advanced detectors (e.g., MTCNN, RetinaFace)
- Try temporal models (e.g., CNN-LSTM, 3D CNNs)
- Add robustness benchmarks

---

## References

- 📄 [EE656_Deepfake_Detection_Presentation.pdf](./EE656_Deepfake_Detection_Presentation.pdf)
- 📘 [EE656_Report.pdf](./EE656_Report.pdf)
