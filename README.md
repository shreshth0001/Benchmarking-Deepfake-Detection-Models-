# Benchmarking-Deepfake-Detection-Models-
Deepfake Detection Benchmarking Project
Overview
This project provides a unified, reproducible benchmark for evaluating deepfake detection models, focusing on face-swapped manipulations. It implements and compares four popular CNN-based detectors—Xception, Patch-ResNet, EfficientNetB0, and MesoNet—using a consistent dataset, preprocessing pipeline, and evaluation protocol. The goal is to transparently assess both the effectiveness (AUC, accuracy) and efficiency (model size, inference latency) of each model, enabling fair comparison and future research extensions12.
Table of Contents
	•	Background 	•	Dataset & Preprocessing 	•	Model Architectures 	•	Training & Evaluation 	•	Results 	•	How to Run 	•	Project Structure 	•	Limitations & Future Work 	•	References
Background
Deepfakes are AI-generated images or videos that swap or alter faces to create highly realistic but fake content. They are typically created using deep learning techniques such as autoencoders and GANs. While deepfakes have legitimate uses, they also pose risks for misinformation, privacy, and security12.
Dataset & Preprocessing
Dataset
	•	UADFV: 98 videos (49 real, 49 deepfake, autoencoder-generated)12 	•	Frame Sampling: Up to 10 uniformly spaced frames per video 	•	Face Detection: Haar-cascade classifier (OpenCV) for efficient face localization
Preprocessing Steps
	•	Face Cropping: Detected faces are cropped from each frame 	•	Resizing:
	•	299×299 for Xception, Patch-ResNet, EfficientNetB0 	•	256×256 for MesoNet
	•	Normalization: Pixel values scaled to 3 	•	Dataset Control: 10 videos per class used for training/testing for memory efficiency
Model Architectures
Model	Description	Parameters (M)	Input Size
Xception	Depthwise separable convolutions, pretrained on ImageNet	20.86	299×299×3
Patch-ResNet	ResNet50 backbone, features from early conv block for texture detection	0.23	299×299×3
EfficientNetB0	Compound scaling, optimized for accuracy and efficiency	4.05	299×299×3
MesoNet	Shallow custom CNN, designed for fast, lightweight inference	0.075	256×256×3

Training & Evaluation
Training Configuration
	•	Epochs: 2 	•	Batch Size: 16 	•	Optimizer: Adam (learning rate = 2 × 10⁻⁴) 	•	Loss Function: Binary Cross-Entropy 	•	Validation Split: 10% of training set 	•	Train/Test Split: 80/20 stratified split
Evaluation Metrics
	•	Frame-level AUC: Area under ROC curve for individual frames 	•	Frame-level Accuracy: Percentage of correctly classified frames 	•	Video-level AUC: Aggregated mean probability per video 	•	Model Size: Number of trainable parameters 	•	Inference Time: Average time per frame (ms)
Results
Performance Summary
Model	Frame-level AUC	Frame-level Accuracy	Video-level AUC	Inference Time (ms)	Parameters (M)
Xception	1.000	1.000	1.000	638.84	20.86
Patch-ResNet	0.911	0.842	0.925	162.90	0.23
EfficientNetB0	0.447	0.474	—	221.93	4.05
MesoNet	0.858	0.526	—	120.46	0.075

	•	Xception: Highest accuracy and AUC, but largest and slowest 	•	Patch-ResNet: Excellent trade-off between accuracy and efficiency 	•	MesoNet: Fastest and smallest, with competitive AUC 	•	EfficientNetB0: Underperformed in this setup12
How to Run
	1	Clone the repository and ensure all dependencies (TensorFlow, Keras, OpenCV, scikit-learn, etc.) are installed. 	2	Prepare the dataset: Download UADFV and organize into real/ and fake/ directories. 	3	Run preprocessing: Extract and preprocess face frames as described above. 	4	Train models: Use the provided Jupyter notebook (ee656_code.ipynb) to train and evaluate all four models. 	5	View results: The notebook outputs AUC, accuracy, confusion matrices, ROC curves, and efficiency metrics.
Project Structure

text ├── ee656_code.ipynb         # Main code notebook (training, evaluation, plots) ├── EE656_Report.pdf         # Detailed project report ├── EE656_Deepfake_Detection_Presentation.pdf  # Slide deck summary ├── data/ │   ├── real/                # Real videos │   └── fake/                # Deepfake videos ├── models/                  # Saved model weights └── README.md                # This file
Limitations & Future Work
	•	Small Dataset: Only 10 videos per class used for training/testing; limits generalizability 	•	Single Deepfake Technique: Focused on autoencoder-based face swaps, not GAN-based or other manipulations 	•	Face Detection: Haar cascade may miss non-frontal or occluded faces 	•	Minimal Training: Only 2 epochs due to compute constraints 	•	Excluded Metrics: No perturbation robustness or segment-level analysis12
Future Directions:
	•	Incorporate more diverse datasets and manipulation types 	•	Use advanced face detectors (e.g., MTCNN, RetinaFace) 	•	Explore temporal models (e.g., CNN-LSTM, 3D CNNs) 	•	Extend benchmarking to new detection methods
References
	•	1 EE656_Deepfake_Detection_Presentation.pdf 	•	2 EE656_Report.pdf
![image](https://github.com/user-attachments/assets/ae16af38-6e3b-4db9-882d-4fae5b3ad4fc)
