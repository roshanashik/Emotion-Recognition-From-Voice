# 🎙️ Emotion Recognition from Voice using Transformer Networks and Hybrid Feature Extraction

![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Publication](https://img.shields.io/badge/Publication-IJRAR-success)

---

## 📌 Project Overview

**Emotion Recognition from Voice** is a deep learning–based system that automatically detects **human emotions from speech signals** using a **Transformer network** combined with **hybrid audio feature extraction**.

Unlike traditional speech processing systems that focus on linguistic content, this system emphasizes **emotional cues** embedded in speech using both:

* **Low-level acoustic features (MFCCs)** and
* **High-level semantic embeddings (VGGish)**

The project supports **file-based emotion recognition** and a **real-time voice emotion detection demo**, making it suitable for both academic research and real-world applications.

📍 *Developed as a Final Year B.Tech Project and accepted for publication.*

---

* System demo screenshot OR emotion prediction UI


![Project Banner](/Images/UI%201.png)


---

## 🎯 Objectives

* Automatically classify emotions from speech audio
* Extract and fuse **MFCC + VGGish** features
* Train and evaluate **Transformer-based SER model**
* Compare performance with **LSTM architecture**
* Enable **real-time emotion recognition**
* Achieve robust performance under noisy conditions

---

## 🧠 Emotion Classes

Based on the **RAVDESS dataset**, the system recognizes:

| Code | Emotion   |
| ---- | --------- |
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

---

## 🏗️ System Architecture

The system follows a **modular pipeline** consisting of:

1. Audio Input (File / Microphone)
2. Preprocessing & Normalization
3. Feature Extraction (MFCC + VGGish)
4. Hybrid Feature Fusion
5. Deep Learning Model (Transformer)
6. Emotion Prediction + Speech-to-Text
7. User Interface (Streamlit)

### 🖼️ Architecture Diagram

![System Architecture](/Images/emotion_recognition_system_architecture_3.png)

---

## 🔍 Feature Extraction Pipeline

### 🎵 MFCC (Mel-Frequency Cepstral Coefficients)

* 13 coefficients per frame
* Captures short-term spectral characteristics
* Mimics human auditory perception

### 🔊 VGGish Embeddings

* 128-dimensional deep audio embeddings
* Pretrained on large-scale YouTube audio
* Captures high-level semantic information

### 🔗 Hybrid Feature Fusion

* MFCCs and VGGish features are:

  * Extracted independently
  * Time-aligned
  * Concatenated into a single feature vector

📌 This fusion significantly improves emotion discrimination accuracy.

---

## 🧠 Model Architectures


### ⚡ Transformer Model (Proposed)

* Multi-head self-attention mechanism
* Parallel processing (faster training)
* Captures both short- and long-term dependencies
* Better generalization across emotions

📌 **Add image here:**


![Transformer Architecture](/Images/model%20architecture.png)


---

## 🔄 Data Augmentation Techniques

To improve robustness and generalization Several Augmentations Techniques are applied to Dataste :

* Noise Addition
* Pitch Shifting
* Time Stretching
* Pre-emphasis
* SpecAugment
* Reverberation
* Echo Addition

📌 *Augmentation significantly improved Transformer performance.*

---

## 🧪 Experimental Setup

### Hardware

* GPU-enabled system
* Minimum 16GB RAM

### Software

* Python 3.x
* TensorFlow 2.x
* Librosa
* Scikit-learn
* Streamlit

### Training Configuration

* Optimizer: Adam / AdamW
* Loss: Sparse Categorical Crossentropy
* Metrics: Accuracy, Precision, Recall, F1-score
* Batch Size: 32
* Epochs: up to 100 (Transformer)

---

## 📊 Results & Evaluation

| Emotion   | Precision | Recall | F1       |
| --------- | --------- | ------ | -------- |
| Neutral   | 0.86      | 0.89   | 0.88     |
| Calm      | 0.80      | 0.91   | 0.85     |
| Happy     | 0.84      | 0.87   | 0.85     |
| Sad       | 0.85      | 0.80   | 0.82     |
| Angry     | 0.91      | 0.89   | 0.90     |
| Fearful   | 0.88      | 0.86   | 0.87     |
| Disgust   | 0.89      | 0.84   | 0.86     |
| Surprised | **0.92**  | 0.86   | **0.89** |


---

## 🎙️ Real-Time Emotion Recognition (Demo)

* Microphone-based audio input
* Live emotion prediction
* Speech-to-text transcription
* Interactive **Streamlit UI**

![UI Design](/Images/UI%202.png)

---

## 🧪 How to Run

```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## 📰 Publication

📄 **Paper Title:**
*Emotion Recognition from Voice Using Transformer Networks and Hybrid Feature Extraction*

📌 **Accepted at:**
International Journal of Research and Analytical Reviews (IJRAR) Conference

---

## 🎓 Academic Context

* Final Year B.Tech Project
* Department of AI & Data Science
* MEA Engineering College, Perinthalmanna
* Guided by **Asst. Prof. Arsheenath Beegam**
* Team of 3 members

---

## 🚀 Future Work

* Multi-dataset training (CREMA-D, TESS, SAVEE)
* wav2vec / BERT-based audio models
* Mobile & web deployment
* Multimodal emotion recognition (face + voice)

---

## 🔗 Pretrained Model Files

This project uses **Google’s VGGish pretrained model**.
Large model files are managed via **Git LFS**, so after cloning the repository, you must run:

```bash
git lfs install
git lfs pull
```
## 🙏 Acknowledgements

* RAVDESS Dataset
* Google VGGish
* TensorFlow & Librosa communities

---
