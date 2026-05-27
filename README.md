---
Title: Multimodal Emotion Recognition
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Multimodal Emotion Recognition

*A Deep Learning Framework for Speech, Text, and Attention-Based Multimodal Emotion Classification*

---

## Overview

This project presents a multimodal emotion recognition system that combines **speech-based acoustic learning** and **text-based contextual understanding** for robust emotion classification.

The system was designed and implemented as an individual research-oriented project submission focused on:

* temporal emotion modelling from speech,
* contextual textual representation learning,
* and multimodal fusion using attention mechanisms.

The project evaluates and compares:

* Speech-only learning
* Text-only learning
* Feature Concatenation Fusion
* Attention-Based Fusion

A live deployment of the system is also available using Hugging Face Spaces.

---

## Key Features

* Speech Emotion Recognition using **Wav2Vec2 + BiLSTM**
* Text Emotion Recognition using **BERT + MLP**
* Multimodal Fusion using:

  * Feature Concatenation
  * Attention-Based Fusion
* t-SNE Visualization of learned embeddings
* Confusion Matrix and Training Curve Analysis
* Failure Case Analysis
* Live Streamlit Deployment on Hugging Face

---

## Architecture

### 1. Speech Pipeline

* **Feature Extractor:** Wav2Vec2
* **Temporal Modelling:** BiLSTM
* **Classifier:** Fully Connected Neural Network

The speech pipeline captures temporal acoustic patterns such as:

* pitch,
* tone,
* energy,
* and emotional prosody.

---

### 2. Text Pipeline

* **Feature Extractor:** BERT Embeddings
* **Classifier:** Multi-Layer Perceptron (MLP)

The text pipeline focuses on contextual semantic understanding from textual emotion representations.

---

### 3. Fusion Pipeline

Two multimodal fusion strategies were implemented:

#### A. Feature Concatenation Fusion

Speech and text embeddings are concatenated directly before classification.

#### B. Attention-Based Fusion

An attention mechanism dynamically learns modality importance and improves feature interaction between speech and text representations.

---

## Experimental Results

| Model            | Architecture           | Accuracy   |
| ---------------- | ---------------------- | ---------- |
| Speech-only      | Wav2Vec2 + BiLSTM      | **95.36%** |
| Text-only        | BERT + MLP             | **12.50%** |
| Fusion Concat    | Feature Concatenation  | **84.46%** |
| Attention Fusion | Attention-Based Fusion | **92.14%** |

### Key Observation

The speech modality produced the strongest emotional separability due to rich acoustic prosodic information present in vocal signals.

Attention-based fusion significantly improved multimodal performance compared to direct feature concatenation by dynamically emphasizing informative modality features.

---

## Emotion Cluster Visualization

t-SNE visualizations were generated for:

* Speech embeddings
* Text embeddings
* Fusion embeddings

These visualizations demonstrate the separability of learned emotional representations across different modelling blocks.

---

## Failure Analysis

Several challenging emotion pairs were identified during evaluation:

| Actual Emotion | Predicted Emotion | Reason                            |
| -------------- | ----------------- | --------------------------------- |
| Happy          | Pleasant Surprise | Similar acoustic energy patterns  |
| Neutral        | Sad               | Low-intensity vocal expressions   |
| Disgust        | Angry             | Overlapping tonal characteristics |

The text-only model showed weak generalization due to limited semantic diversity in textual emotion cues.

---

## Project Structure

```text
project/
├── models/
│
├── speech_pipeline/
│   ├── train.py
│   ├── test.py
│   └── emotion_bilstm.pth
│
├── text_pipeline/
│   ├── train.py
│   ├── test.py
│   └── text_mlp.pth
│
├── fusion_pipeline/
│   ├── train.py
│   ├── test.py
│   ├── fusion_concat_model.pth
│   └── attention_fusion_model.pth
│
├── Results/
│   ├── accuracy_tables/
│   ├── speech_pipeline/
│   ├── text_pipeline/
│   ├── fusion_pipeline/
│   └── failure_cases/
│
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

---

## Live Demo

GitHub Repository:
https://github.com/TagoreNandan/multimodal-emotion-recognition

Hugging Face Deployment:

[Multimodal Emotion Recognition Demo](https://huggingface.co/spaces/Tagorenandan22/multimodal-emotion-recognition?utm_source=chatgpt.com)

---

## How to Execute the Project

### 1. Clone the Repository

```bash
git clone https://github.com/TagoreNandan/multimodal-emotion-recognition
cd multimodal-emotion-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Download and organize the TESS dataset inside the project directory following the folder structure expected by the training scripts.

### 4. Run Individual Pipelines

Speech Pipeline:

```bash
python models/speech_pipeline/train.py
```

Text Pipeline:

```bash
python models/text_pipeline/train.py
```

Fusion Pipeline:

```bash
python models/fusion_pipeline/train.py
```

### 5. Launch the Web Application

```bash
streamlit run app.py
```

---

## Technologies Used

* Python
* PyTorch
* Transformers
* Wav2Vec2
* BERT
* Streamlit
* Scikit-learn
* Matplotlib
* Hugging Face Spaces

---

## Future Improvements

* Incorporating larger multimodal datasets
* Exploring Transformer-based fusion architectures
* Real-time streaming emotion recognition
* GPU-optimized training using larger-scale transformer models

---

## Author

**Atmakuri Tagore Nandan** <br>
B.Tech — Computer Science and Engineering <br>
Hyderabad Institute of Technology and Management

---

