# multimodal-emotion-recognition

# Multimodal Emotion Recognition System


The project presents a **Multimodal Emotion Recognition System** capable of detecting human emotions using:

* **Speech-only input**
* **Text-only input**
* **Multimodal input (Speech + Text)**

The system combines modern Deep Learning and Transformer-based architectures for extracting emotional representations from both acoustic and textual modalities.

A complete end-to-end Streamlit application was also developed for real-time emotion prediction.

---

# Objective

The goal of this project is to build and compare:

1. **Speech-based Emotion Recognition**
2. **Text-based Emotion Recognition**
3. **Multimodal Emotion Recognition**

and analyze:

* modality effectiveness,
* fusion behavior,
* classification performance,
* representation separability.

---

# Architectures Used

| Block                      | Architecture           |
| -------------------------- | ---------------------- |
| Speech Pipeline            | wav2vec2 + BiLSTM      |
| Text Pipeline              | BERT + MLP             |
| Fusion Pipeline (Baseline) | Concatenation Fusion   |
| Fusion Pipeline (Advanced) | Attention-based Fusion |

---

# Project Structure

```text id="u7m2qp"
project/

├── models/
│
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   ├── test.py
│   │
│   ├── text_pipeline/
│   │   ├── train.py
│   │   ├── test.py
│   │
│   ├── fusion_pipeline/
│       ├── train.py
│       ├── test.py
│
├── Results/
│   ├── confusion matrices
│   ├── training curves
│   ├── t-SNE visualizations
│
├── utils/
│
├── notebooks/
│
├── app.py
│
├── README.md
│
└── requirements.txt
```

---

# Speech Emotion Recognition

## Architecture

* wav2vec2 Transformer embeddings
* BiLSTM temporal modeling
* Fully connected classifier

## Why This Architecture?

Speech emotions strongly depend on:

* tone,
* pitch,
* intensity,
* temporal acoustic patterns.

BiLSTM effectively captures sequential emotional dependencies in speech representations.

---

# Text Emotion Recognition

## Architecture

* BERT embeddings
* Multi-layer Perceptron (MLP)

## Why This Architecture?

BERT provides contextual semantic embeddings capable of capturing:

* sentence meaning,
* emotional semantics,
* contextual relationships.

---

# Multimodal Fusion

## 1. Concatenation Fusion

Speech and text embeddings were directly concatenated and passed through a classifier.

## 2. Attention Fusion

An attention mechanism was introduced to adaptively weight:

* speech representations,
* text representations.

---

# Experimental Results

| Model                | Accuracy |
| -------------------- | -------- |
| Speech MLP           | 93.2%    |
| Speech BiLSTM        | 96.0%    |
| Text BERT + MLP      | 12.0%    |
| Concatenation Fusion | 86.4%    |
| Attention Fusion     | 86.0%    |

---

# Key Findings

## Speech Dominates Emotion Recognition

Speech-based models achieved the highest performance because acoustic information strongly captures emotional characteristics.

## Text Modality Was Weak

The text-only model performed poorly due to limited lexical emotional information in the dataset.

## Fusion Did Not Improve Accuracy

Although multimodal fusion was expected to improve performance, the weak text modality limited the effectiveness of fusion.

## Attention Fusion Analysis

Attention-based fusion produced performance similar to concatenation fusion, indicating that:

* speech already contained dominant emotional information,
* text contributed limited additional emotional cues.

---

# Error Analysis

Some commonly confused emotions included:

| Actual Emotion | Predicted Emotion | Possible Reason          |
| -------------- | ----------------- | ------------------------ |
| happy          | ps                | Similar vocal excitement |
| angry          | disgust           | Harsh acoustic overlap   |
| sad            | neutral           | Low-energy similarity    |

---

# Representation Visualization

t-SNE visualizations were generated for:

* Speech embeddings
* Text embeddings
* Fusion embeddings

## Observations

* Speech embeddings formed clearer emotion clusters.
* Text embeddings showed significant overlap.
* Fusion embeddings partially preserved speech separability.

---

# Streamlit Demo Application

An interactive Streamlit application was developed supporting:

## Features

### Speech-only Emotion Prediction

Upload `.wav` audio and predict emotion.

### Text-only Emotion Prediction

Input text and predict emotion.

### Multimodal Emotion Prediction

Combine audio + text for multimodal emotion recognition.

---

# Running the Application

## 1. Activate Environment

```bash id="m4x8rk"
conda activate emotion311
```

## 2. Launch Streamlit App

```bash id="k1u9tv"
python -m streamlit run app.py --server.fileWatcherType none
```

---

# Dependencies

Install required packages:

```bash id="x6m2qp"
pip install -r requirements.txt
```

---

# Demo Screenshots

Include:

* speech prediction screenshots,
* text prediction screenshots,
* multimodal prediction screenshots,
* confusion matrices,
* t-SNE plots.

---

# 🏁 Conclusion

This project demonstrates that:

* Speech carries dominant emotional information.
* Text-only emotion recognition is highly dataset-dependent.
* Multimodal fusion effectiveness depends heavily on modality quality.
* Attention mechanisms cannot compensate for weak modality representations.

The project successfully implements:

* speech emotion recognition,
* text emotion recognition,
* multimodal emotion recognition,
* experimental analysis,
* visualization,
* real-time inference system.

---

# 👨‍💻 Author

Tagore Nandan Atmakuri

---

# 📚 Future Improvements

* Larger multimodal datasets
* Better text emotion datasets
* Cross-attention transformers
* Real-time microphone inference
* Model deployment to cloud platforms
