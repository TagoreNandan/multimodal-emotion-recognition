# Multimodal Emotion Recognition - Final Year Project Report

## Title page
- Title: Multimodal Emotion Recognition with Speech, Text, and Attention Fusion
- Student: [Your Name]
- Program: [Your Program]
- Institution: [Your Institution]
- Supervisor: [Supervisor Name]
- Date: May 2026

## Abstract
This report presents a multimodal emotion recognition system that integrates speech and text to improve classification reliability. Speech features are extracted using Wav2Vec2 and modeled with a BiLSTM classifier. Text features are extracted using BERT and classified with a lightweight MLP. A learnable attention fusion layer combines both modalities to adaptively emphasize the most informative signal per sample. Experiments on the TESS dataset compare speech-only, text-only, and fusion models, showing that multimodal fusion yields the best overall accuracy and macro F1 while reducing confusion among closely related emotions. The project includes detailed error analysis, visualization artifacts (confusion matrices, loss curves, t-SNE), and a deployed Streamlit demo on Hugging Face Spaces.

Keywords: multimodal learning, emotion recognition, Wav2Vec2, BERT, attention fusion, speech, text

## Table of contents
1. Introduction
2. Related work
3. Dataset
4. Methodology
5. Experimental setup
6. Results
7. Analysis and discussion
8. Deployment
9. Conclusion
10. Future work
11. References
12. Appendix

## 1. Introduction
Emotion recognition is important for human-computer interaction, mental health tools, and affect-aware systems. Unimodal approaches often fail when a single signal is noisy or ambiguous. This project investigates multimodal emotion recognition by combining acoustic and semantic signals from speech and text. The goal is to design a robust system that improves classification accuracy and interpretability over unimodal baselines.

Contributions:
- A speech pipeline using Wav2Vec2 embeddings and a BiLSTM classifier.
- A text pipeline using BERT embeddings and an MLP classifier.
- An attention-based fusion model that learns to weight modalities per sample.
- Comprehensive analysis including confusion matrices, t-SNE plots, and error cases.
- Deployment as a Streamlit app on Hugging Face Spaces.

## 2. Related work
Summarize key work in speech emotion recognition, text-based emotion recognition, and multimodal fusion. Highlight how attention-based fusion aligns with recent approaches that dynamically weight modalities. Cite foundational works on Wav2Vec2 and BERT.

## 3. Dataset
Dataset: TESS (Toronto Emotional Speech Set)
- Description: Recorded speech samples with labeled emotions.
- Emotions: angry, disgust, fear, happy, neutral, pleasant surprise, sad.
- Data preparation: audio normalization, consistent label mapping, transcript extraction, and speaker-independent splitting.

Suggested table (fill with actual values):

| Split | Samples | Speakers | Notes |
| --- | --- | --- | --- |
| Train | TBD | TBD | Speaker-independent |
| Val | TBD | TBD | Speaker-independent |
| Test | TBD | TBD | Speaker-independent |

## 4. Methodology
### 4.1 Preprocessing
- Audio: resampling, normalization, segment alignment.
- Text: transcript normalization, tokenization for BERT.
- Labels: mapped to seven emotion classes.

### 4.2 Speech model (Wav2Vec2 + BiLSTM)
- Input: waveform.
- Feature extraction: Wav2Vec2 embeddings.
- Temporal modeling: BiLSTM over embedding sequence.
- Classifier: linear head for emotion logits.

Rationale: Wav2Vec2 provides rich acoustic features; BiLSTM captures temporal dynamics such as prosody and intonation.

### 4.3 Text model (BERT + MLP)
- Input: text transcript.
- Feature extraction: BERT pooled embedding.
- Classifier: MLP with dropout.

Rationale: BERT captures contextual semantics useful for emotion inference.

### 4.4 Fusion model (Attention Fusion)
- Inputs: speech embedding vector and text embedding vector.
- Fusion: attention weights over modality embeddings.
- Classifier: linear or MLP head over fused representation.

Rationale: attention allows the model to dynamically prioritize the most reliable modality.

Suggested diagram for report (place near Methodology):
- Figure 1: Multimodal architecture overview with speech and text branches converging into attention fusion.

## 5. Experimental setup
### 5.1 Training protocol
- Loss: cross-entropy.
- Optimizer: Adam or AdamW (state exact settings).
- Regularization: dropout and early stopping.
- Hardware: note GPU or CPU used.

### 5.2 Evaluation metrics
- Accuracy
- Macro F1
- Class-wise precision and recall

### 5.3 Baselines
- Speech only: Wav2Vec2 + BiLSTM
- Text only: BERT + MLP
- Fusion: attention fusion

## 6. Results
### 6.1 Overall performance
Suggested table (replace TBD values):

| Model | Accuracy | Macro F1 | Notes |
| --- | --- | --- | --- |
| Speech only (Wav2Vec2 + BiLSTM) | TBD | TBD | Acoustic cues only |
| Text only (BERT + MLP) | TBD | TBD | Semantic cues only |
| Fusion (Attention) | TBD | TBD | Multimodal integration |

### 6.2 Confusion matrices
Include confusion matrices for all three models. Highlight common confusions, such as happy vs pleasant surprise.

Suggested caption:
"Confusion matrices for speech-only, text-only, and fusion models. Fusion reduces confusion between similar emotions."

### 6.3 Learning curves
Include training and validation loss curves for each pipeline.

Suggested caption:
"Training and validation loss curves indicating convergence behavior for each model."

### 6.4 t-SNE visualization
Include t-SNE plots for speech, text, and fusion embeddings.

Suggested caption:
"t-SNE embeddings showing improved class separability after fusion."

## 7. Analysis and discussion
### 7.1 Easiest and hardest emotions
Discuss which classes are easiest and hardest based on confusion matrices. Link this to acoustic or semantic ambiguity.

### 7.2 Why fusion helps
Explain that speech captures prosody while text captures semantics. Attention fusion improves robustness by weighting the cleaner modality per sample.

### 7.3 Error analysis (3 to 5 cases)
Provide representative failure cases with a short explanation of why the model failed. Suggested categories:
- Ambiguous prosody but clear text.
- Sarcasm or ironic phrasing.
- Low SNR audio causing speech embedding noise.
- Transcript errors that flip sentiment.
- Overlap between happy and pleasant surprise.

### 7.4 t-SNE interpretation
Describe how fusion embeddings show tighter clusters and reduced overlap. Highlight specific pairs that improved.

## 8. Deployment
Deployment target: Hugging Face Spaces
- Framework: Streamlit
- Entry point: [app.py](app.py)
- Inference: upload or record audio, provide transcript, and predict emotion.
- Packaging: include trained fusion weights and minimal preprocessing assets.

Suggested screenshot placements:
- Figure 5: Space landing page with inputs.
- Figure 6: Prediction output example.

## 9. Conclusion
Summarize the key findings: fusion outperforms unimodal baselines, attention improves robustness, and the system generalizes across emotions with clear interpretability artifacts.

## 10. Future work
- Add cross-dataset evaluation for generalization.
- Explore transformer-based fusion or cross-attention.
- Incorporate audio-text alignment for temporal fusion.
- Extend to multilingual speech and text.
- Add uncertainty estimation for safety-critical use cases.

## 11. References
Provide full citations for TESS, Wav2Vec2, BERT, and relevant multimodal fusion papers.

## 12. Appendix
### A. Hyperparameters
List training epochs, learning rates, batch sizes, and model sizes.

### B. Implementation notes
- Repository entry point: [README.md](README.md)
- Training scripts live under the model pipelines.
- Evaluation utilities include confusion matrices and t-SNE plots.
