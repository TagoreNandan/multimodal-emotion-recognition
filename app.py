import streamlit as st

import numpy as np
import librosa

import torch
import torch.nn as nn

from transformers import (
    AutoProcessor,
    Wav2Vec2Model,
    BertTokenizer,
    BertModel
)

from sklearn.preprocessing import LabelEncoder


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    layout="centered"
)

st.title("🎭 Multimodal Emotion Recognition")

st.write(
    "Speech + Text + Multimodal Emotion Detection"
)


# =========================
# LABELS
# =========================

emotion_labels = np.array([
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "ps",
    "sad"
])

label_encoder = LabelEncoder()

label_encoder.fit(emotion_labels)


# =========================
# LOAD WAV2VEC2
# =========================

@st.cache_resource
def load_wav2vec():

    processor = AutoProcessor.from_pretrained(
        "facebook/wav2vec2-base"
    )

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )

    model.eval()

    return processor, model


# =========================
# LOAD BERT
# =========================

@st.cache_resource
def load_bert():

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )

    model = BertModel.from_pretrained(
        "bert-base-uncased"
    )

    model.eval()

    return tokenizer, model


# =========================
# SPEECH MODEL
# =========================

class EmotionBiLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 7)
        )

    def forward(self, x):

        _, (hidden, _) = self.lstm(x)

        hidden = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        return self.classifier(hidden)


# =========================
# TEXT MODEL
# =========================

class TextMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 7)
        )

    def forward(self, x):

        return self.network(x)


# =========================
# ATTENTION FUSION MODEL
# =========================

class AttentionFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.speech_attention = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()
        )

        self.text_attention = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(

            nn.Linear(1536, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 7)
        )

    def forward(self, speech, text):

        speech_weight = self.speech_attention(speech)

        text_weight = self.text_attention(text)

        speech = speech * speech_weight

        text = text * text_weight

        fused = torch.cat(
            [speech, text],
            dim=1
        )

        return self.classifier(fused)


# =========================
# LOAD INDIVIDUAL MODELS
# =========================

@st.cache_resource
def load_speech_model():

    speech_model = EmotionBiLSTM()

    speech_model.load_state_dict(
        torch.load(
            "models/speech_pipeline/emotion_bilstm.pth",
            map_location="cpu"
        )
    )

    speech_model.eval()

    return speech_model


@st.cache_resource
def load_text_model():

    text_model = TextMLP()

    text_model.load_state_dict(
        torch.load(
            "models/text_pipeline/text_mlp.pth",
            map_location="cpu"
        )
    )

    text_model.eval()

    return text_model


@st.cache_resource
def load_fusion_model():

    fusion_model = AttentionFusionModel()

    fusion_model.load_state_dict(
        torch.load(
            "models/fusion_pipeline/attention_fusion_model.pth",
            map_location="cpu"
        )
    )

    fusion_model.eval()

    return fusion_model


# =========================
# AUDIO EMBEDDINGS
# =========================

def extract_speech_embedding(audio_file):

    wav_processor, wav_model = load_wav2vec()

    audio, sr = librosa.load(
        audio_file,
        sr=16000
    )

    inputs = wav_processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():

        outputs = wav_model(**inputs)

    sequence_embedding = (
        outputs.last_hidden_state
    )

    pooled_embedding = (
        sequence_embedding.mean(dim=1)
    )

    return sequence_embedding, pooled_embedding


# =========================
# TEXT EMBEDDINGS
# =========================

def extract_text_embedding(text):

    bert_tokenizer, bert_model = load_bert()

    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=32
    )

    with torch.no_grad():

        outputs = bert_model(**inputs)

    embedding = outputs.last_hidden_state[
        :,
        0,
        :
    ]

    return embedding


# =========================
# UI
# =========================

mode = st.selectbox(
    "Choose Mode",
    [
        "Speech Only",
        "Text Only",
        "Multimodal"
    ]
)


# =========================
# SPEECH ONLY
# =========================

if mode == "Speech Only":

    uploaded_audio = st.file_uploader(
        "Upload WAV File",
        type=["wav"]
    )

    if uploaded_audio:

        speech_model = load_speech_model()

        st.audio(uploaded_audio)

        sequence_embedding, _ = (
            extract_speech_embedding(
                uploaded_audio
            )
        )

        with torch.no_grad():

            outputs = speech_model(
                sequence_embedding
            )

            pred = torch.argmax(
                outputs,
                dim=1
            ).item()

        emotion = label_encoder.inverse_transform(
            [pred]
        )[0]

        st.success(
            f"Predicted Emotion: {emotion}"
        )


# =========================
# TEXT ONLY
# =========================

elif mode == "Text Only":

    text_input = st.text_input(
        "Enter Text"
    )

    if text_input:

        text_model = load_text_model()

        embedding = extract_text_embedding(
            text_input
        )

        with torch.no_grad():

            outputs = text_model(
                embedding
            )

            pred = torch.argmax(
                outputs,
                dim=1
            ).item()

        emotion = label_encoder.inverse_transform(
            [pred]
        )[0]

        st.success(
            f"Predicted Emotion: {emotion}"
        )


# =========================
# MULTIMODAL
# =========================

elif mode == "Multimodal":

    uploaded_audio = st.file_uploader(
        "Upload WAV File",
        type=["wav"]
    )

    text_input = st.text_input(
        "Enter Associated Text"
    )

    if uploaded_audio and text_input:

        fusion_model = load_fusion_model()

        st.audio(uploaded_audio)

        _, speech_embedding = (
            extract_speech_embedding(
                uploaded_audio
            )
        )

        text_embedding = (
            extract_text_embedding(
                text_input
            )
        )

        with torch.no_grad():

            outputs = fusion_model(
                speech_embedding,
                text_embedding
            )

            pred = torch.argmax(
                outputs,
                dim=1
            ).item()

        emotion = label_encoder.inverse_transform(
            [pred]
        )[0]

        st.success(
            f"Predicted Emotion: {emotion}"
        )