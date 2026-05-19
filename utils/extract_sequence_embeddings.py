import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from preprocess_audio import load_and_preprocess_audio


# Paths
DATASET_PATH = "data/raw/TESS Toronto emotional speech set data"

SAVE_PATH = "data/processed"


# Load wav2vec2
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

model.eval()


# Build dataframe
data = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            filename = file.replace(".wav", "")

            parts = filename.split("_")

            speaker = parts[0]
            word = parts[1]
            emotion = parts[2]

            data.append({
                "filepath": os.path.join(folder_path, file),
                "emotion": emotion
            })

df = pd.DataFrame(data)


# Storage
all_embeddings = []
all_labels = []


# Extract sequence embeddings
for _, row in tqdm(df.iterrows(), total=len(df)):

    filepath = row["filepath"]
    emotion = row["emotion"]

    # preprocess
    audio = load_and_preprocess_audio(filepath)

    # processor
    input_values = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    # wav2vec2 forward
    with torch.no_grad():

        outputs = model(input_values)

    # KEEP SEQUENCE
    embeddings = outputs.last_hidden_state.squeeze(0)

    # Convert to numpy
    embeddings = embeddings.numpy()

    all_embeddings.append(embeddings)

    all_labels.append(emotion)


# Convert
X = np.array(all_embeddings)

y = np.array(all_labels)


# Save
np.save(
    os.path.join(
        SAVE_PATH,
        "speech_sequence_embeddings.npy"
    ),
    X
)

np.save(
    os.path.join(
        SAVE_PATH,
        "speech_sequence_labels.npy"
    ),
    y
)

print("Saved successfully!")

print("Embedding shape:", X.shape)

print("Labels shape:", y.shape)