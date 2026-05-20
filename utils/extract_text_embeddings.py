import pandas as pd
import numpy as np

from transformers import (
    BertTokenizer,
    BertModel
)

import torch


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "data/processed/text_dataset.csv"
)

texts = df["text"].tolist()

labels = df["emotion"].tolist()


# =========================
# LOAD BERT
# =========================

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertModel.from_pretrained(
    "bert-base-uncased"
)

model.eval()


# =========================
# EXTRACT EMBEDDINGS
# =========================

embeddings = []

with torch.no_grad():

    for text in texts:

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16
        )

        outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[
            :,
            0,
            :
        ]

        embeddings.append(
            cls_embedding.squeeze().numpy()
        )


# =========================
# SAVE EMBEDDINGS
# =========================

X = np.array(embeddings)

y = np.array(labels)

print("Embeddings shape:", X.shape)

print("Labels shape:", y.shape)


np.save(
    "data/processed/text_embeddings.npy",
    X
)

np.save(
    "data/processed/text_labels.npy",
    y
)

print("\nText embeddings saved successfully!")