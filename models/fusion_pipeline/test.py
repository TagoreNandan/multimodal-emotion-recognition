import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# =========================
# LOAD EMBEDDINGS
# =========================

speech_X = np.load(
    "data/processed/speech_embeddings.npy"
)

text_X = np.load(
    "data/processed/text_embeddings.npy"
)

y = np.load(
    "data/processed/speech_labels.npy",
    allow_pickle=True
)


# =========================
# LABEL ENCODING
# =========================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================
# TRAIN TEST SPLIT
# =========================

(
    speech_train,
    speech_test,
    text_train,
    text_test,
    y_train,
    y_test

) = train_test_split(

    speech_X,
    text_X,
    y_encoded,

    test_size=0.2,
    random_state=42
)


# =========================
# TENSORS
# =========================

speech_train = torch.tensor(
    speech_train,
    dtype=torch.float32
)

speech_test = torch.tensor(
    speech_test,
    dtype=torch.float32
)

text_train = torch.tensor(
    text_train,
    dtype=torch.float32
)

text_test = torch.tensor(
    text_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# =========================
# DATASET
# =========================

train_dataset = TensorDataset(
    speech_train,
    text_train,
    y_train
)

test_dataset = TensorDataset(
    speech_test,
    text_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)


# =========================
# GATED ATTENTION FUSION
# =========================

class AttentionFusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Speech attention
        self.speech_attention = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()
        )

        # Text attention
        self.text_attention = nn.Sequential(

            nn.Linear(768, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()
        )

        # Final classifier
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

            nn.Linear(
                128,
                len(label_encoder.classes_)
            )
        )

    def forward(self, speech, text):

        # Attention weights
        speech_weight = self.speech_attention(speech)

        text_weight = self.text_attention(text)

        # Apply attention
        speech_attended = speech * speech_weight

        text_attended = text * text_weight

        # Fusion
        fused = torch.cat(
            [speech_attended, text_attended],
            dim=1
        )

        # Classification
        output = self.classifier(fused)

        return output


# =========================
# MODEL SETUP
# =========================

model = AttentionFusionModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =========================
# TRAINING
# =========================

EPOCHS = 20

train_losses = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for speech_batch, text_batch, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(
            speech_batch,
            text_batch
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Loss: {avg_loss:.4f}"
    )


# =========================
# EVALUATION
# =========================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for speech_batch, text_batch, labels in test_loader:

        outputs = model(
            speech_batch,
            text_batch
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

print(
    f"\nAttention Fusion Accuracy: "
    f"{accuracy:.4f}\n"
)


# =========================
# REPORT
# =========================

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_
    )
)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Attention Fusion Confusion Matrix"
)

plt.savefig(
    "Results/attention_fusion_confusion_matrix.png"
)

plt.close()


# =========================
# TRAINING LOSS
# =========================

plt.figure(figsize=(8, 5))

plt.plot(train_losses)

plt.title(
    "Attention Fusion Training Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig(
    "Results/attention_fusion_training_loss.png"
)

plt.close()


# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "models/fusion_pipeline/attention_fusion_model.pth"
)

print(
    "\nAttention fusion model saved successfully!"
)