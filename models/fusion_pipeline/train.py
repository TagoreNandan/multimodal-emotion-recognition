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
# CONCATENATE FEATURES
# =========================

X = np.concatenate(
    [speech_X, text_X],
    axis=1
)

print("Fusion shape:", X.shape)


# =========================
# LABEL ENCODING
# =========================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)


# =========================
# TENSORS
# =========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
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
# DATALOADERS
# =========================

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
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
# FUSION MODEL
# =========================

class FusionMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

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

    def forward(self, x):

        return self.network(x)


# =========================
# MODEL SETUP
# =========================

model = FusionMLP()

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

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

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

    for batch_X, batch_y in test_loader:

        outputs = model(batch_X)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        all_preds.extend(
            preds.numpy()
        )

        all_labels.extend(
            batch_y.numpy()
        )


# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    all_labels,
    all_preds
)

print(f"\nFusion Accuracy: {accuracy:.4f}\n")


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
    cmap="Purples",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Fusion Confusion Matrix")

plt.savefig(
    "Results/fusion_confusion_matrix.png"
)

plt.close()


# =========================
# TRAINING LOSS
# =========================

plt.figure(figsize=(8, 5))

plt.plot(train_losses)

plt.title("Fusion Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.savefig(
    "Results/fusion_training_loss.png"
)

plt.close()


# =========================
# SAVE MODEL
# =========================

torch.save(
    model.state_dict(),
    "models/fusion_pipeline/fusion_model.pth"
)

print("\nFusion model saved successfully!")