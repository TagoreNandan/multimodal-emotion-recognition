import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn


# =========================
# LOAD DATA
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
# MODEL DEFINITION
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

            nn.Linear(
                128,
                len(label_encoder.classes_)
            )
        )

    def forward(self, speech, text):

        speech_weight = self.speech_attention(speech)

        text_weight = self.text_attention(text)

        speech_attended = speech * speech_weight

        text_attended = text * text_weight

        fused = torch.cat(
            [speech_attended, text_attended],
            dim=1
        )

        output = self.classifier(fused)

        return output


# =========================
# LOAD TRAINED MODEL
# =========================

model = AttentionFusionModel()

model.load_state_dict(
    torch.load(
        "models/fusion_pipeline/attention_fusion_model.pth"
    )
)

model.eval()


# =========================
# TEST TENSORS
# =========================

speech_test_tensor = torch.tensor(
    speech_test,
    dtype=torch.float32
)

text_test_tensor = torch.tensor(
    text_test,
    dtype=torch.float32
)


# =========================
# PREDICTIONS
# =========================

with torch.no_grad():

    outputs = model(
        speech_test_tensor,
        text_test_tensor
    )

    preds = torch.argmax(
        outputs,
        dim=1
    ).numpy()


# =========================
# FIND ERRORS
# =========================

print("\n===== MISCLASSIFIED SAMPLES =====\n")

count = 0

for i in range(len(y_test)):

    if preds[i] != y_test[i]:

        actual = label_encoder.inverse_transform(
            [y_test[i]]
        )[0]

        predicted = label_encoder.inverse_transform(
            [preds[i]]
        )[0]

        print(f"Sample {i}")

        print(f"Actual    : {actual}")

        print(f"Predicted : {predicted}")

        print("-" * 40)

        count += 1

    if count == 10:
        break