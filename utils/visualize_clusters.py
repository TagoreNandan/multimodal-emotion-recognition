import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


# =========================
# LOAD EMBEDDINGS
# =========================

speech_X = np.load(
    "data/processed/speech_embeddings.npy"
)

text_X = np.load(
    "data/processed/text_embeddings.npy"
)

labels = np.load(
    "data/processed/speech_labels.npy",
    allow_pickle=True
)


# =========================
# CREATE FUSION EMBEDDINGS
# =========================

fusion_X = np.concatenate(
    [speech_X, text_X],
    axis=1
)


# =========================
# LABEL ENCODING
# =========================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(labels)

emotion_names = label_encoder.classes_


# =========================
# t-SNE FUNCTION
# =========================

def plot_tsne(data, title, save_path):

    print(f"\nRunning t-SNE for {title}...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42
    )

    reduced = tsne.fit_transform(data)

    plt.figure(figsize=(10, 8))

    for i, emotion in enumerate(emotion_names):

        indices = y == i

        plt.scatter(
            reduced[indices, 0],
            reduced[indices, 1],
            label=emotion,
            alpha=0.7
        )

    plt.legend()

    plt.title(title)

    plt.xlabel("t-SNE Dimension 1")

    plt.ylabel("t-SNE Dimension 2")

    plt.savefig(save_path)

    plt.close()

    print(f"Saved: {save_path}")


# =========================
# SPEECH CLUSTERS
# =========================

plot_tsne(
    speech_X,
    "Speech Embedding Clusters",
    "Results/speech_tsne.png"
)


# =========================
# TEXT CLUSTERS
# =========================

plot_tsne(
    text_X,
    "Text Embedding Clusters",
    "Results/text_tsne.png"
)


# =========================
# FUSION CLUSTERS
# =========================

plot_tsne(
    fusion_X,
    "Fusion Embedding Clusters",
    "Results/fusion_tsne.png"
)


print("\nAll cluster visualizations generated!")