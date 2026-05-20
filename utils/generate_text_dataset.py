import os
import pandas as pd


DATASET_PATH = "data/raw/TESS Toronto emotional speech set data"

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

            sentence = f"The speaker said the word {word}"

            data.append({
                "text": sentence,
                "emotion": emotion
            })


df = pd.DataFrame(data)

print(df.head())

print("\nDataset Shape:", df.shape)


df.to_csv(
    "data/processed/text_dataset.csv",
    index=False
)

print("\nText dataset saved successfully!")