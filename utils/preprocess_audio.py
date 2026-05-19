import librosa
import numpy as np

TARGET_SR = 16000
MAX_DURATION = 4  # seconds


def load_and_preprocess_audio(filepath):

    # Load audio and resample to 16kHz
    audio, sr = librosa.load(filepath, sr=TARGET_SR)

    # Remove silence from beginning and end
    audio, _ = librosa.effects.trim(audio)

    # Normalize audio amplitude
    audio = librosa.util.normalize(audio)

    # Define fixed audio length
    max_length = TARGET_SR * MAX_DURATION

    # Pad shorter audio
    if len(audio) < max_length:

        padding = max_length - len(audio)

        audio = np.pad(audio, (0, padding))

    # Truncate longer audio
    else:
        audio = audio[:max_length]

    return audio