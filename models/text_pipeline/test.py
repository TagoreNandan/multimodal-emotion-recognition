import torch

MODEL_PATH = "text_mlp.pth"

try:
    model = torch.load(MODEL_PATH, map_location="cpu")
    print("Text emotion model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")