import torch

print("PyTorch version:", torch.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS GPU is available")
else:
    device = torch.device("cpu")
    print("Using CPU")

x = torch.rand(3, 3).to(device)
print(x)