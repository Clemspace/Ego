# File: cuda_test.py

import torch
import os

cuda_available = torch.cuda.is_available()
print(f"CUDA is available: {cuda_available}")

if cuda_available:
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available")

# Print environment details
print("Environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

print("Torch version:", torch.__version__)
