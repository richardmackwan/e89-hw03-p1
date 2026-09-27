"""Step 1: imports and device selection for the Fashion MNIST image classifier.

Later steps import from this module, e.g. `from setup import device`.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchmetrics
import torchvision

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

if __name__ == "__main__":
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, "
          f"torchmetrics {torchmetrics.__version__}")
    print(f"Using device: {device}")
