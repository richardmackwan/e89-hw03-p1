"""Step 8: predict the first 3 validation images with the trained model.

Loads outputs/fashion_mnist_weights.pt (written by train.py) and prints the
predicted vs. true class names, the softmax probabilities and the top-4
probabilities.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from load_data import train_and_valid_data, valid_loader
from model import ImageClassifier
from setup import device

weights_path = Path("outputs") / "fashion_mnist_weights.pt"
if not weights_path.exists():
    raise SystemExit(f"{weights_path} not found: run train.py first")

model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
classes = train_and_valid_data.classes

model.eval()
X_new, y_new = next(iter(valid_loader))
X_new, y_new = X_new[:3].to(device), y_new[:3]
with torch.no_grad():
    y_pred_logits = model(X_new)
y_pred = y_pred_logits.argmax(dim=1).cpu()  # index of the largest logit

print("Predicted vs. true:")
for i, (pred, true) in enumerate(zip(y_pred, y_new)):
    mark = "correct" if pred == true else "WRONG"
    print(f"  image {i}: predicted {classes[pred]:<12} "
          f"true {classes[true]:<12} ({mark})")

# .cpu() because round() is not supported on MPS; it is a no-op on the CPU
y_proba = F.softmax(y_pred_logits, dim=1).cpu()
print("\nSoftmax probabilities (one row per image, one column per class):")
print(y_proba.round(decimals=3))

y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1).cpu()
print("\nTop-4 probabilities (softmax over the 4 largest logits):")
for i, (indices, probas) in enumerate(zip(y_top4_indices.cpu(),
                                          y_top4_probas)):
    top4 = ", ".join(f"{classes[index]} {proba:.3f}"
                     for index, proba in zip(indices, probas))
    print(f"  image {i}: {top4}")
