"""Step 7: plot training and validation accuracy per epoch.

Reads outputs/history.json (written by train.py) and saves
training_accuracy.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

history_path = Path("outputs") / "history.json"
if not history_path.exists():
    raise SystemExit(f"{history_path} not found: run train.py first")
history = json.loads(history_path.read_text())

train_acc = history["train_metrics"]
valid_acc = history["valid_metrics"]
n_epochs = len(train_acc)
epochs = np.arange(n_epochs)

# The training accuracy is averaged over the epoch while the model is still
# learning, so it is plotted half an epoch earlier than the validation
# accuracy, which is measured at the end of each epoch (as in the notebook).
plt.figure(figsize=(8, 5))
plt.plot(epochs + 0.5, train_acc, ".--", label="Training")
plt.plot(epochs + 1.0, valid_acc, ".-", label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xlim(0, n_epochs + 0.5)
plt.xticks(range(0, n_epochs + 1, 2))
plt.grid()
plt.title("Fashion MNIST classifier: accuracy per epoch")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png")
print(f"final train accuracy: {train_acc[-1]:.4f}, "
      f"final valid accuracy: {valid_acc[-1]:.4f}, "
      f"best valid accuracy: {max(valid_acc):.4f} "
      f"(epoch {int(np.argmax(valid_acc)) + 1})")
print("saved training_accuracy.png")
plt.show()
