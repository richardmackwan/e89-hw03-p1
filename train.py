"""Step 6: train the ImageClassifier and save the history and the weights.

Outputs:
- outputs/history.json: per-epoch train loss, train accuracy, valid accuracy
- outputs/fashion_mnist_weights.pt: the trained model's state_dict
"""

import json
from pathlib import Path

import torch
import torchmetrics

from load_data import train_loader, valid_loader
from model import model, xentropy
from setup import device
from train_utils import train2

n_epochs = 20
output_dir = Path("outputs")
history_path = output_dir / "history.json"
weights_path = output_dir / "fashion_mnist_weights.pt"

if __name__ == "__main__":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accuracy = torchmetrics.Accuracy(task="multiclass",
                                     num_classes=10).to(device)
    history = train2(model, optimizer, xentropy, accuracy, train_loader,
                     valid_loader, n_epochs)

    output_dir.mkdir(exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), weights_path)
    print(f"saved {history_path} and {weights_path}")
