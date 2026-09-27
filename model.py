"""Step 4: define the ImageClassifier MLP and the loss function.

Later steps import from this module, e.g.
`from model import ImageClassifier, model, xentropy`.
"""

import torch
import torch.nn as nn

from setup import device


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


torch.manual_seed(42)
model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                        n_classes=10).to(device)
xentropy = nn.CrossEntropyLoss()

if __name__ == "__main__":
    print(model)
    n_params = sum(param.numel() for param in model.parameters())
    print(f"parameters: {n_params:,}")
    X_dummy = torch.rand(32, 1, 28, 28, device=device)
    print(f"output shape for a batch of 32: {tuple(model(X_dummy).shape)}")
