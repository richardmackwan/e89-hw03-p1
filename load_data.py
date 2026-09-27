"""Step 2: load Fashion MNIST and build the train/valid/test DataLoaders.

Later steps import from this module, e.g.
`from load_data import train_loader, valid_loader, test_loader`.
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

if __name__ == "__main__":
    print(f"train: {len(train_data)}, valid: {len(valid_data)}, "
          f"test: {len(test_data)}")
    X_sample, y_sample = train_data[0]
    print(f"sample shape: {tuple(X_sample.shape)}, dtype: {X_sample.dtype}, "
          f"range: [{X_sample.min():.1f}, {X_sample.max():.1f}], "
          f"class: {train_and_valid_data.classes[y_sample]}")
    X_batch, y_batch = next(iter(train_loader))
    print(f"batch shapes: X {tuple(X_batch.shape)}, y {tuple(y_batch.shape)}")
