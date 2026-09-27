"""Step 9: print the number of model parameters, in total and per layer."""

from model import model

for name, param in model.named_parameters():
    print(f"{name:<14} {str(tuple(param.shape)):<12} {param.numel():>8,}")

n_params = sum(param.numel() for param in model.parameters())
print(f"total parameters: {n_params:,}")
