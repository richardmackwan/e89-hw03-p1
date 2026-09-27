"""Step 3: inspect a training sample and plot a few images with their labels."""

import matplotlib.pyplot as plt

from load_data import train_and_valid_data, train_data

classes = train_and_valid_data.classes

X_sample, y_sample = train_data[0]
print(f"shape: {tuple(X_sample.shape)}")  # [channels, rows, columns]
print(f"dtype: {X_sample.dtype}")
print(f"class: {classes[y_sample]}")

n_rows, n_cols = 2, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2.4))
for index, ax in enumerate(axes.flat):
    X, y = train_data[index]
    ax.imshow(X.squeeze(0), cmap="binary")  # drop the channel dimension
    ax.set_title(classes[y], fontsize=10)
    ax.axis("off")
fig.tight_layout()
fig.savefig("sample_images.png")
print("saved sample_images.png")
plt.show()
