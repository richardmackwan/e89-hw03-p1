# Summary

This repository rebuilds the section "Building an Image Classifier with PyTorch"
from `reference/10_neural_nets_with_pytorch.ipynb` as small Python scripts, one
per step. The table below lists each request, what was done, and the files it
produced. Every script was run before it was committed.

| # | Request | What was done | Files |
|---|---------|---------------|-------|
| 1 | Read the notebook section and start with `setup.py`: imports (torch, nn, torchvision, torchmetrics, matplotlib) and device selection (cuda → mps → cpu). | Read the section and the earlier notebook cells it depends on. Wrote the imports and the notebook's device check into a module-level `device` that later scripts import. Installed the libraries in the session and ran the script (it selected `cpu`). | `setup.py` |
| 2 | `load_data.py`: load FashionMNIST with `transforms.v2` (ToImage + ToDtype float32 scaled), split train 55,000/5,000 with seed 42, create train/valid/test DataLoaders with `batch_size=32`. | Implemented as in the notebook, including re-seeding with 42 before creating the loaders. Checked the split sizes (55,000 / 5,000 / 10,000) and batch shape (32, 1, 28, 28). Added a `.gitignore` for `datasets/` and `__pycache__/`. | `load_data.py`, `.gitignore` |
| 3 | `inspect_data.py`: print the shape, dtype and class of the first training sample; plot a few sample images with their labels. | Prints `(1, 28, 28)`, `torch.float32`, `Ankle boot`. Plots the first 10 training images with class names in a 2×5 grid and saves `sample_images.png`. Added `*.png` to `.gitignore`. | `inspect_data.py` (plus `sample_images.png`, not committed) |
| 4 | `model.py`: the `ImageClassifier` network (Flatten → 784→300→ReLU→100→ReLU→10) and `CrossEntropyLoss`, seed 42. | Defined the class as in the notebook, taking the layer sizes as arguments so it can be reused. Seeds with 42, creates `model` on `device`, and defines `xentropy`. Checked the output shape (32, 10) and the parameter count (266,610). | `model.py` |
| 5 | `train_utils.py`: `evaluate_tm` and `train2` as in the notebook, returning a history of train loss, train accuracy and validation accuracy. | Copied both helpers from earlier in the notebook. Kept the notebook's history keys (`train_losses`, `train_metrics`, `valid_metrics`) because the notebook's Optuna code reads `valid_metrics`. Checked with a quick 2-epoch run on a subset of the data. | `train_utils.py` |
| 6 | `train.py`: SGD with `lr=0.1`, torchmetrics multiclass Accuracy, 20 epochs; save the history and the trained model. | Trained for 20 epochs (about 4 minutes on CPU): final training accuracy 0.929, final validation accuracy 0.879, best validation accuracy 0.889 at epoch 16. Saves `outputs/history.json` and `outputs/fashion_mnist_weights.pt` (the `state_dict`). Added `outputs/` to `.gitignore`. | `train.py` (plus `outputs/`, not committed) |
| 7 | `plot_accuracy.py`: plot training (and validation) accuracy per epoch and save `training_accuracy.png`. **This was the required addition.** | Reads `outputs/history.json` and plots both curves. As in the notebook, training accuracy is plotted half an epoch earlier, because it is averaged during the epoch. Prints the final and best accuracies. The PNG was committed despite the `*.png` rule because it is the required deliverable. | `plot_accuracy.py`, `training_accuracy.png` |
| 8 | `predict.py`: predict the first 3 validation images; show predicted vs. true class names, softmax probabilities and top-4 probabilities. | Loads the saved weights into a new `ImageClassifier`. All 3 images are predicted correctly (Sneaker, Coat, Pullover). Prints the 3×10 softmax table and the top-4 predictions as class-name/probability pairs. Moves results to the CPU on every device, not only on MPS as the notebook does. | `predict.py` |
| 9 | `count_params.py`: print the number of model parameters. | Prints each layer's shape and count, then the total: 266,610, matching the notebook. | `count_params.py` |
| 10 | Where are the scripts? List them in execution order, then commit and push. | Listed the scripts from the repository root in execution order. Every script had already been committed and pushed after its own step, so there was nothing new to commit. | none |
| 11 | Write `SUMMARY.md` summarizing the dialog and commit it. | Wrote this file. | `SUMMARY.md` |

## How to run

```
python train.py          # trains the model and writes outputs/ (about 4 min on CPU)
python plot_accuracy.py  # writes training_accuracy.png
python predict.py
python count_params.py
python inspect_data.py   # optional, can run at any time
```

`setup.py`, `load_data.py`, `model.py` and `train_utils.py` are modules the
other scripts import; running them directly only prints a quick check.

Downloaded data (`datasets/`) and training outputs (`outputs/`) are not committed.
Running `train.py` recreates the outputs.
