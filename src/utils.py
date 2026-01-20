from pathlib import Path
from matplotlib import pyplot as plt


def plot_history_mae(history, out_path: Path):
    """Plot training and validation MAE from history object and save to out_path."""
    h = history.history
    epochs = range(1, len(h["mae"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, h["mae"], label="train_mae")
    plt.plot(epochs, h["val_mae"], label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Baseline (feature extraction) — MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()
