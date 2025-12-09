import os
import csv


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def init_csv(path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])
    print(f"[✓] Initialized CSV log: {path}")


def append_csv(path: str, epoch: int, train_loss: float, val_loss: float, val_acc: float):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, val_acc])
