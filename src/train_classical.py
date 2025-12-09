import argparse
import os
import csv
import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data_utils import get_fashion_mnist_loaders
from models import ClassicalModelCNN, ClassicalModelViT
from utils.checkpoint_utils import ensure_dirs, save_checkpoint, save_config, save_log
from utils.logging_utils import init_csv, append_csv


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / total
    acc = correct / total
    return avg_loss, acc


def train(
    backbone: str = "cnn",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    run_name: str | None = None,
    log_dir: str = "../results",
    ckpt_dir: str = "../checkpoints",
    txt_log_dir: str = "../logs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Classical] Using device: {device}")

    ensure_dirs(log_dir, ckpt_dir, txt_log_dir)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"classical_{backbone}_{ts}"

    csv_path = os.path.join(log_dir, f"{run_name}.csv")
    best_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")
    last_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")
    log_file = os.path.join(txt_log_dir, f"{run_name}.txt")
    config_file = os.path.join(log_dir, f"{run_name}_config.json")

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    embedding_dim = 4
    if backbone == "cnn":
        model = ClassicalModelCNN(embedding_dim=embedding_dim, n_classes=10)
    elif backbone == "vit":
        model = ClassicalModelViT(embedding_dim=embedding_dim, n_classes=10)
    else:
        raise ValueError("backbone must be 'cnn' or 'vit'")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    init_csv(csv_path)
    save_config(
        {
            "backbone": backbone,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "run_name": run_name,
            "model_type": "classical",
        },
        config_file,
    )
    save_log("Run started.", log_file)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Classical {backbone}] Epoch {epoch}/{epochs}")
        running_loss = 0.0

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        append_csv(csv_path, epoch, train_loss, val_loss, val_acc)
        save_log(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}",
            log_file,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_ckpt_path,
                extra={
                    "val_acc": val_acc,
                    "backbone": backbone,
                    "run_name": run_name,
                },
            )

    save_checkpoint(
        model,
        optimizer,
        epochs,
        last_ckpt_path,
        extra={
            "val_acc": val_acc,
            "backbone": backbone,
            "run_name": run_name,
        },
    )
    save_log("Training completed.", log_file)
    print(f"[Classical] Saved best checkpoint to {best_ckpt_path}")
    print(f"[Classical] Saved last checkpoint to {last_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "vit"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="../results")
    parser.add_argument("--ckpt_dir", type=str, default="../checkpoints")
    parser.add_argument("--txt_log_dir", type=str, default="../logs")

    args = parser.parse_args()
    train(
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
        log_dir=args.log_dir,
        ckpt_dir=args.ckpt_dir,
        txt_log_dir=args.txt_log_dir,
    )
