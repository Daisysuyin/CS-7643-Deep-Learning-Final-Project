import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from data_utils import get_fashion_mnist_loaders
from models import ClassicalModelCNN, ClassicalModelViT, HybridModelCNN, HybridModelViT
from utils.checkpoint_utils import ensure_dirs, save_checkpoint, save_log, save_config
from utils.logging_utils import init_csv, append_csv


@dataclass
class Config:
    name: str
    model_type: str   # 'classical' or 'hybrid'
    backbone: str     # 'cnn' or 'vit'
    epochs: int
    batch_size: int
    lr: float


def evaluate(model: nn.Module, data_loader, device):
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


def run_single_experiment(cfg: Config, results_dir: str = "../results", checkpoints_dir: str = "../checkpoints", logs_dir: str = "../logs"):
    ensure_dirs(results_dir, checkpoints_dir, logs_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg.name}] Using device: {device}")

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=cfg.batch_size)

    n_qubits = 4
    n_layers = 2
    embedding_dim = n_qubits

    if cfg.model_type == "classical":
        if cfg.backbone == "cnn":
            model = ClassicalModelCNN(embedding_dim=embedding_dim, n_classes=10)
        elif cfg.backbone == "vit":
            model = ClassicalModelViT(embedding_dim=embedding_dim, n_classes=10)
        else:
            raise ValueError("backbone must be 'cnn' or 'vit'")
    elif cfg.model_type == "hybrid":
        if cfg.backbone == "cnn":
            model = HybridModelCNN(n_qubits=n_qubits, n_layers=n_layers, n_classes=10)
        elif cfg.backbone == "vit":
            model = HybridModelViT(n_qubits=n_qubits, n_layers=n_layers, n_classes=10)
        else:
            raise ValueError("backbone must be 'cnn' or 'vit'")
    else:
        raise ValueError("model_type must be 'classical' or 'hybrid'")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    csv_path = os.path.join(results_dir, f"{cfg.name}.csv")
    best_ckpt_path = os.path.join(checkpoints_dir, f"{cfg.name}_best.pt")
    last_ckpt_path = os.path.join(checkpoints_dir, f"{cfg.name}_last.pt")
    log_file = os.path.join(logs_dir, f"{cfg.name}.txt")
    config_file = os.path.join(results_dir, f"{cfg.name}_config.json")

    init_csv(csv_path)
    save_config(
        {
            "name": cfg.name,
            "model_type": cfg.model_type,
            "backbone": cfg.backbone,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
        },
        config_file,
    )
    save_log("Run started.", log_file)

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[{cfg.name}] Epoch {epoch}/{cfg.epochs}")
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
            f"[{cfg.name}][Epoch {epoch}] train_loss={train_loss:.4f} "
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
                    "cfg": cfg.__dict__,
                    "n_qubits": n_qubits,
                    "n_layers": n_layers,
                },
            )

    save_checkpoint(
        model,
        optimizer,
        cfg.epochs,
        last_ckpt_path,
        extra={
            "val_acc": val_acc,
            "cfg": cfg.__dict__,
            "n_qubits": n_qubits,
            "n_layers": n_layers,
        },
    )
    save_log("Training completed.", log_file)
    print(f"[{cfg.name}] Saved best checkpoint to {best_ckpt_path}")
    print(f"[{cfg.name}] Saved last checkpoint to {last_ckpt_path}")


def main():
    classical_epochs = 5
    hybrid_epochs = 3

    configs = [
        Config("classical_cnn", "classical", "cnn", classical_epochs, 64, 1e-3),
        Config("classical_vit", "classical", "vit", classical_epochs, 64, 1e-3),
        Config("hybrid_cnn", "hybrid", "cnn", hybrid_epochs, 32, 1e-3),
        Config("hybrid_vit", "hybrid", "vit", hybrid_epochs, 32, 1e-3),
    ]

    for cfg in configs:
        run_single_experiment(cfg, results_dir="../results", checkpoints_dir="../checkpoints", logs_dir="../logs")


if __name__ == "__main__":
    main()
