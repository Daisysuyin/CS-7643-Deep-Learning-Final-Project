import os
import csv

import matplotlib.pyplot as plt

RESULTS_DIR = "../results"
FIG_DIR = "../figures"


def load_experiment(name):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    epochs, train_loss, val_loss, val_acc = [], [], [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, val_loss, val_acc


def main():
    exp_names = ["classical_cnn", "classical_vit", "hybrid_cnn", "hybrid_vit"]
    colors = {
        "classical_cnn": "tab:blue",
        "classical_vit": "tab:orange",
        "hybrid_cnn": "tab:green",
        "hybrid_vit": "tab:red",
        "quantum_vit_qattn": "black",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Loss curves
    plt.figure()
    for name in exp_names:
        csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] Missing results for {name}, skipping.")
            continue
        epochs, train_loss, val_loss, _ = load_experiment(name)
        plt.plot(
            epochs,
            val_loss,
            label=f"{name} (val)",
            linestyle="-",
            color=colors.get(name, None),
        )
        plt.plot(
            epochs,
            train_loss,
            label=f"{name} (train)",
            linestyle="--",
            color=colors.get(name, None),
        )

    # Optionally add quantum ViT if present
    qt_csv = os.path.join(RESULTS_DIR, "quantum_vit_qattn.csv")
    if os.path.exists(qt_csv):
        epochs, train_loss, val_loss, _ = load_experiment("quantum_vit_qattn")
        plt.plot(
            epochs,
            val_loss,
            label="quantum_vit_qattn (val)",
            linestyle="-",
            color=colors["quantum_vit_qattn"],
        )
        plt.plot(
            epochs,
            train_loss,
            label="quantum_vit_qattn (train)",
            linestyle="--",
            color=colors["quantum_vit_qattn"],
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Val Loss (Classical vs Hybrid vs Quantum)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_path_results = os.path.join(RESULTS_DIR, "loss_curves.png")
    loss_path_figs = os.path.join(FIG_DIR, "loss_curves.png")
    plt.savefig(loss_path_results, dpi=200)
    plt.savefig(loss_path_figs, dpi=200)

    # Accuracy curves
    plt.figure()
    for name in exp_names:
        csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
        if not os.path.exists(csv_path):
            continue
        epochs, _, _, val_acc = load_experiment(name)
        plt.plot(epochs, val_acc, label=name, color=colors.get(name, None))

    if os.path.exists(qt_csv):
        epochs, _, _, val_acc = load_experiment("quantum_vit_qattn")
        plt.plot(epochs, val_acc, label="quantum_vit_qattn", color=colors["quantum_vit_qattn"])

    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Validation Accuracy (Classical vs Hybrid vs Quantum)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_path_results = os.path.join(RESULTS_DIR, "val_accuracy.png")
    acc_path_figs = os.path.join(FIG_DIR, "val_accuracy.png")
    plt.savefig(acc_path_results, dpi=200)
    plt.savefig(acc_path_figs, dpi=200)

    print("Saved plots to:")
    print(" ", loss_path_results)
    print(" ", loss_path_figs)
    print(" ", acc_path_results)
    print(" ", acc_path_figs)


if __name__ == "__main__":
    main()
