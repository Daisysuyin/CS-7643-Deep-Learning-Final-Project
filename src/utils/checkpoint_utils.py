import os
import json
from datetime import datetime

import torch


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_config(config: dict, path: str):
    ensure_dirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def save_log(message: str, path: str):
    ensure_dirs(os.path.dirname(path))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict | None = None):
    ensure_dirs(os.path.dirname(path))
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[✓] Saved checkpoint: {path}")
