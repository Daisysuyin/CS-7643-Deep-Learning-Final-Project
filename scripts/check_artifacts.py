"""Validate tracked experiment outputs without CUDA, torch or network access."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    "classical_cnn",
    "classical_vit",
    "hybrid_cnn",
    "hybrid_vit",
    "quantum_vit_qattn",
)
REQUIRED = {"epoch", "train_loss", "val_loss", "val_acc"}


def main() -> None:
    for run in RUNS:
        csv_path = ROOT / "results" / f"{run}.csv"
        config_path = ROOT / "results" / f"{run}_config.json"
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames is not None, csv_path
            assert REQUIRED.issubset(reader.fieldnames), csv_path
            rows = list(reader)
        assert rows, f"No epoch data: {csv_path}"
        epochs = [int(row["epoch"]) for row in rows]
        assert epochs == list(range(1, len(rows) + 1)), csv_path
        for row in rows:
            for field in ("train_loss", "val_loss", "val_acc"):
                value = float(row[field])
                assert value == value, (csv_path, field)
            assert 0 <= float(row["val_acc"]) <= 1, csv_path
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        assert isinstance(config, dict), config_path
        print(f"{run}: {len(rows)} rows, best logged acc={max(float(x['val_acc']) for x in rows):.1%}")
    print("Recorded artifact structure checks passed (not model-training validation).")


if __name__ == "__main__":
    main()
