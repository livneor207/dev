#!/usr/bin/env python3
"""Train the current best validated RGB ellipse-detector configuration.

This is intentionally a thin, explicit wrapper around ``scripts/train.py``:
the complete training command remains visible here, while the implementation
continues to live in one place.  Change ``--output-dir`` for every experiment;
never reuse a directory or write to ``runs/best_baseline``.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Best validation run as of 2026-09-12:
# runs/rgb_photometric_aug_dropout0p1 (epoch 44)
# angle MAE 5.55°, center MAE 0.00958, axes MAE 0.00954, F1 0.9989.
BEST_RGB_ARGUMENTS = [
    "--csv", "train_data.csv",
    "--base-dir", ".",
    # RGB residual CNN.  The 4-stage fine-grained backbone preserves the
    # detail needed to estimate ellipse orientation in 50×50 images.
    "--rgb",
    "--color-space", "rgb",
    "--fine-grained",
    "--depth", "4",
    "--base-filters", "32",
    "--kernel-size", "3",
    "--head-hidden-dim", "128",
    "--dropout", "0.1",
    # Regression and optimization selected by the best fixed-split validation
    # result.  The scheduler lowers LR when validation loss stops improving.
    "--cls-loss", "bce",
    "--reg-loss", "mse",
    "--optimizer", "adamw",
    "--lr", "0.001",
    "--weight-decay", "0.01",
    "--scheduler", "plateau",
    "--scheduler-patience", "5",
    "--scheduler-factor", "0.5",
    "--min-lr", "1e-6",
    # Photometric-only augmentation: it adds appearance robustness without
    # changing the target ellipse geometry.
    "--brightness-jitter", "0.1",
    "--contrast-jitter", "0.1",
    "--saturation-jitter", "0.1",
    "--grayscale-aug-prob", "0.1",
    "--blur-prob", "0.15",
    "--blur-radius", "0.75",
    "--batch-size", "128",
    "--epochs", "50",
    "--seed", "42",
]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    # A timestamp makes every invocation a separate, non-overwriting run.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = f"runs/rgb_photometric_aug_dropout0p1_repro_{timestamp}"
    command = [
        sys.executable,
        "scripts/train.py",
        *BEST_RGB_ARGUMENTS,
        "--output-dir",
        output_dir,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
