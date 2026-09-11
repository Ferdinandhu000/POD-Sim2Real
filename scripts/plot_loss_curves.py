#!/usr/bin/env python3
"""
Utility script to plot loss curves from POD-Sim2Real training runs.
Supports reading from:
1. artifacts/runs/<model>/pretrain_sim/history.json & finetune_real/history.json
2. metrics.jsonl files
3. Direct comparison between multiple models (e.g. triad-mno old vs new)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_history(stage_dir: Path) -> list[dict]:
    history_file = stage_dir / "history.json"
    if history_file.exists():
        try:
            return json.loads(history_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    metrics_file = stage_dir / "metrics.jsonl"
    if metrics_file.exists():
        records = []
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
        if records:
            return records

    return []


def plot_single_run(run_dir: Path, output_path: Path):
    sim_dir = run_dir / "pretrain_sim"
    real_dir = run_dir / "finetune_real"

    sim_hist = load_history(sim_dir)
    real_hist = load_history(real_dir)

    if not sim_hist and not real_hist:
        print(f"Warning: No history.json or metrics.jsonl found under {run_dir}")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    if sim_hist:
        epochs = [r["epoch"] for r in sim_hist]
        train_loss = [r.get("train_loss") for r in sim_hist]
        val_loss = [r.get("val_loss") for r in sim_hist]

        ax = axs[0]
        ax.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=1.5)
        ax.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=1.5, linestyle="--")
        ax.set_title(f"{run_dir.name}: Pretrain Sim Stage", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_yscale("log")
        ax.legend(frameon=True)

    if real_hist:
        epochs = [r["epoch"] for r in real_hist]
        train_loss = [r.get("train_loss") for r in real_hist]
        val_loss = [r.get("val_loss") for r in real_hist]

        ax = axs[1]
        ax.plot(epochs, train_loss, label="Train Loss", color="#2ca02c", linewidth=1.5)
        ax.plot(epochs, val_loss, label="Val Loss", color="#d62728", linewidth=1.5, linestyle="--")
        ax.set_title(f"{run_dir.name}: Finetune Real Stage", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_yscale("log")
        ax.legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved loss curve plot to {output_path}")


def plot_comparison(run_dirs: list[Path], labels: list[str], output_path: Path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for run_dir, label in zip(run_dirs, labels):
        sim_hist = load_history(run_dir / "pretrain_sim")
        real_hist = load_history(run_dir / "finetune_real")

        if sim_hist:
            epochs = [r["epoch"] for r in sim_hist]
            val_loss = [r.get("val_loss") for r in sim_hist]
            axs[0].plot(epochs, val_loss, label=f"{label} (Val)", linewidth=1.8)

        if real_hist:
            epochs = [r["epoch"] for r in real_hist]
            val_loss = [r.get("val_loss") for r in real_hist]
            axs[1].plot(epochs, val_loss, label=f"{label} (Val)", linewidth=1.8)

    axs[0].set_title("Pretrain Sim Validation Loss Comparison", fontsize=12, fontweight="bold")
    axs[0].set_xlabel("Epoch", fontsize=10)
    axs[0].set_ylabel("Validation Loss", fontsize=10)
    axs[0].set_yscale("log")
    axs[0].legend(frameon=True)

    axs[1].set_title("Finetune Real Validation Loss Comparison", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("Epoch", fontsize=10)
    axs[1].set_ylabel("Validation Loss", fontsize=10)
    axs[1].set_yscale("log")
    axs[1].legend(frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved comparison loss curve plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss curves from training runs")
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True, help="Path(s) to run directory under artifacts/runs")
    parser.add_argument("--labels", nargs="*", type=str, help="Labels for comparison")
    parser.add_argument("--output", type=Path, default=Path("loss_curve.png"), help="Output image file path")
    args = parser.parse_args()

    if len(args.run_dirs) == 1:
        plot_single_run(args.run_dirs[0], args.output)
    else:
        labels = args.labels or [p.name for p in args.run_dirs]
        plot_comparison(args.run_dirs, labels, args.output)


if __name__ == "__main__":
    main()
