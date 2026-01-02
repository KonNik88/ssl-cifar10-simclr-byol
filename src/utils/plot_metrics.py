from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def _latest_version_dir(method_logs_dir: Path) -> Optional[Path]:
    """Find latest version_* directory by numeric suffix."""
    if not method_logs_dir.exists():
        return None
    candidates = []
    for p in method_logs_dir.glob("version_*"):
        try:
            v = int(p.name.split("_")[-1])
            candidates.append((v, p))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def parse_args():
    p = argparse.ArgumentParser(description="Plot Lightning CSVLogger metrics to PNG.")
    p.add_argument("--method", type=str, default="simclr", help="simclr / byol / ...")
    p.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts root directory")
    p.add_argument(
        "--metric",
        type=str,
        default="train_loss_epoch",
        help="Which metric column from metrics.csv to plot (e.g., train_loss_epoch, train_loss_step)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    method = args.method.lower().strip()
    artifacts_root = Path(args.artifacts)

    method_logs_dir = artifacts_root / "logs" / method
    vdir = _latest_version_dir(method_logs_dir)
    if vdir is None:
        raise FileNotFoundError(f"No runs found in {method_logs_dir}")

    metrics_csv = vdir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    metric = args.metric

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in {metrics_csv}. Available: {list(df.columns)}")

    df_plot = df[df[metric].notna()].copy()
    if df_plot.empty:
        raise ValueError(
            f"All values for '{metric}' are NaN in {metrics_csv}. "
            f"Try plotting another metric, e.g. 'train_loss_step'."
        )

    # Prefer x-axis by epoch for *_epoch metrics, otherwise by step
    if "epoch" in df_plot.columns and metric.endswith("_epoch"):
        x = df_plot["epoch"].astype(float)
        xlabel = "epoch"
    elif "step" in df_plot.columns:
        x = df_plot["step"].astype(float)
        xlabel = "step"
    else:
        x = range(len(df_plot))
        xlabel = "index"

    y = df_plot[metric].astype(float)

    figures_dir = artifacts_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{method}_{vdir.name}"
    out_path = figures_dir / f"{metric}_{run_id}.png"

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(metric)
    plt.title(f"{run_id}: {metric} (non-NaN only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Saved figure: {out_path}")


if __name__ == "__main__":
    main()
