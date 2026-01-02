from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def upsert_run_index(
    metrics_dir: Path,
    run_id: str,
    method: str,
    version_dir: Path,
    metrics_csv: Path,
    ckpt_last: Path,
    ckpt_best: Optional[Path],
    config_path: Path,
    seed: int,
    max_epochs: int,
    batch_size: int,
    temperature: float,
    lr: float,
) -> None:
    """
    Maintains artifacts/metrics/runs_index.csv with stable paths.
    Paths are stored relative to project root.
    """
    metrics_dir = Path(metrics_dir)
    ensure_dir(metrics_dir)
    path = metrics_dir / "runs_index.csv"

    row = {
        "run_id": run_id,
        "method": method,
        "version_dir": str(version_dir).replace("\\", "/"),
        "metrics_csv": str(metrics_csv).replace("\\", "/"),
        "ckpt_last": str(ckpt_last).replace("\\", "/"),
        "ckpt_best": str(ckpt_best).replace("\\", "/") if ckpt_best is not None else "",
        "config_path": str(config_path).replace("\\", "/"),
        "seed": int(seed),
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "lr": float(lr),
    }

    df = _read_csv(path)
    if df.empty:
        df = pd.DataFrame([row])
        _write_csv(path, df)
        return

    if "run_id" not in df.columns:
        # backward-compat
        df = pd.DataFrame([row])
        _write_csv(path, df)
        return

    if (df["run_id"] == run_id).any():
        df.loc[df["run_id"] == run_id, :] = pd.DataFrame([row]).iloc[0].to_numpy()
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    _write_csv(path, df)


def upsert_summary_row(
    metrics_dir: Path,
    run_id: str,
    method: str,
    max_epochs: int,
    batch_size: int,
    temperature: float,
    lr: float,
    train_loss_epoch_last: Optional[float],
    ckpt_last: str,
    ckpt_best: str,
    config_path: str,
    knn_k: Optional[int] = None,
    knn_acc: Optional[float] = None,
    linear_epochs: Optional[int] = None,
    linear_acc: Optional[float] = None,
    fewshot_shots: Optional[int] = None,
    fewshot_acc: Optional[float] = None,
) -> None:
    """
    Maintains artifacts/metrics/summary.csv with training + eval metrics.
    Safe to call multiple times: updates by run_id.
    """
    metrics_dir = Path(metrics_dir)
    ensure_dir(metrics_dir)
    path = metrics_dir / "summary.csv"

    row = {
        "run_id": run_id,
        "method": method,
        "max_epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "lr": float(lr),
        "train_loss_epoch_last": train_loss_epoch_last if train_loss_epoch_last is not None else "",
        "knn_k": knn_k if knn_k is not None else "",
        "knn_acc": knn_acc if knn_acc is not None else "",
        "linear_epochs": linear_epochs if linear_epochs is not None else "",
        "linear_acc": linear_acc if linear_acc is not None else "",
        "fewshot_shots": fewshot_shots if fewshot_shots is not None else "",
        "fewshot_acc": fewshot_acc if fewshot_acc is not None else "",
        "ckpt_last": ckpt_last,
        "ckpt_best": ckpt_best,
        "config_path": config_path,
    }

    df = _read_csv(path)
    if df.empty:
        df = pd.DataFrame([row])
        _write_csv(path, df)
        return

    if "run_id" not in df.columns:
        df = pd.DataFrame([row])
        _write_csv(path, df)
        return

    if (df["run_id"] == run_id).any():
        # Update only provided fields, keep existing otherwise
        idx = df.index[df["run_id"] == run_id][0]
        for k, v in row.items():
            if v != "" and v is not None:
                df.at[idx, k] = v
        # also update always-these fields
        for k in ["method", "max_epochs", "batch_size", "temperature", "lr", "ckpt_last", "ckpt_best", "config_path"]:
            df.at[idx, k] = row[k]
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    _write_csv(path, df)
