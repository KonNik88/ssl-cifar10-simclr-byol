from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def project_root_from_file(file: str | Path) -> Path:
    """Resolve project root (assumes this file lives in src/utils/...)."""
    p = Path(file).resolve()
    return p.parents[2]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_runs_index(metrics_dir: str | Path) -> pd.DataFrame:
    metrics_dir = Path(metrics_dir)
    path = metrics_dir / "runs_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"runs_index.csv not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("runs_index.csv is empty")
    return df


def latest_run(metrics_dir: str | Path) -> dict:
    df = read_runs_index(metrics_dir)
    return df.iloc[-1].to_dict()


def resolve_metrics_csv(project_root: str | Path, run: dict) -> Path:
    project_root = Path(project_root)
    p = project_root / Path(run["metrics_csv"])
    if not p.exists():
        raise FileNotFoundError(f"metrics.csv not found: {p}")
    return p


def resolve_version_dir(project_root: str | Path, run: dict) -> Path:
    project_root = Path(project_root)
    p = project_root / Path(run["version_dir"])
    if not p.exists():
        raise FileNotFoundError(f"version_dir not found: {p}")
    return p


def find_checkpoint(project_root: str | Path, method: str, prefer: str = "last") -> Path:
    """
    Find checkpoint under artifacts/checkpoints/{method}/.
    prefer: 'last' or 'best' (string match in filename).
    """
    project_root = Path(project_root)
    ckpt_dir = project_root / "artifacts" / "checkpoints" / method
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    ckpts = list(ckpt_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found under: {ckpt_dir}")

    for c in ckpts:
        if c.name.lower() == f"{prefer}.ckpt":
            return c

    return max(ckpts, key=lambda p: p.stat().st_mtime)
