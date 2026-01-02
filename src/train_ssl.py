from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import yaml
import torch

from src.data.stl10_datamodule import STL10DataModule
from src.models.simclr import SimCLR
from src.utils.seed import set_seed


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    project_root = Path.cwd()
    cfg_path = project_root / args.config
    assert cfg_path.exists(), f"Config not found: {cfg_path}"

    cfg = load_yaml(cfg_path)

    # --- Global ---
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # --- Data ---
    data_cfg = cfg.get("data", {})
    data_dir = str(project_root / data_cfg.get("data_dir", "data"))
    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 4))
    image_size = int(data_cfg.get("image_size", 96))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    dm = STL10DataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        pin_memory=pin_memory,
    )
    dm.prepare_data()
    dm.setup()

    # --- Model ---
    model_cfg = cfg.get("model", {})
    model = SimCLR(
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        temperature=float(model_cfg.get("temperature", 0.2)),
        proj_hidden_dim=int(model_cfg.get("proj_hidden_dim", 2048)),
        proj_out_dim=int(model_cfg.get("proj_out_dim", 128)),
        use_bn_in_head=bool(model_cfg.get("use_bn_in_head", True)),
    )

    # --- Trainer ---
    trainer_cfg = cfg.get("trainer", {})
    max_epochs = int(trainer_cfg.get("max_epochs", 5))
    precision = trainer_cfg.get("precision", "16-mixed")
    accumulate = int(trainer_cfg.get("accumulate_grad_batches", 1))
    log_every_n_steps = int(trainer_cfg.get("log_every_n_steps", 20))

    # --- Artifacts dirs ---
    artifacts_dir = project_root / "artifacts"
    logs_root = ensure_dir(artifacts_dir / "logs" / "simclr")
    ckpt_root = ensure_dir(artifacts_dir / "checkpoints" / "simclr")
    metrics_root = ensure_dir(artifacts_dir / "metrics")

    # CSVLogger creates version_{i}
    logger = CSVLogger(save_dir=str(logs_root), name="", version=None)

    # Checkpointing: save both last and best
    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_root),
        filename="{run_id}",
        save_last=True,
        save_top_k=1,
        monitor="train_loss_epoch",
        mode="min",
        auto_insert_metric_name=False,
    )

    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=max_epochs,
        precision=precision,
        accumulate_grad_batches=accumulate,
        log_every_n_steps=log_every_n_steps,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=dm.train_dataloader())

    # --- Resolve run_id / paths ---
    # CSVLogger uses "version_X"; store a stable run_id:
    version = trainer.logger.version  # int
    run_id = f"simclr_version_{version}"

    # Move/organize checkpoints into ckpt_root/simclr_version_X/
    run_ckpt_dir = ensure_dir(ckpt_root / run_id)

    # Lightning saved into ckpt_root directly; collect best/last from callback paths.
    # Save as standard names last.ckpt and best.ckpt
    last_path = ckpt_cb.last_model_path
    best_path = ckpt_cb.best_model_path

    if last_path:
        src = Path(last_path)
        (run_ckpt_dir / "last.ckpt").write_bytes(src.read_bytes())

    if best_path:
        src = Path(best_path)
        (run_ckpt_dir / "best.ckpt").write_bytes(src.read_bytes())

    # Logger dir
    version_dir = Path(logger.log_dir)
    metrics_csv = version_dir / "metrics.csv"

    # Update runs_index.csv + summary.csv (training-only part)
    from src.utils.collect_metrics import upsert_run_index, upsert_summary_row

    upsert_run_index(
        metrics_dir=metrics_root,
        run_id=run_id,
        method="simclr",
        version_dir=version_dir.relative_to(project_root),
        metrics_csv=metrics_csv.relative_to(project_root),
        ckpt_last=(run_ckpt_dir / "last.ckpt").relative_to(project_root),
        ckpt_best=(run_ckpt_dir / "best.ckpt").relative_to(project_root) if (run_ckpt_dir / "best.ckpt").exists() else None,
        config_path=cfg_path.relative_to(project_root),
        seed=seed,
        max_epochs=max_epochs,
        batch_size=batch_size,
        temperature=float(model_cfg.get("temperature", 0.2)),
        lr=float(model_cfg.get("lr", 1e-3)),
    )

    # training summary values (take last epoch loss if exists)
    import pandas as pd
    train_loss_epoch_last = None
    if metrics_csv.exists():
        m = pd.read_csv(metrics_csv)
        if "train_loss_epoch" in m.columns:
            vals = m["train_loss_epoch"].dropna()
            if len(vals) > 0:
                train_loss_epoch_last = float(vals.iloc[-1])

    upsert_summary_row(
        metrics_dir=metrics_root,
        run_id=run_id,
        method="simclr",
        max_epochs=max_epochs,
        batch_size=batch_size,
        temperature=float(model_cfg.get("temperature", 0.2)),
        lr=float(model_cfg.get("lr", 1e-3)),
        train_loss_epoch_last=train_loss_epoch_last,
        ckpt_last=str((run_ckpt_dir / "last.ckpt").relative_to(project_root)),
        ckpt_best=str((run_ckpt_dir / "best.ckpt").relative_to(project_root)) if (run_ckpt_dir / "best.ckpt").exists() else "",
        config_path=str(cfg_path.relative_to(project_root)),
    )

    print("\n=== RUN DONE ===")
    print("run_id:", run_id)
    print("version_dir:", version_dir)
    print("metrics_csv:", metrics_csv)
    print("ckpt_last:", run_ckpt_dir / "last.ckpt")
    if (run_ckpt_dir / "best.ckpt").exists():
        print("ckpt_best:", run_ckpt_dir / "best.ckpt")


if __name__ == "__main__":
    main()
