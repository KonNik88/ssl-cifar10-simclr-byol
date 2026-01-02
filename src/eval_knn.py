from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.stl10_datamodule import STL10DataModule
from src.utils.io import latest_run, ensure_dir  # from your utils/io.py
from src.utils.embeddings import load_simclr_from_ckpt, infer_encoder_from_simclr, compute_embeddings
from src.utils.collect_metrics import upsert_summary_row
from src.models.simclr import SimCLR


def knn_predict(train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, k: int = 20) -> np.ndarray:
    sims = test_z @ train_z.T
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    topk_y = train_y[idx]
    pred = np.array([np.bincount(row, minlength=int(train_y.max()) + 1).argmax() for row in topk_y])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--use", type=str, default="best", choices=["best", "last"])
    args = ap.parse_args()

    root = Path(args.project_root)
    metrics_dir = root / "artifacts" / "metrics"
    run = latest_run(metrics_dir)

    run_id = run["run_id"]
    method = run["method"]
    ckpt_path = root / Path(run["ckpt_best"] if args.use == "best" and str(run.get("ckpt_best", "")).strip() else run["ckpt_last"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using run_id:", run_id)
    print("Using ckpt :", ckpt_path)

    simclr = load_simclr_from_ckpt(ckpt_path, device=device, model_cls=SimCLR, strict=True)
    encoder = infer_encoder_from_simclr(simclr).to(device).eval()

    dm = STL10DataModule(data_dir=str(root / "data"), batch_size=args.batch_size, num_workers=4, image_size=96)
    dm.prepare_data()
    dm.setup()

    tr_loader = DataLoader(dm.eval_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(dm.eval_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pack_tr = compute_embeddings(tr_loader, encoder, device=device, normalize=True)
    pack_te = compute_embeddings(te_loader, encoder, device=device, normalize=True)

    y_tr = pack_tr.y.astype(int)
    y_te = pack_te.y.astype(int)

    pred = knn_predict(pack_tr.z, y_tr, pack_te.z, k=args.k)
    acc = float((pred == y_te).mean())

    print(f"kNN@{args.k} accuracy: {acc:.4f}")

    # Update summary.csv for this run
    upsert_summary_row(
        metrics_dir=metrics_dir,
        run_id=run_id,
        method=method,
        max_epochs=int(run.get("max_epochs", "")) if str(run.get("max_epochs", "")).strip() else 0,
        batch_size=int(run.get("batch_size", "")) if str(run.get("batch_size", "")).strip() else args.batch_size,
        temperature=float(run.get("temperature", 0.2)),
        lr=float(run.get("lr", 1e-3)),
        train_loss_epoch_last=None,
        ckpt_last=str(run.get("ckpt_last", "")),
        ckpt_best=str(run.get("ckpt_best", "")),
        config_path=str(run.get("config_path", "")),
        knn_k=args.k,
        knn_acc=acc,
    )


if __name__ == "__main__":
    main()
