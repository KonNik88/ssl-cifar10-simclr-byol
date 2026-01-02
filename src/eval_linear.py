from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.stl10_datamodule import STL10DataModule
from src.utils.io import latest_run
from src.utils.embeddings import load_simclr_from_ckpt, infer_encoder_from_simclr, compute_embeddings
from src.utils.collect_metrics import upsert_summary_row
from src.models.simclr import SimCLR


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
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

    pack_tr = compute_embeddings(tr_loader, encoder, device=device, normalize=False)
    pack_te = compute_embeddings(te_loader, encoder, device=device, normalize=False)

    Xtr = torch.tensor(pack_tr.z, dtype=torch.float32, device=device)
    ytr = torch.tensor(pack_tr.y.astype(int), dtype=torch.long, device=device)
    Xte = torch.tensor(pack_te.z, dtype=torch.float32, device=device)
    yte = torch.tensor(pack_te.y.astype(int), dtype=torch.long, device=device)

    num_classes = int(ytr.max().item()) + 1
    probe = LinearProbe(in_dim=Xtr.shape[1], num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        probe.train()
        logits = probe(Xtr)
        loss = loss_fn(logits, ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(Xte).argmax(dim=1)
            acc = float((pred == yte).float().mean().item())

        print(f"epoch {epoch+1:02d}/{args.epochs} | loss={loss.item():.4f} | test_acc={acc:.4f}")

    print("Final linear-probe acc:", acc)

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
        linear_epochs=args.epochs,
        linear_acc=acc,
    )


if __name__ == "__main__":
    main()
