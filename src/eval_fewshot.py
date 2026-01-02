from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.stl10_datamodule import STL10DataModule
from src.utils.io import latest_run, find_checkpoint
from src.utils.embeddings import load_simclr_from_ckpt, infer_encoder_from_simclr, compute_embeddings
from src.models.simclr import SimCLR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--shots", type=int, default=5, help="samples per class")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.project_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = latest_run(root / "artifacts" / "metrics")
    method = run["method"]
    ckpt = find_checkpoint(root, method=method, prefer="last")
    print("Using ckpt:", ckpt)

    simclr = load_simclr_from_ckpt(ckpt, device=device, model_cls=SimCLR, strict=True)
    encoder = infer_encoder_from_simclr(simclr).to(device).eval()

    dm = STL10DataModule(data_dir=str(root / "data"), batch_size=args.batch_size, num_workers=4, image_size=96)
    dm.prepare_data()
    dm.setup()

    tr_loader = DataLoader(dm.eval_train, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(dm.eval_test, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    pack_tr = compute_embeddings(tr_loader, encoder, device=device, normalize=False)
    pack_te = compute_embeddings(te_loader, encoder, device=device, normalize=False)

    X = pack_tr.z
    y = pack_tr.y.astype(int)
    num_classes = int(y.max()) + 1

    # few-shot subset
    idx_keep = []
    rng = np.random.default_rng(42)
    for c in range(num_classes):
        idx_c = np.where(y == c)[0]
        pick = rng.choice(idx_c, size=min(args.shots, len(idx_c)), replace=False)
        idx_keep.append(pick)
    idx_keep = np.concatenate(idx_keep)

    Xfs = torch.tensor(X[idx_keep], dtype=torch.float32, device=device)
    yfs = torch.tensor(y[idx_keep], dtype=torch.long, device=device)

    Xte = torch.tensor(pack_te.z, dtype=torch.float32, device=device)
    yte = torch.tensor(pack_te.y.astype(int), dtype=torch.long, device=device)

    probe = nn.Linear(Xfs.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        probe.train()
        loss = loss_fn(probe(Xfs), yfs)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(Xte).argmax(dim=1)
        acc = (pred == yte).float().mean().item()

    print(f"few-shot ({args.shots}-shot) test_acc: {acc:.4f}")


if __name__ == "__main__":
    main()
