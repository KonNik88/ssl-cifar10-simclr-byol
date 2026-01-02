from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch


@dataclass
class EmbeddingsPack:
    z: np.ndarray              # (N, D) embeddings
    y: np.ndarray              # (N,) labels (or -1 if unavailable)
    indices: np.ndarray        # (N,) dataset indices (if available)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def infer_encoder_from_simclr(simclr_model: torch.nn.Module) -> torch.nn.Module:
    """
    Tries to extract encoder/backbone from a SimCLR LightningModule/model.
    Supports common attribute names.
    """
    for name in ["encoder", "backbone", "online_encoder", "net"]:
        if hasattr(simclr_model, name):
            enc = getattr(simclr_model, name)
            if isinstance(enc, torch.nn.Module):
                return enc
    raise AttributeError(
        "Could not infer encoder from the loaded model. "
        "Expected attribute like: encoder/backbone/online_encoder/net."
    )


def load_simclr_from_ckpt(
    ckpt_path: Path,
    device: torch.device,
    model_cls=None,
    strict: bool = True,
) -> torch.nn.Module:
    """
    Loads Lightning checkpoint and returns the full SimCLR model.
    You can pass model_cls explicitly (recommended).
    If model_cls is None, we attempt torch.load and expect a plain state_dict won't work.
    """
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"

    if model_cls is None:
        raise ValueError(
            "model_cls is required for loading a Lightning checkpoint cleanly. "
            "Pass your SimCLR LightningModule class, e.g. from src.models.simclr import SimCLRLightning."
        )

    # Lightning-style: model_cls.load_from_checkpoint
    model = model_cls.load_from_checkpoint(str(ckpt_path), strict=strict, map_location=device)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_embeddings(
    dataloader: torch.utils.data.DataLoader,
    encoder: torch.nn.Module,
    device: torch.device,
    normalize: bool = True,
    max_batches: Optional[int] = None,
) -> EmbeddingsPack:
    """
    Computes embeddings for a labeled STL-10 split.
    Assumes batch is (x, y) or (x, y, idx). If idx not present -> indices=-1.
    """
    encoder.eval()

    zs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    idxs: List[np.ndarray] = []

    for bi, batch in enumerate(dataloader):
        if max_batches is not None and bi >= max_batches:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
            idx = None
        elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
            x, y, idx = batch[0], batch[1], batch[2]
        else:
            raise ValueError("Unexpected batch format. Expected (x,y) or (x,y,idx).")

        x = x.to(device, non_blocking=True)

        z = encoder(x)
        if isinstance(z, (list, tuple)):
            z = z[0]

        if normalize:
            z = torch.nn.functional.normalize(z, dim=1)

        zs.append(_to_numpy(z))
        ys.append(_to_numpy(y))
        if idx is None:
            idxs.append(np.full((len(y),), -1, dtype=np.int64))
        else:
            idxs.append(_to_numpy(idx).astype(np.int64))

    Z = np.concatenate(zs, axis=0) if zs else np.empty((0, 0), dtype=np.float32)
    Y = np.concatenate(ys, axis=0) if ys else np.empty((0,), dtype=np.int64)
    I = np.concatenate(idxs, axis=0) if idxs else np.empty((0,), dtype=np.int64)

    return EmbeddingsPack(z=Z, y=Y, indices=I)


def save_embeddings_npz(path: Path, pack: EmbeddingsPack, meta: Optional[Dict[str, Any]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = {}
    # np.savez stores arrays; metadata stored as stringified dict for simplicity
    np.savez(path, z=pack.z, y=pack.y, indices=pack.indices, meta=str(meta))


def load_embeddings_npz(path: Path) -> Tuple[EmbeddingsPack, Dict[str, Any]]:
    path = Path(path)
    assert path.exists(), f"Embeddings file not found: {path}"
    data = np.load(path, allow_pickle=True)
    meta_raw = data.get("meta", None)
    meta: Dict[str, Any] = {}
    if meta_raw is not None:
        try:
            meta = eval(str(meta_raw))
        except Exception:
            meta = {"meta": str(meta_raw)}
    pack = EmbeddingsPack(
        z=data["z"],
        y=data["y"],
        indices=data["indices"],
    )
    return pack, meta
