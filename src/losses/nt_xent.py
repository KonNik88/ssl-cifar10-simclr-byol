from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    NT-Xent / InfoNCE loss used in SimCLR.
    Args:
        z1, z2: [B, D] normalized embeddings (L2)
        temperature: scaling
    Returns:
        scalar loss
    """
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must be 2D tensors of shape [B, D].")
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}.")

    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # cosine similarities since z is normalized
    sim = torch.matmul(z, z.T)  # [2B, 2B]
    sim = sim / temperature

    # mask self-similarity
    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float("-inf"))

    # positives: (i, i+B) and (i+B, i)
    pos_idx = torch.arange(B, device=z.device)
    targets = torch.cat([pos_idx + B, pos_idx], dim=0)  # [2B]

    loss = F.cross_entropy(sim, targets)
    return loss
