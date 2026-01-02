from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)


class MLPHead(nn.Module):
    """
    Projector head: in_dim -> hidden_dim -> out_dim
    SimCLR commonly uses BN in the MLP. We'll make it configurable.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, out_dim, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
