from __future__ import annotations

import copy
from typing import Optional

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import ResNet18Encoder
from src.models.heads import ProjectionHead


def byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2.0 - 2.0 * (p * z).sum(dim=1).mean()


@torch.no_grad()
def ema_update(online: nn.Module, target: nn.Module, m: float) -> None:
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)


class BYOL(L.LightningModule):
    """
    Minimal BYOL implementation for STL-10.
    Expects dataloader to return (view1, view2) for SSL training.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        ema_m: float = 0.996,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 256,
        pred_hidden_dim: int = 512,
        use_bn_in_head: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.online_encoder = ResNet18Encoder()
        self.online_proj = ProjectionHead(
            in_dim=self.online_encoder.out_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            use_bn=use_bn_in_head,
        )
        self.online_pred = ProjectionHead(
            in_dim=proj_out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=proj_out_dim,
            use_bn=use_bn_in_head,
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_proj = copy.deepcopy(self.online_proj)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.online_encoder(x)
        z = self.online_proj(z)
        return z

    def training_step(self, batch, batch_idx: int):
        # expects (v1, v2)
        v1, v2 = batch
        o1 = self.online_pred(self.online_proj(self.online_encoder(v1)))
        o2 = self.online_pred(self.online_proj(self.online_encoder(v2)))

        with torch.no_grad():
            t1 = self.target_proj(self.target_encoder(v1))
            t2 = self.target_proj(self.target_encoder(v2))

        loss = byol_loss(o1, t2) + byol_loss(o2, t1)

        self.log("train_loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train_loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        # EMA update
        ema_update(self.online_encoder, self.target_encoder, self.hparams.ema_m)
        ema_update(self.online_proj, self.target_proj, self.hparams.ema_m)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
