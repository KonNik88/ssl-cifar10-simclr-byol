from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from src.losses.nt_xent import nt_xent_loss
from src.models.backbone import ResNet18Encoder
from src.models.heads import MLPHead, l2_normalize


class SimCLR(L.LightningModule):
    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        temperature: float = 0.2,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 128,
        use_bn_in_head: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = ResNet18Encoder()
        self.projector = MLPHead(
            in_dim=self.encoder.feat_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            use_bn=use_bn_in_head,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projector(h)
        z = l2_normalize(z)
        return z

    def training_step(self, batch, batch_idx):
        (x1, x2), _ = batch  # unlabeled: y is -1
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        loss = nt_xent_loss(z1, z2, temperature=self.hparams.temperature)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt
