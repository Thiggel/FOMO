"""Masked autoencoder objective for the repository's Hugging Face ViT backbone.

This follows the core MAE protocol: mask a large random subset of image
patches, encode the visible context, and regress normalized pixels only at
masked locations.  The lightweight linear decoder is intentional for the
rebuttal screen; downstream evaluation always uses the encoder features.
"""

import lightning.pytorch as L
import torch
from torch import nn

from ._scheduling import ContinuousScheduleMixin


class MAE(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1.5e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 100,
        mask_ratio: float = 0.75,
        warmup_epochs: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(model, "model") or not hasattr(model, "config"):
            raise ValueError("MAE requires one of FOMO's ViT backbones")
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        if self.model.model.embeddings.mask_token is None:
            hidden_size = int(model.config.hidden_size)
            self.model.model.embeddings.mask_token = nn.Parameter(
                torch.zeros(1, 1, hidden_size)
            )
            nn.init.normal_(
                self.model.model.embeddings.mask_token, mean=0.0, std=0.02
            )
        patch_size = int(model.config.patch_size)
        channels = int(model.config.num_channels)
        self.patch_size = patch_size
        self.decoder = nn.Linear(
            int(model.config.hidden_size), patch_size * patch_size * channels
        )

    def _patchify(self, images):
        p = self.patch_size
        batch, channels, height, width = images.shape
        if height % p or width % p:
            raise ValueError("Input dimensions must be divisible by the patch size")
        patches = images.reshape(
            batch, channels, height // p, p, width // p, p
        )
        patches = torch.einsum("nchpwq->nhwpcq", patches)
        patches = patches.reshape(batch, -1, p * p * channels)
        mean = patches.mean(dim=-1, keepdim=True)
        var = patches.var(dim=-1, keepdim=True, unbiased=False)
        return (patches - mean) / torch.sqrt(var + 1e-6)

    def _loss(self, batch, stage):
        images, _ = batch
        targets = self._patchify(images)
        batch_size, num_patches, _ = targets.shape
        mask = torch.rand(batch_size, num_patches, device=images.device)
        mask = mask < float(self.hparams.mask_ratio)

        # HF ViT uses its learned mask token at positions marked true.
        outputs = self.model.model(
            pixel_values=images, bool_masked_pos=mask
        ).last_hidden_state[:, 1:]
        predictions = self.decoder(outputs)
        per_patch = (predictions - targets).square().mean(dim=-1)
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(f"{stage}_mask_ratio", mask.float().mean(), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            start_factor=1e-4,
            base_lr=self.hparams.lr,
            eta_min=1e-6,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
