import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import torch.nn.functional as F


class Supervised(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        max_epochs: int = 500,
        hidden_dim=128,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        # you need to do this on resnet but I havent decided how to nicely make that dynamic yet. Ill be using ViT for now
        self.model = model
        try:
            if self.model.fc is not None:
                self.model.fc = nn.Sequential(
                    self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
                    nn.ReLU(inplace=True),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                )
        except Exception:
            pass

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=1e-3, betas=(0.9, 0.95))
        else:
            optimizer = AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95))

        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )

        return [optimizer], [lr_scheduler]

    def step(self, batch, mode="train"):
        imgs, labels = batch

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        self.log(f"{mode}_loss", loss, prog_bar=True)

        if mode != "train":
            preds = torch.argmax(logits, dim=1)
            acc = (preds == labels).float().mean()
            self.log(f"{mode}_acc", acc, prog_bar=True)

        return loss

    def training_step(self, batch, _):
        loss = self.step(batch, mode="train")
        return loss

    def validation_step(self, batch, _):
        loss = self.step(batch, mode="val")
        return loss

    def test_step(self, batch, _):
        loss = self.step(batch, mode="test")
        return loss
