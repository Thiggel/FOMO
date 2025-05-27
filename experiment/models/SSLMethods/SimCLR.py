import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LambdaLR
import torch.nn.functional as F
import math


class SimCLR(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float,
        weight_decay: float,
        max_epochs: int = 500,
        hidden_dim=128,
        use_temperature_schedule: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"

        # you need to do this on resnet but I havent decided how to nicely make that dynamic yet. Ill be using ViT for now
        self.model = model
        try:
            if self.model.fc is not None:
                self.model.fc = nn.Sequential(
                    self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
                    nn.ReLU(inplace=True),
                    nn.Linear(4 * hidden_dim, hidden_dim),
                )
        except:
            pass

    @property
    def temperature(self) -> float:
        if not self.hparams.use_temperature_schedule:
            return self.hparams.temperature

        t_max = 1.0
        t_min = 0.1

        temperature = (t_max - t_min) * (
            1 + math.cos(2 * math.pi * self.current_epoch / self.hparams.max_epochs)
        ) / 2 + t_min

        print(
            f"Current epoch: {self.current_epoch}, current temperature: {temperature:.4f}"
        )

        return temperature

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        adam_params = {
            "lr": self.hparams.lr,
            "betas": (0.9, 0.95),
        }

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                self.parameters(), **adam_params, adamw_mode=True
            )
        else:
            optimizer = AdamW(self.parameters(), **adam_params)

        # Define the number of warmup epochs
        warmup_epochs = 10
        max_epochs = self.hparams.max_epochs
        base_lr = self.hparams.lr

        # Combined linear warmup and cosine annealing function
        def lr_lambda(epoch):
            warmup_epochs = 10
            total_epochs = 500
            min_lr_ratio = 0.01  # e.g., final LR is 1% of base_lr
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                    1 + math.cos(math.pi * progress)
                )

        # Single scheduler with combined behavior
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        feats = self.model(imgs)

        feats = F.normalize(feats, dim=-1)

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll, sync_dist=True)

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],
                cos_sim.masked_fill(pos_mask, -9e15),
            ],  # First position positive example
            dim=-1,
        )

        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log(
            mode + "_acc_top1",
            (sim_argsort == 0).float().mean(),
            sync_dist=True,
        )
        self.log(
            mode + "_acc_top5",
            (sim_argsort < 5).float().mean(),
            sync_dist=True,
        )
        self.log(
            mode + "_acc_mean_pos",
            1 + sim_argsort.float().mean(),
            sync_dist=True,
        )

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="test")
