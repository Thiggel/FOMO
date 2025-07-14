import lightning.pytorch as L
import torch.distributed as dist
import torch
from torch import nn
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, LambdaLR
import torch.nn.functional as F
from flash.core.optimizers import LARS
import math

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


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

        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"

        # you need to do this on resnet but I havent decided how to nicely make that dynamic yet. Ill be using ViT for now
        self.model = model
        try:
            if self.model.fc is not None:
                self.model.fc = SimCLRProjectionHead(
                    in_dim=self.model.fc.in_features,
                    hidden_dim=2048,
                    out_dim=hidden_dim,
                )
        except:
            pass

    @property
    def temperature(self) -> float:
        if not self.hparams.use_temperature_schedule:
            return self.hparams.temperature

        t_min = self.hparams.temperature_min
        t_max = self.hparams.temperature_max
        T = self.hparams.t_max

        temperature = (t_max - t_min) * (
            1 + math.cos(math.pi * self.current_epoch / T)
        ) / 2 + t_min

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
            optimizer = SGD(self.parameters(), lr=0.5, weight_decay=1e-4, momentum=0.9)

        # Define the number of warmup epochs
        warmup_epochs = 10
        max_epochs = self.hparams.max_epochs
        base_lr = self.hparams.lr

        # Combined linear warmup and cosine annealing function
        def lr_lambda(epoch):
            warmup_epochs = 10
            total_epochs = 800
            min_lr_ratio = 2 * 1e-6  # get 1e-6 at the end of training
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

    def concat_all_gather(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all ranks so every process gets the full batch.
        Gradient flows only through the local slice.
        """
        if dist.is_available() and dist.is_initialized():
            # 1) collect detached copies from *every* rank
            tensors = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors, t.detach())          # no grad on remote parts
            # 2) replace *this* rank's slice with the *live* tensor (keeps grad)
            tensors[dist.get_rank()] = t
            t = torch.cat(tensors, dim=0)
        return t

    def info_nce_loss(self, batch, mode="train"):
        # --- split the two views -----------------------------------------------
        (x_i, x_j), _ = batch

        z_i = F.normalize(self.model(x_i), dim=-1)
        z_j = F.normalize(self.model(x_j), dim=-1)

        # --- NEW: enlarge batch with features from every GPU -------------------
        z_i = self.concat_all_gather(z_i)
        z_j = self.concat_all_gather(z_j)

        feats = torch.cat([z_i, z_j], dim=0)          # 2 × B × world_size
        # -----------------------------------------------------------------------

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # positive pairs lie exactly half-way across the concatenated tensor
        pos_mask = self_mask.roll(shifts=feats.shape[0] // 2, dims=0)

        temperature = self.temperature
        self.log(f"{mode}_temperature", temperature, sync_dist=True)

        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()
        self.log(f"{mode}_loss", loss, sync_dist=True)

        # accuracy metrics (unchanged)
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],
                              cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        self.log(f"{mode}_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        self.log(f"{mode}_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        self.log(f"{mode}_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="test")
