import copy
import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F

from experiment.models.SSLMethods.SimCLR import SimCLRProjectionHead
from ._scheduling import ContinuousScheduleMixin


class SDCLR(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float,
        weight_decay: float,
        max_epochs: int = 500,
        hidden_dim: int = 128,
        use_temperature_schedule: bool = False,
        sdclr_prune_rate: float = 0.3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"

        self.model = model
        try:
            if self.model.fc is not None:
                self.model.fc = SimCLRProjectionHead(
                    in_dim=self.model.fc.in_features,
                    hidden_dim=2048,
                    out_dim=hidden_dim,
                )
        except Exception:
            pass

        self.model_sparse = copy.deepcopy(self.model)
        for p in self.model_sparse.parameters():
            p.requires_grad = False

        self._apply_pruning(self.hparams.sdclr_prune_rate)

    # ------------------------------------------------------------------
    def _compute_threshold(self, model: nn.Module, prune_rate: float) -> float:
        weights = [p.data.abs().flatten() for p in model.parameters() if p.dim() > 1]
        if not weights:
            return 0.0
        all_weights = torch.cat(weights)
        k = int(prune_rate * all_weights.numel())
        if k < 1:
            return 0.0
        threshold = torch.kthvalue(all_weights, k).values.item()
        return threshold

    def _apply_pruning(self, prune_rate: float) -> None:
        threshold = self._compute_threshold(self.model, prune_rate)
        for p in self.model_sparse.parameters():
            if p.dim() > 1:
                mask = p.data.abs() >= threshold
                p.data.mul_(mask)

    def on_train_epoch_start(self) -> None:
        self.model_sparse.load_state_dict(self.model.state_dict())
        self._apply_pruning(self.hparams.sdclr_prune_rate)

    # ------------------------------------------------------------------
    def temperature(self) -> float:
        if not self.hparams.use_temperature_schedule:
            return self.hparams.temperature

        return self.cosine_anneal(
            self.hparams.temperature_min,
            self.hparams.temperature_max,
            self.hparams.t_max,
        )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
        )

        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.hparams.max_epochs,
            min_lr_ratio=2e-6,
        )
        return [optimizer], [scheduler]

    # ------------------------------------------------------------------
    def concat_all_gather(self, t: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            tensors = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors, t.detach())
            tensors[dist.get_rank()] = t
            t = torch.cat(tensors, dim=0)
        return t

    def info_nce_loss(self, batch, mode: str = "train"):
        (x_i, x_j), _ = batch
        z_i = F.normalize(self.model(x_i), dim=-1)
        z_j = F.normalize(self.model_sparse(x_j), dim=-1)

        z_i = self.concat_all_gather(z_i)
        z_j = self.concat_all_gather(z_j)
        feats = torch.cat([z_i, z_j], dim=0)

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        pos_mask = self_mask.roll(shifts=feats.shape[0] // 2, dims=0)

        temperature = self.temperature()
        self.log(f"{mode}_temperature", temperature, sync_dist=True)
        cos_sim = cos_sim / temperature

        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()
        self.log(f"{mode}_loss", loss, sync_dist=True)

        comb_sim = torch.cat([cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
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
