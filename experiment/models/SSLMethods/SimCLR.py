import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import torch.nn.functional as F


class SimCLR(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float,
        weight_decay: float,
        max_epochs: int = 500,
        hidden_dim = 128,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"
        
        #you need to do this on resnet but I havent decided how to nicely make that dynamic yet. Ill be using ViT for now 
        #self.model = model
        #if self.model.fc is not None:
        #    self.model.fc = nn.Sequential(
        #        self.model.fc,  # Linear(ResNet output, 4*hidden_dim)
        #        nn.ReLU(inplace=True),
        #        nn.Linear(4*hidden_dim, hidden_dim)
        #    )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )

        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        feats = self.model(imgs) #yep thats the issue, why can a resnet deal with two concatenated images??

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll, prog_bar = True)

        # Get ranking position of positive example
        comb_sim = torch.cat(
            [
                cos_sim[pos_mask][:, None],
                cos_sim.masked_fill(pos_mask, -9e15),
            ],  # First position positive example
            dim=-1,
        )

        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar = True)
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), prog_bar = True)
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar = True)

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")
