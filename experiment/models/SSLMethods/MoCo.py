"""MoCo v3 training objective.

This implementation follows the symmetric momentum-encoder formulation from
the official MoCo v3 code. In particular, it does not mix the MoCo v2 queue
with the v3 predictor, as the previous implementation did.
"""

import copy
import math
import random
from PIL import ImageFilter

import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from ._scheduling import ContinuousScheduleMixin


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, image):
        return [self.base_transform(image), self.base_transform(image)]


class GaussianBlur:
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image):
        return image.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma))
        )


def moco_transform(crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return TwoCropsTransform(augmentation)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Module:
    modules = []
    for layer_idx in range(layers):
        in_dim = input_dim if layer_idx == 0 else hidden_dim
        out_dim = output_dim if layer_idx == layers - 1 else hidden_dim
        modules.append(nn.Linear(in_dim, out_dim, bias=False))
        modules.append(nn.BatchNorm1d(out_dim))
        if layer_idx < layers - 1:
            modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class MoCo(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float = 1.0,
        weight_decay: float = 0.1,
        max_epochs: int = 500,
        momentum: float = 0.99,
        dim: int = 256,
        mlp_dim: int = 4096,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.momentum_model = copy.deepcopy(model)

        with torch.no_grad():
            was_training = model.training
            model.eval()
            input_size = int(getattr(kwargs.get("parserargs"), "crop_size", 224))
            output_dim = int(
                model(torch.zeros(2, 3, input_size, input_size)).shape[-1]
            )
            model.train(was_training)

        self.projector = _mlp(output_dim, mlp_dim, dim, layers=3)
        self.momentum_projector = copy.deepcopy(self.projector)
        self.predictor = _mlp(dim, mlp_dim, dim, layers=2)

        for online, target in zip(
            list(self.model.parameters()) + list(self.projector.parameters()),
            list(self.momentum_model.parameters())
            + list(self.momentum_projector.parameters()),
        ):
            target.data.copy_(online.data)
            target.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        max_steps = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, float(self.global_step) / max_steps)
        base_m = float(self.hparams.momentum)
        momentum = 1.0 - (1.0 - base_m) * (
            math.cos(math.pi * progress) + 1.0
        ) / 2.0
        for online, target in zip(
            list(self.model.parameters()) + list(self.projector.parameters()),
            list(self.momentum_model.parameters())
            + list(self.momentum_projector.parameters()),
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _contrastive_loss(self, query: torch.Tensor, key: torch.Tensor):
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        gathered_key = self._gather(key)
        logits = query @ gathered_key.T / float(self.hparams.temperature)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        labels = (
            torch.arange(query.shape[0], device=query.device)
            + rank * query.shape[0]
        )
        return F.cross_entropy(logits, labels) * (2.0 * self.hparams.temperature)

    def training_step(self, batch, batch_idx):
        (view1, view2), _ = batch
        query1 = self.predictor(self.projector(self.model(view1)))
        query2 = self.predictor(self.projector(self.model(view2)))

        with torch.no_grad():
            self._momentum_update()
            key1 = self.momentum_projector(self.momentum_model(view1))
            key2 = self.momentum_projector(self.momentum_model(view2))

        loss = self._contrastive_loss(query1, key2)
        loss = loss + self._contrastive_loss(query2, key1)
        self.log("train_loss", loss, sync_dist=True)
        self.log(
            "feature_std",
            F.normalize(query1.detach(), dim=-1).std(dim=0).mean(),
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        parserargs = self.hparams.get("parserargs")
        model_name = (
            str(parserargs.model.model_name).lower()
            if parserargs is not None
            else ""
        )
        parameters = (
            list(self.model.parameters())
            + list(self.projector.parameters())
            + list(self.predictor.parameters())
        )
        if "vit" in model_name:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.SGD(
                parameters,
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.hparams.max_epochs,
        )
        return [optimizer], [scheduler]
