import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer, SGD
import torch.nn.functional as F
import copy
import random
from PIL import ImageFilter
from torchvision import transforms

from ._scheduling import ContinuousScheduleMixin


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur:
    """Gaussian blur augmentation used in MoCo v2"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def moco_transform(crop_size=224):
    """Create the augmentation transforms following MoCo v2"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return TwoCropsTransform(augmentation)


class MoCo(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float = 0.07,
        weight_decay: float = 1e-6,
        max_epochs: int = 500,
        momentum: float = 0.999,
        dim: int = 128,
        mlp: bool = True,
        queue_size: int = 65536,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Create encoder Q (online network)
        self.encoder_q = model

        # Create encoder K (momentum network)
        self.encoder_k = copy.deepcopy(model)

        # Get input dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            out = model(dummy_input)
            dim_mlp = out.shape[1]

        # Add projection head (as in MoCo v2)
        if mlp:
            # 3-layer MLP projection head for both networks
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim),
                nn.BatchNorm1d(dim),
            )

            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(dim_mlp, dim),
                nn.BatchNorm1d(dim),
            )
        else:
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim),
                nn.BatchNorm1d(dim),
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim),
                nn.BatchNorm1d(dim),
            )

        # Add prediction head (new in MoCo v3)
        self.predictor = nn.Sequential(
            nn.Linear(dim, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim),
            nn.BatchNorm1d(dim),
        )

        # Initialize momentum encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(dim, queue_size), dim=0),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.hparams.momentum + param_q.data * (
                1.0 - self.hparams.momentum
            )

    def contrastive_loss(self, q, k):
        """Compute InfoNCE loss with a dictionary of negatives."""
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Positive logits: each query against its key
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: queries against the queue
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.hparams.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def _concat_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors, tensor)
            tensor = torch.cat(tensors, dim=0)
        return tensor

    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue if using distributed training
        keys = self._concat_all_gather(keys)

        keys = F.normalize(keys, dim=1)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        queue_size = self.queue.shape[1]

        if batch_size > queue_size:
            keys = keys[:queue_size]
            batch_size = queue_size

        # replace the keys at ptr (dequeue and enqueue)
        end = ptr + batch_size
        if end <= queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first_part = queue_size - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, : batch_size - first_part] = keys[first_part:].T

        ptr = (ptr + batch_size) % queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """Forward computation during training"""
        # Compute query features
        q = self.encoder_q(im_q)
        q = self.predictor(q)

        # Compute key features with momentum encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)

        return q, k

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        im_q, im_k = imgs

        q, k = self(im_q, im_k)
        loss = self.contrastive_loss(q, k)

        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        # Log metrics
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Define optimizer
        optimizer = SGD(
            list(self.encoder_q.parameters()) + list(self.predictor.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
        )

        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.hparams.max_epochs,
        )

        return [optimizer], [scheduler]
