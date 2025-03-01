import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
import copy
import math
from torch.optim.lr_scheduler import LambdaLR
import random
from PIL import ImageFilter
from torchvision import transforms
import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
import copy
import math
from torch.optim.lr_scheduler import LambdaLR


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


class MoCo(L.LightningModule):
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
        """
        Contrastive loss function

        Args:
            q: queries (batch)
            k: keys (batch)
        """
        # Normalize features
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)

        # Compute logits
        # positive logits are the diagonal
        logits = torch.mm(q, k.T) / self.hparams.temperature

        # Targets: positive pairs are the diagonal elements
        labels = torch.arange(logits.shape[0], device=logits.device)

        return F.cross_entropy(logits, labels) * (2 * self.hparams.temperature)

    def forward(self, im_q, im_k):
        """Forward computation during training"""
        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.predictor(q)  # apply predictor to queries (MoCo v3)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC

        return q, k

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # imgs is a list containing im_q and im_k
        im_q, im_k = imgs

        # Forward pass for both directions (symmetrized loss - MoCo v3)
        q1, k2 = self(
            im_q, im_k
        )  # q1: queries from first view, k2: keys from second view
        q2, k1 = self(
            im_k, im_q
        )  # q2: queries from second view, k1: keys from first view

        # Compute loss for both directions
        loss_1 = self.contrastive_loss(q1, k2)
        loss_2 = self.contrastive_loss(q2, k1)

        # Total loss (symmetrized)
        loss = loss_1 + loss_2

        # Log metrics
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # Define optimizer
        adam_params = {
            "lr": self.hparams.lr,
            "betas": (0.9, 0.95),
        }

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                self.encoder_q.parameters(),
                **adam_params,
                weight_decay=self.hparams.weight_decay,
                adamw_mode=True,
            )
        else:
            optimizer = AdamW(
                self.encoder_q.parameters(),
                **adam_params,
                weight_decay=self.hparams.weight_decay,
            )

        # Warmup + cosine decay scheduler
        warmup_epochs = 10
        max_epochs = self.hparams.max_epochs
        base_lr = self.hparams.lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
                return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [scheduler]
