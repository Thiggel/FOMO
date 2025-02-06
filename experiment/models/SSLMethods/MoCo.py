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
        K: int = 65536,
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

        # Add MLP projection head if specified (as in MoCo v2)
        if mlp:
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
            )
        else:
            self.encoder_q.fc = nn.Linear(dim_mlp, dim)
            self.encoder_k.fc = nn.Linear(dim_mlp, dim)

        # Initialize momentum encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.K  # Move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """Forward computation during training"""
        # Compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # Calculate logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def training_step(self, batch, batch_idx):
        imgs, _ = batch  # imgs is a list containing im_q and im_k
        im_q, im_k = imgs

        # Forward pass
        output, target = self(im_q, im_k)

        # InfoNCE loss
        loss = F.cross_entropy(output, target)

        # Log metrics
        self.log("train_loss", loss, sync_dist=True)
        acc1, acc5 = self._accuracy(output, target, topk=(1, 5))
        self.log("train_acc1", acc1, sync_dist=True)
        self.log("train_acc5", acc5, sync_dist=True)

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

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
