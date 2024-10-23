import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Tuple, List
import copy


class DINO(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.04,
        momentum_teacher: float = 0.996,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        out_dim: int = 65536,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Create student and teacher networks
        self.student = model
        self.teacher = copy.deepcopy(model)

        # Disable gradient updates for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Create projection heads for student and teacher
        self.student_head = self._build_projection_head(
            self.student.num_features, out_dim
        )
        self.teacher_head = self._build_projection_head(
            self.teacher.num_features, out_dim
        )

        # Copy student head weights to teacher head
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Create center for loss calculation
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Initialize the current temperature for teacher
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp

    def _build_projection_head(self, in_dim: int, out_dim: int) -> nn.Module:
        """Builds a 3-layer projection head."""
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    @torch.no_grad()
    def _update_teacher(self):
        """Updates teacher model using momentum update."""
        m = self.hparams.momentum_teacher
        for param_student, param_teacher in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

        for param_student, param_teacher in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

    def _compute_cross_entropy(self, student_output, teacher_output):
        """Compute cross entropy between student and teacher predictions."""
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(2)  # Split for global and local views

        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for iv, v in enumerate(student_out):
                if (
                    iv == iq
                ):  # Skip cases where student and teacher operate on same view
                    continue

                loss = torch.sum(-q * F.log_softmax(v, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)

        self.center = self.center * self.hparams.center_momentum + batch_center * (
            1 - self.hparams.center_momentum
        )

    def forward(self, x):
        """Forward pass through the student network."""
        feat = self.student(x)
        out = self.student_head(feat)
        return out

    def training_step(self, batch, batch_idx):
        views, _ = batch  # Expect multiple views of the images
        views = torch.cat(views, dim=0)

        # Get student output
        student_features = self.student(views)
        student_output = self.student_head(student_features)

        # Get teacher output
        with torch.no_grad():
            teacher_features = self.teacher(views)
            teacher_output = self.teacher_head(teacher_features)
            self._update_center(teacher_output)
            teacher_output = teacher_output - self.center

        # Compute loss
        loss = self._compute_cross_entropy(student_output, teacher_output)

        # Update teacher
        self._update_teacher()

        # Log loss
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Set up optimizer with weight decay
        param_groups = [
            {
                "params": [p for n, p in self.named_parameters() if "bias" not in n],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(param_groups, lr=self.hparams.lr)

        # Set up learning rate scheduler
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-4, total_iters=self.hparams.warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
            eta_min=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
