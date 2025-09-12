import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import Optimizer, SGD
import torch.nn.functional as F
from typing import Tuple, List
import copy

from ._scheduling import ContinuousScheduleMixin


class Dino(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.04,
        momentum_teacher: float = 0.996,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        out_dim: int = 65536,
        hidden_dim: int = 2048,  # Added hidden_dim parameter
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        n_local_crops: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Create student and teacher networks
        self.model = model
        self.teacher = copy.deepcopy(model)

        # Get the actual output dimension from the model
        # Try to get it from different possible attributes
        # If we can't find the dimension, we'll need to do a forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            out = model(dummy_input)
            in_dim = out.shape[1]

        # Create projection heads for student and teacher
        self.student_head = self._build_projection_head(in_dim, hidden_dim, out_dim)
        self.teacher_head = self._build_projection_head(in_dim, hidden_dim, out_dim)

        # Disable gradient updates for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Copy student head weights to teacher head
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Create center for loss calculation
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Number of crops
        self.n_global_crops = 2
        self.n_local_crops = n_local_crops


    def _build_projection_head(
        self, in_dim: int, hidden_dim: int, out_dim: int
    ) -> nn.Module:
        """
        Builds a 3-layer projection head.
        Args:
            in_dim: Input dimension from the backbone
            hidden_dim: Hidden dimension of the projection head
            out_dim: Output dimension (number of dimensions in the learned representation)
        """
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @torch.no_grad()
    def _update_teacher(self):
        """Updates teacher model using momentum update."""
        m = self.hparams.momentum_teacher
        for param_student, param_teacher in zip(
            self.model.parameters(), self.teacher.parameters()
        ):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

        for param_student, param_teacher in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)

    def _get_teacher_output(self, global_views):
        """Get teacher output for global views only."""
        with torch.no_grad():
            teacher_output = []
            for view in global_views:
                feat = self.teacher(view)
                out = self.teacher_head(feat)
                # Center and normalize
                out = out - self.center
                out = F.normalize(out, dim=-1)
                # Apply softmax with temperature
                out = F.softmax(out / self.hparams.teacher_temp, dim=-1)
                teacher_output.append(out)
        return teacher_output

    def _get_student_output(self, views):
        """Get student output for all views."""
        student_output = []
        for view in views:
            feat = self.model(view)
            out = self.student_head(feat)
            # Normalize
            out = F.normalize(out, dim=-1)
            student_output.append(out)
        return student_output

    def _compute_dino_loss(self, student_output, teacher_output):
        """
        Compute DINO loss between student and teacher predictions.
        student_output: list of predictions for all views
        teacher_output: list of predictions for global views only
        """
        total_loss = 0
        n_loss_terms = 0

        # We only compute loss from teacher's global views
        for teacher_idx, teacher_out in enumerate(teacher_output):
            # But we use all student views (global + local) for prediction
            for student_idx, student_out in enumerate(student_output):
                # Skip when student and teacher are looking at the same global view
                if student_idx == teacher_idx:
                    continue

                # Compute cross entropy
                loss = torch.sum(
                    -teacher_out
                    * F.log_softmax(student_out / self.hparams.student_temp, dim=-1),
                    dim=-1,
                )
                total_loss += loss.mean()
                n_loss_terms += 1

        return total_loss / n_loss_terms

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """Update center used for teacher output."""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.hparams.center_momentum + batch_center * (
            1 - self.hparams.center_momentum
        )

    def training_step(self, batch, batch_idx):
        views, _ = batch  # Expect a list of crops (2 global + n local)

        # Split global and local crops
        global_views = views[: self.n_global_crops]
        all_views = views  # All views for student

        # Get teacher output (only for global views)
        teacher_output = self._get_teacher_output(global_views)

        # Update center
        self._update_center(teacher_output)

        # Get student output (for all views)
        student_output = self._get_student_output(all_views)

        # Compute loss
        loss = self._compute_dino_loss(student_output, teacher_output)

        # Update teacher
        self._update_teacher()

        # Log loss
        self.log("train_loss", loss, sync_dist=True)

        # Log average student output norm for monitoring collapse
        student_out_norm = torch.cat(
            [F.normalize(out, dim=-1) for out in student_output]
        )
        student_out_norm = torch.norm(student_out_norm, dim=1).mean()
        self.log("student_output_norm", student_out_norm, sync_dist=True)

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

        optimizer = SGD(
            param_groups,
            lr=self.hparams.lr,
            momentum=0.9,
        )

        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            start_factor=1e-4,
            base_lr=self.hparams.lr,
            eta_min=1e-6,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
