import lightning.pytorch as L
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer, SGD
import torch.nn.functional as F
from typing import Tuple, List
import copy
import math

from ._scheduling import ContinuousScheduleMixin


class DinoHead(nn.Module):
    """Reference DINO projection head (Caron et al., 2021).

    The MLP narrows to a low-dimensional bottleneck, the bottleneck is L2
    normalized, and the prototype layer is weight normalized with a fixed unit
    magnitude.  That normalization pair is what keeps the 65536-way softmax from
    collapsing; an unnormalized ``Linear`` straight to ``out_dim`` trains far
    less stably, which is what this head previously did.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        # DINO keeps the prototype magnitudes fixed at one.
        self.last_layer.weight_g.requires_grad = False

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class Dino(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-4,
        weight_decay: float = 0.04,
        weight_decay_end: float = 0.4,
        momentum_teacher: float = 0.996,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_teacher_temp_epochs: int = 30,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
        clip_grad: float = 3.0,
        freeze_last_layer_epochs: int = 1,
        n_local_crops: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        # Create student and teacher networks
        self.model = model
        self.teacher = copy.deepcopy(model)

        # Get the actual output dimension from the model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            out = model(dummy_input)
            in_dim = out.shape[1]

        # Create projection heads for student and teacher
        self.student_head = DinoHead(in_dim, hidden_dim, bottleneck_dim, out_dim)
        self.teacher_head = DinoHead(in_dim, hidden_dim, bottleneck_dim, out_dim)

        # Disable gradient updates for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        # Copy student head weights to teacher head
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        # Create center for loss calculation
        self.register_buffer("center", torch.zeros(1, out_dim))

        # Number of crops
        self.n_global_crops = 2
        self.n_local_crops = n_local_crops

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------
    @property
    def _current_epoch(self) -> int:
        """Epoch index that keeps counting across BRIDGE's per-cycle trainers."""
        return int(self.total_epochs_completed)

    def _teacher_temp(self) -> float:
        """Linear warmup from ``warmup_teacher_temp`` to ``teacher_temp``.

        A cold teacher early on is what stops the student from chasing a sharp,
        arbitrary target before the prototypes mean anything.
        """
        warm_epochs = max(1, int(self.hparams.warmup_teacher_temp_epochs))
        if self._current_epoch >= warm_epochs:
            return float(self.hparams.teacher_temp)
        progress = self._current_epoch / warm_epochs
        start = float(self.hparams.warmup_teacher_temp)
        end = float(self.hparams.teacher_temp)
        return start + (end - start) * progress

    def _weight_decay(self) -> float:
        """Cosine ramp from ``weight_decay`` to ``weight_decay_end``."""
        total = max(1, int(self.hparams.max_epochs))
        progress = min(1.0, self._current_epoch / total)
        start = float(self.hparams.weight_decay)
        end = float(self.hparams.weight_decay_end)
        return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2

    @torch.no_grad()
    def _update_teacher(self):
        """Updates teacher model using momentum update."""
        max_steps = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, float(self.global_step) / max_steps)
        base_m = float(self.hparams.momentum_teacher)
        m = 1.0 - (1.0 - base_m) * (
            math.cos(math.pi * progress) + 1.0
        ) / 2.0
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
        temp = self._teacher_temp()
        with torch.no_grad():
            teacher_probs = []
            teacher_logits = []
            for view in global_views:
                feat = self.teacher(view)
                logits = self.teacher_head(feat)
                teacher_logits.append(logits)
                centered = (logits - self.center) / temp
                probs = F.softmax(centered, dim=-1)
                teacher_probs.append(probs)
        return teacher_probs, teacher_logits

    def _get_student_output(self, views):
        """Get student output for all views.

        Global and local crops have different spatial sizes, so they are run as
        two batched groups rather than one concatenated tensor.
        """
        student_output = []
        for view in views:
            feat = self.model(view)
            out = self.student_head(feat)
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
        """Update center used for teacher output.

        The mean is reduced across ranks so every replica keeps an identical
        center; a rank-local center makes the centering term depend on how the
        batch happened to be sharded.
        """
        stacked = torch.cat(teacher_output)
        batch_sum = stacked.sum(dim=0, keepdim=True)
        count = torch.tensor(
            [stacked.shape[0]], dtype=batch_sum.dtype, device=batch_sum.device
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_sum)
            dist.all_reduce(count)
        batch_center = batch_sum / count.clamp(min=1)
        self.center = self.center * self.hparams.center_momentum + batch_center * (
            1 - self.hparams.center_momentum
        )

    def training_step(self, batch, batch_idx):
        views, _ = batch  # Expect a list of crops (2 global + n local)

        # Split global and local crops
        global_views = views[: self.n_global_crops]
        all_views = views  # All views for student

        # Get teacher output (only for global views)
        teacher_output, teacher_logits = self._get_teacher_output(global_views)

        # Update center with logits before softmax
        self._update_center(teacher_logits)

        # Get student output (for all views)
        student_output = self._get_student_output(all_views)

        # Compute loss
        loss = self._compute_dino_loss(student_output, teacher_output)

        # Log loss
        self.log("train_loss", loss, sync_dist=True)
        self.log("teacher_temp", self._teacher_temp(), sync_dist=True)

        # Log average student output norm for monitoring collapse
        student_out_norm = torch.cat(
            [F.normalize(out, dim=-1) for out in student_output]
        )
        student_out_norm = torch.norm(student_out_norm, dim=1).mean()
        self.log("student_output_norm", student_out_norm, sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Clip gradients, freeze the prototype layer, and step the decay schedule.

        DINO cancels the prototype-layer gradient for the first epochs because
        those weights otherwise race ahead of a still-random backbone.
        """
        clip = float(self.hparams.clip_grad or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.grad is not None], clip
            )
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.student_head.parameters() if p.grad is not None],
                clip,
            )

        if self._current_epoch < int(self.hparams.freeze_last_layer_epochs):
            for param in self.student_head.last_layer.parameters():
                param.grad = None

        # Weight decay follows its own cosine schedule, independent of the lr.
        decay = self._weight_decay()
        for group in optimizer.param_groups:
            if group.get("weight_decay", 0.0) > 0 or group.get("apply_wd", False):
                group["weight_decay"] = decay

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # The teacher trails the *updated* student, so the momentum update runs
        # after the optimizer step rather than before it.
        self._update_teacher()

    def configure_optimizers(self):
        # Only the student is optimized; the teacher is momentum-updated.
        # DINO exempts every 1-d parameter (biases and norm weights) from decay,
        # not just parameters whose name contains "bias".
        trainable = [
            (n, p)
            for n, p in self.named_parameters()
            if p.requires_grad and not n.startswith(("teacher.", "teacher_head."))
        ]
        decay_params = [p for n, p in trainable if p.ndim > 1]
        no_decay_params = [p for n, p in trainable if p.ndim <= 1]

        param_groups = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.weight_decay,
                "apply_wd": True,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
                "apply_wd": False,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.hparams.lr,
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
