"""DINOv2 (Oquab et al., 2024) on top of the DINO v1 implementation.

DINOv2 keeps the image-level self-distillation objective of DINO and adds
three things that are not optional parts of the recipe:

1. an iBOT patch-level objective, in which patches of the student's global
   crops are replaced by a learnable [MASK] token and the student predicts the
   teacher's distribution at those positions from the unmasked view,
2. Sinkhorn-Knopp normalization of the teacher distribution, taken from SwAV,
   in place of DINO's moving-average centering, and
3. the KoLeo regularizer, which spreads the batch's features by penalizing
   small nearest-neighbour distances.

The heads for the image-level and patch-level objectives are untied by
default, following the corresponding DINOv2 ablation.

What this does not reproduce is LVD-142M and the retrieval-based curation
pipeline that produced it.  DINOv2's published numbers are in large part a
result of that corpus, so a run of this objective on a small source set should
not be read as a reproduction of DINOv2 the model.  It is the DINOv2 training
objective, evaluated in the source regime this paper studies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .Dino import Dino, DinoHead


class DinoV2(Dino):
    def __init__(
        self,
        model: nn.Module,
        *args,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.1,
        ibot_out_dim: int = 65536,
        mask_ratio_min: float = 0.1,
        mask_ratio_max: float = 0.5,
        mask_probability: float = 0.5,
        sinkhorn_iterations: int = 3,
        untie_head_weights: bool = True,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.save_hyperparameters(
            ignore=["model"],
        )

        # The patch objective replaces masked positions with a learnable [MASK]
        # embedding.  The shared ViT backbone is built without one on purpose,
        # so that every other objective's state dict is unchanged; install it
        # here, on the student and the teacher, for the two models this class
        # owns.  Hugging Face's embedding layer uses ``self.mask_token``
        # whenever ``bool_masked_pos`` is passed, so adding the attribute is
        # sufficient.
        for net in (self.model, self.teacher):
            embeddings = getattr(getattr(net, "model", None), "embeddings", None)
            if embeddings is not None and getattr(embeddings, "mask_token", None) is None:
                token = nn.Parameter(torch.zeros(1, 1, net.num_features))
                nn.init.trunc_normal_(token, std=0.02)
                embeddings.mask_token = token
        for param in self.teacher.parameters():
            param.requires_grad = False

        in_dim = self.student_head.mlp[0].in_features
        hidden_dim = int(self.hparams.hidden_dim)
        bottleneck_dim = int(self.hparams.bottleneck_dim)

        if untie_head_weights:
            self.student_ibot_head = DinoHead(
                in_dim, hidden_dim, bottleneck_dim, ibot_out_dim
            )
            self.teacher_ibot_head = DinoHead(
                in_dim, hidden_dim, bottleneck_dim, ibot_out_dim
            )
            self.teacher_ibot_head.load_state_dict(
                self.student_ibot_head.state_dict()
            )
            for param in self.teacher_ibot_head.parameters():
                param.requires_grad = False
        else:
            # Sharing the modules keeps a single set of prototypes for both
            # objectives, which is the tied variant of the ablation.
            self.student_ibot_head = self.student_head
            self.teacher_ibot_head = self.teacher_head

    @staticmethod
    def _split_tokens(net, images, bool_masked_pos=None):
        """Return ``(cls, patches)`` in the same space as ``net(images)``.

        ``ViT.forward`` runs the encoder output through a projection head, so
        the CLS feature the image-level objective consumes is that projection
        and not the raw hidden state.  ``extract_tokens`` deliberately stops
        before it.  Applying the head here keeps both objectives in one space
        and keeps the CLS branch identical to what DINO v1 sees; without it the
        two differ by exactly that projection, which is a silent width
        mismatch when ``output_size`` and ``hidden_size`` disagree.
        """
        tokens = net.extract_tokens(images, bool_masked_pos=bool_masked_pos)
        projected = net.head(tokens)
        return projected[:, 0], projected[:, 1:]

    # ------------------------------------------------------------------
    # Teacher normalization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _sinkhorn_knopp(self, logits: torch.Tensor, temp: float) -> torch.Tensor:
        """Doubly-stochastic normalization of the teacher assignment.

        Replaces DINO's centering-and-sharpening.  Operating on the transpose
        makes the row constraint a constraint on prototype usage, which is what
        discourages collapse onto a few prototypes.
        """
        Q = torch.exp(logits.float() / temp).t()
        Q /= Q.sum().clamp(min=1e-12)
        K, B = Q.shape
        for _ in range(int(self.hparams.sinkhorn_iterations)):
            Q /= Q.sum(dim=1, keepdim=True).clamp(min=1e-12)
            Q /= K
            Q /= Q.sum(dim=0, keepdim=True).clamp(min=1e-12)
            Q /= B
        return (Q * B).t()

    # ------------------------------------------------------------------
    # KoLeo
    # ------------------------------------------------------------------
    def _koleo_loss(self, features: torch.Tensor) -> torch.Tensor:
        """Differential-entropy regularizer on nearest-neighbour distances.

        Encourages a locally uniform spread of the batch inside the unit
        sphere.  Distances are computed on L2-normalized features and the
        self-match is removed by masking the diagonal.
        """
        if features.shape[0] < 2:
            return features.new_zeros(())
        z = F.normalize(features.float(), dim=-1, p=2, eps=1e-8)
        dots = z @ z.t()
        dots.fill_diagonal_(-2.0)
        # max inner product -> min euclidean distance on the unit sphere
        nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * dots.max(dim=1).values, min=1e-8))
        return -torch.log(nn_dist + 1e-8).mean()

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _sample_masks(self, batch: int, num_patches: int, device) -> torch.Tensor:
        """Per-sample block of masked positions, empty for unmasked samples."""
        mask = torch.zeros(batch, num_patches, dtype=torch.bool, device=device)
        selected = torch.rand(batch, device=device) < float(
            self.hparams.mask_probability
        )
        if not bool(selected.any()):
            return mask
        lo = float(self.hparams.mask_ratio_min)
        hi = float(self.hparams.mask_ratio_max)
        ratios = torch.empty(batch, device=device).uniform_(lo, hi)
        noise = torch.rand(batch, num_patches, device=device)
        order = noise.argsort(dim=1)
        counts = (ratios * num_patches).long().clamp(min=1, max=num_patches - 1)
        ranks = order.argsort(dim=1)
        mask = ranks < counts.unsqueeze(1)
        mask = mask & selected.unsqueeze(1)
        return mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        views, _ = batch
        global_views = views[: self.n_global_crops]
        temp = self._teacher_temp()

        backbone = self.model
        teacher_backbone = self.teacher
        supports_tokens = hasattr(backbone, "extract_tokens")

        # ---- teacher: unmasked global crops -------------------------------
        teacher_cls_targets, teacher_patch_targets = [], []
        with torch.no_grad():
            for view in global_views:
                if supports_tokens:
                    cls_feat, patch_feat = self._split_tokens(teacher_backbone, view)
                else:
                    cls_feat, patch_feat = teacher_backbone(view), None
                teacher_cls_targets.append(
                    self._sinkhorn_knopp(self.teacher_head(cls_feat), temp)
                )
                if patch_feat is not None:
                    b, n, d = patch_feat.shape
                    logits = self.teacher_ibot_head(patch_feat.reshape(b * n, d))
                    teacher_patch_targets.append(
                        self._sinkhorn_knopp(logits, temp).reshape(b, n, -1)
                    )

        # ---- student: masked global crops, unmasked local crops -----------
        student_cls_outputs, ibot_terms = [], []
        cls_features_for_koleo = []
        for idx, view in enumerate(views):
            is_global = idx < self.n_global_crops
            if supports_tokens and is_global:
                num_patches = (view.shape[-1] // backbone.config.patch_size) * (
                    view.shape[-2] // backbone.config.patch_size
                )
                mask = self._sample_masks(view.shape[0], num_patches, view.device)
                cls_feat, patch_feat = self._split_tokens(
                    backbone, view, bool_masked_pos=mask
                )
                if mask.any() and teacher_patch_targets:
                    target = teacher_patch_targets[idx][mask]
                    pred = self.student_ibot_head(patch_feat[mask])
                    ibot_terms.append(
                        torch.sum(
                            -target
                            * F.log_softmax(pred / self.hparams.student_temp, dim=-1),
                            dim=-1,
                        ).mean()
                    )
            else:
                cls_feat = backbone(view)
            if is_global:
                cls_features_for_koleo.append(cls_feat)
            student_cls_outputs.append(self.student_head(cls_feat))

        dino_loss = self._compute_dino_loss(student_cls_outputs, teacher_cls_targets)
        ibot_loss = (
            torch.stack(ibot_terms).mean()
            if ibot_terms
            else dino_loss.new_zeros(())
        )
        koleo = (
            torch.stack([self._koleo_loss(f) for f in cls_features_for_koleo]).mean()
            if cls_features_for_koleo
            else dino_loss.new_zeros(())
        )

        loss = (
            dino_loss
            + float(self.hparams.ibot_loss_weight) * ibot_loss
            + float(self.hparams.koleo_loss_weight) * koleo
        )

        self.log("train_loss", loss, sync_dist=True)
        self.log("dino_loss", dino_loss, sync_dist=True)
        self.log("ibot_loss", ibot_loss, sync_dist=True)
        self.log("koleo_loss", koleo, sync_dist=True)
        self.log("teacher_temp", temp, sync_dist=True)
        return loss

    @torch.no_grad()
    def _update_teacher(self):
        super()._update_teacher()
        if self.teacher_ibot_head is self.teacher_head:
            return
        max_steps = max(1, int(self.trainer.estimated_stepping_batches))
        progress = min(1.0, float(self.global_step) / max_steps)
        import math

        base_m = float(self.hparams.momentum_teacher)
        m = 1.0 - (1.0 - base_m) * (math.cos(math.pi * progress) + 1.0) / 2.0
        for student_param, teacher_param in zip(
            self.student_ibot_head.parameters(),
            self.teacher_ibot_head.parameters(),
        ):
            teacher_param.data.mul_(m).add_((1 - m) * student_param.detach().data)

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        if self._current_epoch < int(self.hparams.freeze_last_layer_epochs):
            for param in self.student_ibot_head.last_layer.parameters():
                param.grad = None
