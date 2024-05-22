import lightning.pytorch as L
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
import torch.nn.functional as F
import copy
import numpy as np


from experiment.models.SSLMethods.masks.utils import apply_masks

from experiment.models.SSLMethods.utils.tensors import repeat_interleave_batch

from experiment.models.SSLMethods.helper import load_checkpoint, init_model, init_opt


class IJepa(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        args,
        iterations_per_epoch,
        max_epochs: int = 500,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.iterations_per_epoch = iterations_per_epoch

        self.encoder, self.predictor = init_model(
            device=self.device,
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            pred_depth=args.pred_depth,
            pred_emb_dim=args.pred_emb_dim,
            model=model,
        )

        

        self.target_encoder = copy.deepcopy(self.encoder)
        self.model = self.target_encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.args = args
        self.max_epochs = max_epochs

        self.momentum_scheduler = (
            self.args.ema[0]
            + i
            * (self.args.ema[1] - self.args.ema[0])
            / (self.iterations_per_epoch * self.max_epochs * self.args.ipe_scale)
            for i in range(
                int(self.iterations_per_epoch * self.max_epochs * self.args.ipe_scale)
                + 1
            )
        )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=self.encoder,
            predictor=self.predictor,
            wd=self.args.weight_decay,
            final_wd=self.args.final_weight_decay,
            start_lr=self.args.start_lr,
            ref_lr=self.args.lr,
            final_lr=self.args.final_lr,
            iterations_per_epoch=self.iterations_per_epoch,
            warmup=self.args.warmup,
            num_epochs=self.max_epochs,  # not entirely the same but fine maybe
            ipe_scale=self.args.ipe_scale,
            use_bfloat16=self.args.use_bfloat16,
        )

        self.scheduler = scheduler
        self.wd_scheduler = wd_scheduler

        return [optimizer]

    def jepa_loss(self, batch, mode):
        self.wd_scheduler.step()
        self.scheduler.step()

        # log the wd_scheduler and scheduler values
        self.log("wd", self.wd_scheduler.current_wd, prog_bar=True)
        self.log("lr", self.scheduler.current_lr, prog_bar=True)

        # Create the collate function and datamodule in such a way that this is true.
        udata, masks_enc, masks_pred = batch

        def load_imgs():
            # -- unsupervised imgs
            imgs = udata[0].to(self.device)
            masks_1 = [u.to(self.device) for u in masks_enc]
            masks_2 = [u.to(self.device) for u in masks_pred]
            return (imgs, masks_1, masks_2)

        imgs, masks_enc, masks_pred = load_imgs()

        def forward_target():
            with torch.no_grad():
                h = self.target_encoder(imgs)
                h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                B = len(h)
                # -- create targets (masked regions of h)
                h = apply_masks(h, masks_pred)
                h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                return h

        def forward_context():
            z = self.encoder(imgs, masks_enc)
            z = self.predictor(z, masks_enc, masks_pred)
            return z

        def loss_fn(z, h):
            loss = F.smooth_l1_loss(z, h)
            return loss

        # Step 1. Forward
        # idk if I should keep the casting
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16, enabled=self.args.use_bfloat16
        ):
            h = forward_target()
            z = forward_context()
            loss = loss_fn(z, h)

        # logging
        self.log(mode + "_loss", loss, prog_bar=True)

        assert not torch.isnan(loss), "loss is nan"

        return loss

    def training_step(self, batch, batch_idx):
        # log current learning rate
        return self.jepa_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.jepa_loss(batch, mode="val")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # step 2. backward and step
        optimizer.step(closure=optimizer_closure)

        # Step 3. momentum update of target encoder (this is nograd and thus shouldnt affect the backward)
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def on_save_checkpoint(self, checkpoint):
        # Save additional model state
        checkpoint['wd_scheduler_state'] = self.wd_scheduler.state_dict()
        checkpoint['scheduler_state'] = self.scheduler.state_dict()

    def on_load_checkpoint(self, checkpoint):
        # Load additional model state
        if 'wd_scheduler_state' in checkpoint:
            self.wd_scheduler.load_state_dict(checkpoint['wd_scheduler_state'])

        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
