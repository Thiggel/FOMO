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

from experiment.models.SSLMethods.helper import (
    load_checkpoint,
    init_model,
    init_opt)


class IJepa(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        weight_decay: float,
        args,
        ipe,
        max_epochs: int = 500,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])
        self.ipe = ipe

        self.encoder, self.predictor = init_model(
            device=self.device,
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            pred_depth=args.pred_depth,
            pred_emb_dim=args.pred_emb_dim,
            model=model)
        
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        self.args = args
        self.max_epochs = max_epochs

        _, _, _, self.wd_scheduler = init_opt(
            encoder=self.encoder,
            predictor=self.predictor,
            wd=self.args.weight_decay,
            final_wd= self.args.final_weight_decay,
            start_lr= self.args.start_lr,
            ref_lr= self.args.lr,
            final_lr= self.args.final_lr,
            iterations_per_epoch= self.ipe,
            warmup=self.args.warmup,
            num_epochs=self.max_epochs,
            ipe_scale=self.args.ipe_scale,
            use_bfloat16=self.args.use_bfloat16
        )

        self.momentum_scheduler = (self.args.ema[0] + i*(self.args.ema[1]-self.args.ema[0])/(self.ipe*self.max_epochs*self.args.ipe_scale)
                          for i in range(int(self.ipe*self.max_epochs*self.args.ipe_scale)+1))
        


    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer, scaler, scheduler, wd_scheduler = init_opt(
            encoder=self.encoder,
            predictor=self.predictor,
            wd=self.args.weight_decay,
            final_wd= self.args.final_weight_decay,
            start_lr= self.args.start_lr,
            ref_lr= self.args.lr,
            final_lr= self.args.final_lr,
            iterations_per_epoch= self.ipe,
            warmup=self.args.warmup,
            num_epochs= self.max_epochs, #not entirely the same but fine maybe
            ipe_scale=self.args.ipe_scale,
            use_bfloat16=self.args.use_bfloat16
        )

        return [optimizer], [scheduler]
    
    def jepa_loss(self, batch, mode):
        _new_wd = self.wd_scheduler.step()

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
        #idk if I should keep the casting
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.args.use_bfloat16):
            h = forward_target()
            z = forward_context()
            loss = loss_fn(z, h)
        
        #logging
        self.log(mode + "_loss", loss, prog_bar=True)

        assert not torch.isnan(loss), 'loss is nan'

        return loss

    def training_step(self, batch, batch_idx):
        return self.jepa_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self.jepa_loss(batch, mode="val")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        #step 2. backward and step 
        optimizer.step(closure=optimizer_closure)

        # Step 3. momentum update of target encoder (this is nograd and thus shouldnt affect the backward)
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
    