import lightning.pytorch as L
import os
import torch.distributed as dist
import torch
from torch import nn
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LRScheduler
import torch.nn.functional as F

from ._scheduling import ContinuousScheduleMixin

class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class SimCLR(ContinuousScheduleMixin, L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        temperature: float,
        weight_decay: float,
        max_epochs: int = 500,
        hidden_dim=128,
        use_temperature_schedule: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        assert (
            self.hparams.temperature > 0.0
        ), "The temperature must be a positive float!"

        # you need to do this on resnet but I havent decided how to nicely make that dynamic yet. Ill be using ViT for now
        self.model = model
        try:
            if self.model.fc is not None:
                self.model.fc = SimCLRProjectionHead(
                    in_dim=self.model.fc.in_features,
                    hidden_dim=2048,
                    out_dim=hidden_dim,
                )
        except:
            pass

        parserargs = kwargs.get("parserargs")
        teacher_name = getattr(parserargs, "external_teacher_model", None)
        self.external_teacher_weight = float(
            getattr(parserargs, "external_teacher_weight", 0.0)
        )
        self.external_teacher = None
        self.external_teacher_adapter = None
        if teacher_name and self.external_teacher_weight > 0:
            from transformers import AutoModel

            # Explicitly use the shared Hugging Face cache.  Rebuttal jobs run
            # offline on the cluster, so relying on the library default cache
            # can accidentally trigger a network lookup on compute nodes.
            self.external_teacher = AutoModel.from_pretrained(
                str(teacher_name),
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
            )
            self.external_teacher.eval()
            for parameter in self.external_teacher.parameters():
                parameter.requires_grad = False
            teacher_dim = int(self.external_teacher.config.hidden_size)
            self.external_teacher_adapter = nn.Linear(hidden_dim, teacher_dim)
        self.diffusion_teacher = None
        self.diffusion_teacher_adapter = None
        self.register_buffer("diffusion_teacher_cache", None, persistent=False)
        if bool(getattr(parserargs, "external_diffusion_teacher", False)):
            cache_path = getattr(parserargs, "external_diffusion_teacher_cache", None)
            if cache_path:
                cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
                self.diffusion_teacher_cache = cached["features"].float()
                teacher_dim = int(self.diffusion_teacher_cache.shape[1])
            else:
                from diffusers import AutoencoderKL

                self.diffusion_teacher = AutoencoderKL.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae"
                )
                self.diffusion_teacher.eval()
                for parameter in self.diffusion_teacher.parameters():
                    parameter.requires_grad = False
                teacher_dim = int(self.diffusion_teacher.config.latent_channels)
            self.diffusion_teacher_adapter = nn.Linear(hidden_dim, teacher_dim)

    def temperature(self) -> float:
        if not self.hparams.use_temperature_schedule:
            return self.hparams.temperature

        return self.cosine_anneal(
            self.hparams.temperature_min,
            self.hparams.temperature_max,
            self.hparams.t_max,
        )

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = self.cosine_warmup_scheduler(
            optimizer,
            warmup_epochs=10,
            max_epochs=self.hparams.max_epochs,
            min_lr_ratio=2e-6,
        )

        return [optimizer], [scheduler]

    def concat_all_gather(self, t: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all ranks so every process gets the full batch.
        Gradient flows only through the local slice.
        """
        if dist.is_available() and dist.is_initialized():
            # 1) collect detached copies from *every* rank
            tensors = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
            dist.all_gather(tensors, t.detach())          # no grad on remote parts
            # 2) replace *this* rank's slice with the *live* tensor (keeps grad)
            tensors[dist.get_rank()] = t
            t = torch.cat(tensors, dim=0)
        return t

    def info_nce_loss(self, batch, mode="train"):
        # --- split the two views -----------------------------------------------
        (x_i, x_j), batch_metadata = batch

        z_i = F.normalize(self.model(x_i), dim=-1)
        z_j = F.normalize(self.model(x_j), dim=-1)

        teacher_loss = None
        if self.external_teacher is not None:
            # SimCLR views use ImageNet normalization; DINOv3 uses its released
            # preprocessing statistics.  Align only the frozen teacher target.
            imagenet_mean = x_i.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            imagenet_std = x_i.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            dino_mean = x_i.new_tensor([0.430, 0.411, 0.296])[None, :, None, None]
            dino_std = x_i.new_tensor([0.213, 0.199, 0.183])[None, :, None, None]
            teacher_pixels = ((x_i * imagenet_std + imagenet_mean) - dino_mean) / dino_std
            with torch.no_grad():
                teacher_output = self.external_teacher(pixel_values=teacher_pixels)
                teacher_features = teacher_output.last_hidden_state[:, 0]
                teacher_features = F.normalize(teacher_features, dim=-1)
            student_features = F.normalize(self.external_teacher_adapter(z_i), dim=-1)
            teacher_loss = 1.0 - (student_features * teacher_features).sum(dim=-1).mean()
        if self.diffusion_teacher is not None or self.diffusion_teacher_cache is not None:
            if self.diffusion_teacher_cache is not None:
                if not isinstance(batch_metadata, (tuple, list)) or len(batch_metadata) < 2:
                    raise RuntimeError(
                        "cached diffusion teacher requires dataset indices from the collate function"
                    )
                indices = batch_metadata[1].to(z_i.device)
                valid = indices < len(self.diffusion_teacher_cache)
                valid = valid & torch.isfinite(
                    self.diffusion_teacher_cache[indices.clamp_max(len(self.diffusion_teacher_cache) - 1)]
                ).all(dim=1)
                if valid.any():
                    target = F.normalize(self.diffusion_teacher_cache[indices[valid]], dim=-1)
                    student = F.normalize(self.diffusion_teacher_adapter(z_i[valid]), dim=-1)
                    diffusion_loss = 1.0 - (student * target).sum(dim=-1).mean()
                    teacher_loss = diffusion_loss if teacher_loss is None else teacher_loss + diffusion_loss
            else:
                # SD3's VAE expects pixels in [-1, 1], while SSL batches use
                # ImageNet normalization.  This is an online fallback only;
                # rebuttal runs use the cached source-image teacher targets.
                imagenet_mean = x_i.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                imagenet_std = x_i.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                vae_pixels = (x_i * imagenet_std + imagenet_mean) * 2.0 - 1.0
                with torch.no_grad():
                    latent = self.diffusion_teacher.encode(vae_pixels).latent_dist.mean
                    target = F.normalize(latent.mean(dim=(2, 3)), dim=-1)
                student = F.normalize(self.diffusion_teacher_adapter(z_i), dim=-1)
                diffusion_loss = 1.0 - (student * target).sum(dim=-1).mean()
                teacher_loss = diffusion_loss if teacher_loss is None else teacher_loss + diffusion_loss

        # --- NEW: enlarge batch with features from every GPU -------------------
        z_i = self.concat_all_gather(z_i)
        z_j = self.concat_all_gather(z_j)

        feats = torch.cat([z_i, z_j], dim=0)          # 2 × B × world_size
        # -----------------------------------------------------------------------

        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # positive pairs lie exactly half-way across the concatenated tensor
        pos_mask = self_mask.roll(shifts=feats.shape[0] // 2, dims=0)

        temperature = self.temperature()
        self.log(f"{mode}_temperature", temperature, sync_dist=True)

        cos_sim = cos_sim / temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = nll.mean()
        if teacher_loss is not None:
            loss = loss + self.external_teacher_weight * teacher_loss
            self.log(f"{mode}_external_teacher_loss", teacher_loss, sync_dist=True)
        self.log(f"{mode}_loss", loss, sync_dist=True)

        # accuracy metrics (unchanged)
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],
                              cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        self.log(f"{mode}_acc_top1", (sim_argsort == 0).float().mean(), sync_dist=True)
        self.log(f"{mode}_acc_top5", (sim_argsort < 5).float().mean(), sync_dist=True)
        self.log(f"{mode}_acc_mean_pos", 1 + sim_argsort.float().mean(), sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="test")
