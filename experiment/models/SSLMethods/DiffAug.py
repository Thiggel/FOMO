"""Embedding-space DiffAug-style SSL baseline.

The public DiffAug release targets non-image modalities, so this is an
explicitly labeled image-SSL adaptation rather than a claim of running its
authors' visual recipe.  It jointly trains a conditional diffusion denoiser in
the SSL projection space and uses its reconstructed sample as an additional
positive for SimCLR.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .SimCLR import SimCLR


class DiffAug(SimCLR):
    def __init__(self, *args, parserargs=None, **kwargs):
        super().__init__(*args, parserargs=parserargs, **kwargs)
        dim = int(getattr(self.model, "fc", None).out_dim) if hasattr(getattr(self.model, "fc", None), "out_dim") else 128
        self.diffaug_steps = int(getattr(parserargs, "diffaug_steps", 32))
        self.diffaug_weight = float(getattr(parserargs, "diffaug_weight", 0.2))
        self.diffaug_denoiser = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim * 2), nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def info_nce_loss(self, batch, mode="train"):
        (x_i, x_j), _ = batch
        z_i = F.normalize(self.model(x_i), dim=-1)
        z_j = F.normalize(self.model(x_j), dim=-1)
        batch_size, dim = z_i.shape
        timestep = torch.randint(1, self.diffaug_steps + 1, (batch_size, 1), device=z_i.device)
        alpha = 1.0 - .02 * timestep.float() / self.diffaug_steps
        noise = torch.randn_like(z_j)
        noisy = alpha.sqrt() * z_j.detach() + (1 - alpha).sqrt() * noise
        predicted_noise = self.diffaug_denoiser(torch.cat([noisy, z_i.detach(), timestep.float() / self.diffaug_steps], dim=1))
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        generated_positive = F.normalize(
            (noisy - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt().clamp_min(1e-6), dim=-1
        )
        z_i_all = self.concat_all_gather(z_i)
        z_j_all = self.concat_all_gather(z_j)
        features = torch.cat([z_i_all, z_j_all], dim=0)
        similarity = F.cosine_similarity(features[:, None], features[None, :], dim=-1)
        mask = torch.eye(len(similarity), dtype=torch.bool, device=similarity.device)
        similarity.masked_fill_(mask, -9e15)
        positives = mask.roll(shifts=len(similarity) // 2, dims=0)
        logits = similarity / self.temperature()
        contrastive_loss = (-logits[positives] + torch.logsumexp(logits, dim=-1)).mean()
        generated_positive_loss = 1.0 - (z_i * generated_positive).sum(dim=-1).mean()
        loss = contrastive_loss + self.diffaug_weight * (diffusion_loss + generated_positive_loss)
        self.log(f"{mode}_loss", loss, sync_dist=True)
        self.log(f"{mode}_diffaug_diffusion_loss", diffusion_loss, sync_dist=True)
        self.log(f"{mode}_diffaug_positive_loss", generated_positive_loss, sync_dist=True)
        return loss
