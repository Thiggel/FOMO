"""Cache frozen SD3-VAE latent features for original source images.

The cache deliberately uses one deterministic source rendering per underlying
dataset index.  SSL still sees stochastic views; only the frozen teacher target
is reused, which makes direct-external-prior controls computationally viable.
"""
import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision import transforms

from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.utils.set_seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    set_seed(args.seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    datamodule = ImbalancedDataModule(
        dataset_path="clane9/imagenet-100", split="train+validation",
        imbalance_method=ImbalanceMethods.init_method("power_law_imbalance"),
        transform=transform, train_batch_size=args.batch_size, val_batch_size=args.batch_size,
        checkpoint_filename=f"sd3_teacher_cache_seed_{args.seed}",
        additional_data_path=f"sd3_teacher_cache_source_seed_{args.seed}",
    )
    datamodule.dataset.return_index = True
    loader = DataLoader(datamodule.train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", subfolder="vae",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        cache_dir=__import__("os").environ.get("HF_HUB_CACHE"), local_files_only=True,
    ).to(device).eval()
    features = None
    with torch.no_grad():
        for images, _, indices in loader:
            images = images.to(device=device, dtype=next(vae.parameters()).dtype)
            vae_pixels = (images * images.new_tensor([.229,.224,.225])[None,:,None,None] + images.new_tensor([.485,.456,.406])[None,:,None,None]) * 2 - 1
            target = vae.encode(vae_pixels).latent_dist.mean.mean(dim=(2,3)).float().cpu()
            if features is None:
                features = torch.full((len(datamodule.dataset), target.shape[1]), float("nan"))
            features[indices.long()] = target
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": features, "seed": args.seed, "n_valid": int(torch.isfinite(features).all(1).sum())}, output)
    print(f"Saved {torch.isfinite(features).all(1).sum().item()} SD3-VAE targets to {output}")


if __name__ == "__main__":
    main()
