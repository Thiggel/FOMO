import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def calculate_mean_std(dataset, batch_size=128, num_workers=4):
    """Calculate mean and std of a dataset"""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    mean = 0.0
    std = 0.0
    total_images = 0
    
    # Calculate mean
    for images, _ in tqdm(loader, desc="Calculating mean"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        total_images += batch_samples
    
    mean /= total_images
    
    # Calculate std
    for images, _ in tqdm(loader, desc="Calculating std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        std += ((images - mean.unsqueeze(1))**2).mean(2).sum(0)
    
    std = torch.sqrt(std / total_images)
    
    return mean.tolist(), std.tolist()
