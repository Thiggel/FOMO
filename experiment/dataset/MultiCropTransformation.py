from torch import Tensor
import torch
from torchvision import transforms
from typing import List, Tuple


class MultiCropTransformation:
    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        size=224,
    ):
        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        # Global transforms
        self.global_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=global_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.GaussianBlur(kernel_size=9),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
                normalize,
            ]
        )

        # Local transforms
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.GaussianBlur(kernel_size=9),
                normalize,
            ]
        )

        self.local_crops_number = local_crops_number

    def __call__(self, image):
        crops = []
        # Two global crops
        crops.append(self.global_transform(image))
        crops.append(self.global_transform(image))
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def dino_collate(batch: list) -> tuple[list[Tensor], Tensor]:
    """
    Collate function for DINO that handles multiple crops.
    Each item in batch contains [crops, label] where crops is a list of tensors.
    """
    all_crops = []
    num_crops = len(batch[0][0])  # Number of crops (2 global + n local)

    # For each crop position
    for i in range(num_crops):
        # Stack all batch items for this crop position
        crop_batch = torch.stack([item[0][i] for item in batch])
        all_crops.append(crop_batch)

    # Stack labels
    labels = torch.tensor([item[1] for item in batch])

    return all_crops, labels
