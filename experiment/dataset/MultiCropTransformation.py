from torch import Tensor
import torch
from torchvision import transforms
from typing import List, Tuple


class MultiCropTransformation:
    """DINO multi-crop augmentation (Caron et al., 2021, Sec. 3 and Appendix).

    Three deviations from the original recipe were corrected here:

    * ``RandomSolarize`` was thresholded at ``0.5``.  The transform runs on a
      PIL image whose values live in ``[0, 255]``, so that threshold inverted
      99.6 percent of the pixels instead of the intended half, turning one in
      five global crops into a photographic negative.  DINO solarizes at 128.
    * Local crops were rendered at the global resolution.  DINO renders them at
      96 pixels, which is what makes eight local views affordable.
    * Both global crops shared one augmentation distribution.  DINO follows BYOL
      in blurring the first global crop always and the second rarely, and in
      solarizing only the second.
    """

    def __init__(
        self,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        size=224,
        local_crops_size=96,
        solarize_threshold=128,
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
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        def blur(p: float):
            return transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=p
            )

        global_crop = transforms.RandomResizedCrop(
            size,
            scale=global_crops_scale,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        # First global crop: always blurred, never solarized.
        self.global_transform = transforms.Compose(
            [global_crop, flip_and_color_jitter, blur(1.0), normalize]
        )

        # Second global crop: rarely blurred, sometimes solarized.
        self.global_transform2 = transforms.Compose(
            [
                global_crop,
                flip_and_color_jitter,
                blur(0.1),
                transforms.RandomSolarize(threshold=solarize_threshold, p=0.2),
                normalize,
            ]
        )

        # Local transforms, rendered at the reduced local resolution.
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size,
                    scale=local_crops_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                blur(0.5),
                normalize,
            ]
        )

        self.local_crops_number = local_crops_number
        self.local_crops_size = local_crops_size

    def __call__(self, image):
        crops = []
        # Two global crops, asymmetrically augmented.
        crops.append(self.global_transform(image))
        crops.append(self.global_transform2(image))
        # Local crops.
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


def dino_collate(batch: list) -> tuple[list[Tensor], Tensor]:
    """
    Collate function for DINO that handles multiple crops.
    Each item in batch contains [crops, label] where crops is a list of tensors.

    Global and local crops have different spatial sizes, so each crop position
    is stacked independently and the positions are returned as a list.
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
