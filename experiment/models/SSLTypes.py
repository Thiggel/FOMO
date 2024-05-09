from typing import Callable
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models, transforms

from experiment.dataset.ContrastiveTransformations import ContrastiveTransformations
from experiment.models.SSLMethods.SimCLR import SimCLR
from experiment.models.SSLMethods.IJepa import IJepa
from experiment.utils.collate_functions import simclr_collate

from experiment.models.SSLMethods.masks.multiblock import MaskCollator as MBMaskCollator
from experiment.dataset.transforms import make_transforms



@dataclass
class SSLType:
    module: nn.Module
    transforms: Callable
    collate_fn: Callable

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.module(*args, **kwargs)

#TODO I dont understand how this all works very well, so please check if the lambda thing to initialize jepa transforms and collate are correct.
class SSLTypes(Enum):
    @staticmethod
    def ssl_types():
        return {
            "SimCLR": SSLType(
                module=lambda model, lr, temperature, weight_decay, max_epochs, *args, **kwargs: SimCLR(
                    model=model,
                    lr=lr,
                    temperature=temperature,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    *args,
                    **kwargs
                ),
                transforms= lambda parserargs: ContrastiveTransformations(
                    transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomResizedCrop(size=parserargs.crop_size), #was 96
                            transforms.RandomApply(
                                [
                                    transforms.ColorJitter(
                                        brightness=0.5,
                                        contrast=0.5,
                                        saturation=0.5,
                                        hue=0.1,
                                    )
                                ],
                                p=0.8,
                            ),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.GaussianBlur(kernel_size=9),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,)),
                        ]
                    ),
                    n_views=2,
                ),
                collate_fn=lambda parserargs: simclr_collate
            ),
            "I-Jepa": SSLType(
                module=lambda model, lr, temperature, weight_decay, max_epochs, parserargs, iterations_per_epoch, *args, **kwargs: IJepa(
                    model=model,
                    lr=lr,
                    args = parserargs,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    iterations_per_epoch = iterations_per_epoch,
                    *args,
                    **kwargs
                ),
                collate_fn= lambda parserargs: MBMaskCollator(
                    input_size=parserargs.crop_size,
                    patch_size=parserargs.patch_size,
                    pred_mask_scale=parserargs.pred_mask_scale,
                    enc_mask_scale=parserargs.enc_mask_scale,
                    aspect_ratio=parserargs.aspect_ratio,
                    nenc=parserargs.num_enc_masks,
                    npred=parserargs.num_pred_masks,
                    allow_overlap=parserargs.allow_overlap,
                    min_keep=parserargs.min_keep),

                transforms= lambda parserargs: make_transforms(
                    crop_size=parserargs.crop_size,
                    crop_scale=parserargs.crop_scale,
                    gaussian_blur=parserargs.use_gaussian_blur,
                    horizontal_flip=parserargs.use_horizontal_flip,
                    color_distortion=parserargs.use_color_distortion,
                    color_jitter=parserargs.color_jitter_strength)
            )
        }

    @staticmethod
    def get_ssl_type(name: str) -> SSLType:
        return SSLTypes.ssl_types()[name]

    @staticmethod
    def get_ssl_types() -> list[str]:
        return list(map(lambda x: x, list(SSLTypes.ssl_types().keys())))

    @staticmethod
    def get_default_ssl_type() -> str:
        return SSLTypes.get_ssl_types()[0]
