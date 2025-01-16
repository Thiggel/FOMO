from typing import Callable
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models, transforms

from experiment.dataset.ContrastiveTransformations import ContrastiveTransformations
from experiment.dataset.MultiCropTransformation import MultiCropTransformation
from experiment.models.SSLMethods.SimCLR import SimCLR
from experiment.models.SSLMethods.Dino import Dino
from experiment.models.SSLMethods.Supervised import Supervised
from experiment.utils.collate_functions import simclr_collate, dino_collate

from experiment.models.SSLMethods.masks.multiblock import MaskCollator as MBMaskCollator
from experiment.dataset.transforms import make_transforms


@dataclass
class SSLType:
    module: nn.Module
    transforms: Callable
    collate_fn: Callable

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.module(*args, **kwargs)


# TODO I dont understand how this all works very well, so please check if the lambda thing to initialize jepa transforms and collate are correct.
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
                transforms=lambda parserargs: ContrastiveTransformations(
                    transforms.Compose(
                        [
                            transforms.Resize(
                                (parserargs.crop_size, parserargs.crop_size)
                            ),
                            transforms.RandomHorizontalFlip(),
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
                collate_fn=lambda parserargs: simclr_collate,
            ),
            "Dino": SSLType(
                module=lambda model, lr, weight_decay, max_epochs, *args, **kwargs: Dino(
                    model=model,
                    lr=lr,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    *args,
                    **kwargs
                ),
                transforms=lambda parserargs: MultiCropTransformation(
                    size=parserargs.crop_size,
                    global_crops_scale=(0.4, 1.0),
                    local_crops_scale=(0.05, 0.4),
                    local_crops_number=6,  # You can adjust this number
                ),
                collate_fn=lambda parserargs: dino_collate,
            ),
            "Supervised": SSLType(
                module=lambda model, lr, weight_decay, max_epochs, *args, **kwargs: Supervised(
                    model=model,
                    lr=lr,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    *args,
                    **kwargs
                ),
                transforms=lambda parserargs: transforms.Compose(
                    [
                        transforms.Resize((parserargs.crop_size, parserargs.crop_size)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
                collate_fn=lambda _: None,
            ),
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
