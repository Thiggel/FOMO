from typing import Callable
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models, transforms

from experiment.dataset.ContrastiveTransformations import ContrastiveTransformations
from experiment.dataset.MultiCropTransformation import MultiCropTransformation
from experiment.models.SSLMethods.SimCLR import SimCLR
from experiment.models.SSLMethods.Dino import Dino
from experiment.models.SSLMethods.MoCo import MoCo, moco_transform
from experiment.models.SSLMethods.Supervised import Supervised
from experiment.utils.collate_functions import simclr_collate, dino_collate

from experiment.dataset.transforms import make_transforms


@dataclass
class SSLType:
    module: nn.Module
    transforms: Callable
    collate_fn: Callable

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.module(*args, **kwargs)


class SSLTypes(Enum):
    @staticmethod
    def ssl_types():
        return {
            "SimCLR": SSLType(
                module=lambda model, lr, temperature, weight_decay, max_epochs, use_temperature_schedule, temperature_min, temperature_max, t_max, *args, **kwargs: SimCLR(
                    model=model,
                    lr=lr,
                    temperature=temperature,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    use_temperature_schedule=use_temperature_schedule,
                    temperature_min=temperature_min,
                    temperature_max=temperature_max,
                    t_max=t_max,
                    *args,
                    **kwargs
                ),
                transforms=lambda parserargs: ContrastiveTransformations(
                    transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                (parserargs.crop_size, parserargs.crop_size)
                            ),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomApply(
                                [
                                    transforms.ColorJitter(
                                        brightness=0.8,
                                        contrast=0.8,
                                        saturation=0.8,
                                        hue=0.2,
                                    )
                                ],
                                p=0.8,
                            ),
                            transforms.RandomApply(
                                [
                                    transforms.GaussianBlur(
                                        kernel_size=23, sigma=(0.1, 2.0)
                                    )
                                ],
                                p=0.5,
                            ),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    ),
                    n_views=2,
                ),
                collate_fn=lambda parserargs: simclr_collate,
            ),
            "MoCo": SSLType(
                module=lambda model, lr, temperature, weight_decay, max_epochs, *args, **kwargs: MoCo(
                    model=model,
                    lr=lr,
                    temperature=temperature,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    *args,
                    **kwargs
                ),
                transforms=lambda parserargs: moco_transform(
                    crop_size=parserargs.crop_size
                ),
                collate_fn=lambda parserargs: simclr_collate,  # MoCo can use the same collate function as SimCLR
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
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
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
