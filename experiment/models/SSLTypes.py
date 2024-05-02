from typing import Callable
from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models

from experiment.models.SSLMethods.SimCLR import SimCLR


@dataclass
class SSLType:
    module: nn.Module
    transforms: Callable

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.module(*args, **kwargs)


class SSLTypes(Enum):
    @staticmethod
    def ssl_types():
        return {
            'SimCLR': SSLType(
                module=lambda \
                    model, \
                    lr, \
                    temperature, \
                    weight_decay, \
                    max_epochs, \
                    *args, **kwargs: SimCLR(
                        model=model,
                        lr=lr,
                        temperature=temperature,
                        weight_decay=weight_decay,
                        max_epochs=max_epochs,
                        *args, **kwargs
                    )
                transforms=ContrastiveTransformations(
                    transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomResizedCrop(size=96),
                        transforms.RandomApply([
                            transforms.ColorJitter(
                                brightness=0.5, 
                                contrast=0.5, 
                                saturation=0.5, 
                                hue=0.1
                            )
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.GaussianBlur(kernel_size=9),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ]),
                    n_views=2,
                )
            ),
        }

    @staticmethod
    def get_ssl_type(name: str) -> SSLType:
        return SSLTypes.ssl_types()[name]

    @staticmethod
    def get_ssl_types() -> list[str]:
        return list(map(
            lambda x: x,
            list(SSLTypes.ssl_types().keys())
        ))

    @staticmethod
    def get_default_ssl_type() -> str:
        return SSLTypes.get_ssl_types()[0]
