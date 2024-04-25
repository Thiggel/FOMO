from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models

from models.SSLMethods.SimCLR import SimCLR


@dataclass
class SSLType:
    module: nn.Module

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
