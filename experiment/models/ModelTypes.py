from dataclasses import dataclass
from enum import Enum
import torch.nn as nn
from torchvision import models

from models.backbones.ViT import ViT


@dataclass
class ModelType:
    resized_image_size: tuple[int, int]
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    @staticmethod
    def model_types():
        return {
            'ResNet18': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs:
                    models.resnet18(
                        pretrained=False,
                        num_classes=output_size,
                        *args, **kwargs
                    ),
            ),
            'ResNet50': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs:
                    models.resnet50(
                        pretrained=False,
                        num_classes=output_size,
                        *args, **kwargs
                    ),
            ),
            'ViTSmall': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs: ViT(
                    model_name='WinKawaks/vit-small-patch16-224',
                    output_size=output_size,
                    *args, **kwargs,
                ),
            ),
            'ViTBase': ModelType(
                resized_image_size=(224, 224),
                model=lambda output_size, *args, **kwargs: ViT(
                    model_name='google/vit-base-patch16-224-in21k',
                    output_size=output_size,
                    *args, **kwargs,
                ),
            ),
        }

    @staticmethod
    def get_model_type(name: str) -> ModelType:
        return ModelTypes.model_types()[name]

    @staticmethod
    def get_model_types() -> list[str]:
        return list(map(
            lambda x: x,
            list(ModelTypes.model_types().keys())
        ))

    @staticmethod
    def get_default_model_type() -> str:
        return ModelTypes.get_model_types()[0]
