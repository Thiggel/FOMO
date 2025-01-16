from dataclasses import dataclass
from enum import Enum
import torch.nn as nn


from experiment.models.backbones.ViT import ViT
from experiment.models.backbones.Resnet import ResNet101, ResNet18, ResNet50, ResNet101


@dataclass
class ModelType:
    model: nn.Module

    def initialize(self, *args, **kwargs) -> nn.Module:
        return self.model(*args, **kwargs)


class ModelTypes(Enum):
    @staticmethod
    def model_types():
        return {
            "ResNet50": ModelType(
                model=lambda output_size, *args, **kwargs: ResNet50(output_size),
            ),
            "ResNet101": ModelType(
                model=lambda output_size, *args, **kwargs: ResNet101(output_size),
            ),
            "ResNet18": ModelType(
                model=lambda output_size, *args, **kwargs: ResNet18(output_size),
            ),
            "ViTTiny": ModelType(
                model=lambda output_size, *args, **kwargs: ViT(
                    model_id="WinKawaks/vit-tiny-patch16-224",
                    output_size=output_size,
                    *args,
                    **kwargs,
                ),
            ),
            "ViTSmall": ModelType(
                model=lambda output_size, *args, **kwargs: ViT(
                    model_id="WinKawaks/vit-small-patch16-224",
                    output_size=output_size,
                    *args,
                    **kwargs,
                ),
            ),
            "ViTBase": ModelType(
                model=lambda output_size, *args, **kwargs: ViT(
                    model_id="google/vit-base-patch16-224-in21k",
                    output_size=output_size,
                    *args,
                    **kwargs,
                ),
            ),
        }

    @staticmethod
    def get_model_type(name: str) -> ModelType:
        return ModelTypes.model_types()[name]

    @staticmethod
    def get_model_types() -> list[str]:
        return list(map(lambda x: x, list(ModelTypes.model_types().keys())))

    @staticmethod
    def get_default_model_type() -> str:
        return ModelTypes.get_model_types()[0]
