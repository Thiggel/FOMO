from dataclasses import dataclass
from enum import Enum
import torch.nn as nn


from experiment.models.backbones.ViT import ViT
from experiment.models.backbones.JeppaViT import VisionTransformer, partial
from experiment.models.backbones.Resnet import ResNet101, ResNet18, ResNet50, ResNet100


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
            ### Add the Jeppa ViT versions, Jeppa needs seperate ViTs because of masking
            "ViTTinyJeppa": ModelType(
                model=lambda image_size, classification_head, output_size, **kwargs: VisionTransformer(
                    patch_size=16,
                    embed_dim=192,
                    depth=12,
                    num_heads=3,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    img_size=[image_size],
                    classification_head=classification_head,
                    output_size=output_size,
                    **kwargs,
                ),
            ),
            "ViTSmallJeppa": ModelType(
                model=lambda image_size, classification_head, output_size, **kwargs: VisionTransformer(
                    patch_size=16,
                    embed_dim=384,
                    depth=12,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    img_size=[image_size],
                    classification_head=classification_head,
                    output_size=output_size,
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
