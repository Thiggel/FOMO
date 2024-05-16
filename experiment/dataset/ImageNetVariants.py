from enum import Enum
from dataclasses import dataclass


@dataclass
class ImageNetVariant:
    name: str
    path: str


class ImageNetVariants(Enum):
    ImageNet100 = ImageNetVariant(
        name="100",
        path="clane9/imagenet-100",
    )

    ImageNet1k = ImageNetVariant(
        name="1k",
        path="imagenet-1k",
    )

    ImageNetDummy = ImageNetVariant(
        name="dummy",
        path="Sijuade/ImageNette",
    )

    @staticmethod
    def init_variant(variant: str):
        return {member.value.name: member for member in ImageNetVariants}[variant]

    @staticmethod
    def get_variants():
        return [member.value.name for member in ImageNetVariants]

    @staticmethod
    def get_default_variant():
        return ImageNetVariants.get_variants()[0]
