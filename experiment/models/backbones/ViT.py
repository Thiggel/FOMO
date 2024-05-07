import torch
from torch import nn
from transformers import ViTModel


class Downscaling(nn.Sequential):
    def __init__(self):
        super().__init__(
            # input shape: 902, 902, 1
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
            ),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )


class ViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        output_size: int,
        downscale: bool = False,
        complex_mlp_head: bool = False,
        *args,
        **kwargs
    ):
        super().__init__()

        self.downscale = downscale
        self.downscaling = Downscaling()
        self.model, self.config = self._load_model(model_name)
        self.head = (
            nn.Linear(self.config.hidden_size, 2 * output_size)
            if not complex_mlp_head
            else nn.Sequential(
                nn.Linear(self.config.hidden_size, 2 * output_size),
                nn.ReLU(),
                nn.Linear(2 * output_size, 2 * output_size),
                nn.ReLU(),
                nn.Linear(2 * output_size, 2 * output_size),
            )
        )
        self.output_size = output_size

    def _load_model(self, model_name: str) -> nn.Module:
        model = ViTModel.from_pretrained(model_name)

        return model, model.config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images).last_hidden_state[:, 0, :]
        output = self.head(output).reshape(-1, self.output_size, 2)

        return output
