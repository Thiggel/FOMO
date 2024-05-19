import torch
from torch import nn
from transformers import ViTModel


class ViT(nn.Module):
    def __init__(self, model_id: str, output_size: int, *args, **kwargs):
        super().__init__()

        self.model, self.config = self._load_model(model_id)
        self.head = nn.Linear(self.config.hidden_size, output_size)
        self.output_size = output_size

    def _load_model(self, model_name: str) -> nn.Module:
        model = ViTModel.from_pretrained(model_name)

        return model, model.config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images).last_hidden_state[:, 0, :]
        output = self.head(output)

        return output

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images).last_hidden_state[:, 0, :]
