import torch
from torch import nn
from transformers import ViTModel, ViTConfig


class ViT(nn.Module):
    def __init__(self, model_id: str, output_size: int, *args, **kwargs):
        super().__init__()

        self.model, self.config = self._load_model(model_id)
        self.head = nn.Linear(self.config.hidden_size, output_size)
        self.output_size = output_size
        self.num_features = self.config.hidden_size

    def _load_model(self, model_name: str) -> nn.Module:
        try:
            config = ViTConfig.from_pretrained(model_name)
        except OSError:
            # The WinKawaks repositories only provide architecture configs, and
            # Alex compute nodes are intentionally offline.  These are the
            # canonical DeiT/ViT Tiny, Small, and Base dimensions; constructing
            # them locally is equivalent and removes a brittle network lookup.
            if "vit-tiny" in model_name:
                hidden_size, layers, heads = 192, 12, 3
            elif "vit-small" in model_name:
                hidden_size, layers, heads = 384, 12, 6
            elif "vit-base" in model_name:
                hidden_size, layers, heads = 768, 12, 12
            else:
                raise
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                hidden_size=hidden_size,
                num_hidden_layers=layers,
                num_attention_heads=heads,
                intermediate_size=hidden_size * 4,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
            )
        model = ViTModel(config)

        return model, config

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self._encode(images)
        output = self.head(output)

        return output

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self._encode(images)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        # DINO's local crops are 96 pixels, so the backbone has to accept inputs
        # smaller than the 224 the position embeddings were sized for.
        # ``interpolate_pos_encoding`` resamples them, exactly as the reference
        # DINO ViT does, and is a no-op at 224 (verified bit-identical), so
        # enabling it unconditionally leaves the other ViT experiments alone.
        return self.model(
            images, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0, :]
