import timm
import torch.nn as nn

#ViT using timm library for Dino

class DinoViT(nn.Module):
    def __init__(self, model_id: str, output_size: int, *args, **kwargs):
        super(DinoViT, self).__init__()
        self.model = timm.create_model(model_id, pretrained=True)
        self.config = self.model.default_cfg

        try:
            if hasattr(self.model, 'fc') and self.model.fc is not None:
                self.model.fc = nn.Sequential(
                    self.model.fc,  # Linear(ResNet output, hidden_dim)?
                    nn.ReLU(inplace=True),
                    nn.Linear(self.model.fc.out_features, output_size)
                )
        except AttributeError:
            # handle when there is no model.fc error TODO revisit
            self.model.head = nn.Sequential(
                self.model.head,
                nn.ReLU(inplace=True),
                nn.Linear(self.model.head.out_features, output_size)
            )

    def forward(self, x):
        return self.model(x)
