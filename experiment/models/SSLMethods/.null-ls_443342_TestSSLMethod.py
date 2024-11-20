import lightning.pytorch as L
from torch.optim import AdamW, Optimizer
import torch


class TestSSLMethod(L.LightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.model = torch.nn.Linear(1, 1)  # Dummy linear layer

    def configure_optimizers(self) -> Optimizer:
        adam_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.95),
        }

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(
                self.parameters(), **adam_params, adamw_mode=True
            )
        else:
            optimizer = AdamW(self.parameters(), **adam_params)

        return optimizer

    def training_step(self, batch, batch_idx):
        self.log("train_loss", 0.0)
        return torch.tensor(0.0, requires_grad=True)

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", 0.0)
        return torch.tensor(0.0, requires_grad=True)
