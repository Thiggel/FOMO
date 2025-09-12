import math
from torch.optim.lr_scheduler import LambdaLR

class ContinuousScheduleMixin:
    """Mixin providing persistent epoch tracking and schedulers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_epochs_completed = 0

    def on_train_epoch_end(self):
        self.total_epochs_completed += 1

    def cosine_anneal(self, min_value: float, max_value: float, T: int) -> float:
        return (max_value - min_value) * (1 + math.cos(math.pi * self.total_epochs_completed / T)) / 2 + min_value

    def cosine_warmup_scheduler(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr_ratio: float = 0.0,
        start_factor: float | None = None,
        base_lr: float | None = None,
        eta_min: float | None = None,
    ):
        start_epoch = self.total_epochs_completed

        def lr_lambda(epoch):
            global_epoch = start_epoch + epoch
            if global_epoch < warmup_epochs:
                warmup = (global_epoch + 1) / warmup_epochs
                if start_factor is not None:
                    return start_factor + (1 - start_factor) * warmup
                return warmup
            progress = (global_epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
            if eta_min is not None and base_lr is not None:
                return eta_min / base_lr + (1 - eta_min / base_lr) * cos_factor
            return min_lr_ratio + (1 - min_lr_ratio) * cos_factor

        for pg in optimizer.param_groups:
            pg.setdefault("initial_lr", pg["lr"])
            pg["lr"] = pg["initial_lr"] * lr_lambda(0)

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
