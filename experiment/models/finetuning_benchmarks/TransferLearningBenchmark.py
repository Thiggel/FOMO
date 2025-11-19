import lightning.pytorch as L
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import torch
import os
from experiment.utils.get_num_workers import get_num_workers


class TransferLearningBenchmark(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float,
        transform: transforms.Compose,
        batch_size: int = 64,
        weight_decay: float = 1e-3,
        max_epochs: int = 500,
        num_classes: int = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.base_transform = transform  # Store original transform
        self.transform = transform
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_features = self.model.num_features
        self.probe = nn.Linear(self.num_features, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self._log_per_class_accuracy = False
        self._test_predictions: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1)
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features

    @property
    def num_workers(self) -> int:
        return min(6, get_num_workers())

    def configure_optimizers(self):
        param_groups = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.AdamW(
            param_groups, lr=1e-3
        )

        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.max_epochs * 0.6), int(self.max_epochs * 0.8)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        features = self(inputs)
        outputs = self.probe(features)
        loss = self.loss(outputs, targets)
        self.log(f"train_loss", loss, sync_dist=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        features = self(inputs)
        outputs = self.probe(features)
        loss = self.loss(outputs, targets)
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        self.log(f"val_loss", loss, sync_dist=True)
        self.log(f"val_accuracy", accuracy, sync_dist=True)
        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        features = self(inputs)
        outputs = self.probe(features)
        loss = self.loss(outputs, targets)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == targets).float().mean()
        dataset_name = self.__class__.__name__.lower().replace("finetune", "")
        self.log(f"{dataset_name}_test_loss", loss, sync_dist=True)
        self.log(f"{dataset_name}_test_accuracy", accuracy, sync_dist=True)
        if self._log_per_class_accuracy:
            self._test_predictions.append(predictions.detach().cpu())
            self._test_targets.append(targets.detach().cpu())
        return loss

    def on_test_epoch_start(self) -> None:
        if self._log_per_class_accuracy:
            self._test_predictions = []
            self._test_targets = []

    def on_test_epoch_end(self) -> None:
        if not self._log_per_class_accuracy or not self._test_targets:
            return

        predictions = torch.cat(self._test_predictions).to(self.device)
        targets = torch.cat(self._test_targets).to(self.device)

        if self.trainer is not None and self.trainer.world_size > 1:  # pragma: no cover
            predictions = self.all_gather(predictions).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)

        predictions = predictions.cpu()
        targets = targets.cpu()

        num_classes = self.probe.out_features
        correct_per_class = torch.bincount(
            targets[predictions == targets], minlength=num_classes
        ).to(torch.float32)
        total_per_class = torch.bincount(targets, minlength=num_classes).to(
            torch.float32
        )

        per_class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
        nonzero_mask = total_per_class > 0
        per_class_accuracy[nonzero_mask] = (
            correct_per_class[nonzero_mask] / total_per_class[nonzero_mask]
        )

        dataset_name = self.__class__.__name__.lower().replace("finetune", "")
        metric_dict = {
            f"{dataset_name}_test_accuracy_class_{class_idx:03d}": acc
            for class_idx, acc in enumerate(per_class_accuracy.tolist())
        }
        self.log_dict(metric_dict, sync_dist=True)
        self._test_predictions = []
        self._test_targets = []

    def collate_fn(self, batch):
        if not batch:
            return None

        if isinstance(batch[0], tuple):
            # Handle (image, label) pairs
            images, labels = zip(*batch)
            images = torch.stack([self.transform(img) for img in images])
            labels = torch.tensor(labels)
            return images, labels
        else:
            # Images only
            return torch.stack([self.transform(img) for img in batch])

    def get_datasets(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError
