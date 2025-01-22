import lightning.pytorch as L
from torch import nn, optim
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
        crop_size: int = 224,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.use_deepspeed = True
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.base_transform = transform  # Store original transform
        self.transform = None  # Will be set in setup
        self.crop_size = crop_size
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_features = self.model.num_features
        self.probe = nn.Linear(self.num_features, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def setup(self, stage=None):
        """Setup runs on every GPU."""
        if stage == "fit" or stage is None:
            self.transform = self.get_transform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features

    @property
    def num_workers(self) -> int:
        return get_num_workers()

    def configure_optimizers(self):
        adam_params = {
            "lr": 1e-3,
            "betas": (0.9, 0.95),
        }

        param_groups = [p for p in self.parameters() if p.requires_grad]

        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(param_groups, **adam_params, adamw_mode=True)
        else:
            from torch.optim import AdamW

            optimizer = AdamW(param_groups, **adam_params)

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
        accuracy = (outputs.argmax(dim=1) == targets).float().mean()
        dataset_name = self.__class__.__name__.lower().replace("finetune", "")
        self.log(f"{dataset_name}_test_loss", loss, sync_dist=True)
        self.log(f"{dataset_name}_test_accuracy", accuracy, sync_dist=True)
        return loss

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

    def get_transform(self) -> transforms.Compose:
        """Get dataset-specific transforms with proper normalization."""
        # Create transform chain
        train_transform = transforms.Compose(
            [
                # Resize with proper interpolation
                transforms.Resize(
                    (self.crop_size, self.crop_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                # Data augmentation for training
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                # Convert to tensor and normalize
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return train_transform

    def get_datasets(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError
