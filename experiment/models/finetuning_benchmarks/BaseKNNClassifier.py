import torch
from torch import nn
import lightning.pytorch as L
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchvision import transforms

from experiment.utils.get_num_workers import get_num_workers


class BaseKNNClassifier(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 64,
        k: int = 5,
        transform: transforms.Compose = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.use_deepspeed = False
        self.max_epochs = 1
        self.save_hyperparameters(ignore=["model"])
        self.transform = transform if transform is not None else self.get_transform()
        self.model = model
        self.batch_size = batch_size
        self.k = k
        self.knn = None
        self.linear = nn.Linear(1, 2)  # Dummy linear layer to satisfy PyTorch Lightning

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
                # Data augmentation for training
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                # Convert to tensor and normalize
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        return train_transform

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

    @property
    def num_workers(self) -> int:
        return min(6, get_num_workers() // 2)

    def extract_features(self, dataloader):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, label in tqdm(dataloader, desc="Extracting features"):
                dtype = next(self.model.parameters()).dtype
                inputs = inputs.to(device=self.device, dtype=dtype)
                outputs = self.model.extract_features(inputs)
                features.append(outputs.cpu())
                labels.append(label)
        return torch.cat(features), torch.cat(labels)

    def on_train_start(self):
        # Fit KNN at the start of training
        train_loader = self.train_dataloader()
        train_features, train_labels = self.extract_features(train_loader)
        self.knn = KNeighborsClassifier(n_neighbors=self.k)
        self.knn.fit(train_features.to(torch.float32).numpy(), train_labels.numpy())

    def training_step(self, batch, batch_idx):
        return torch.tensor(0.0, requires_grad=True)

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        inputs = inputs.to(self.device)
        features = self.model.extract_features(inputs)
        predictions = self.knn.predict(features.to(torch.float32).cpu().numpy())
        accuracy = accuracy_score(targets.cpu().numpy(), predictions)
        dataset_name = self.__class__.__name__.lower().replace("classifier", "")
        self.log(
            f"{dataset_name}_knn_test_accuracy", accuracy, prog_bar=True, sync_dist=True
        )
        return torch.tensor(accuracy)

    def configure_optimizers(self):
        if torch.cuda.is_available():
            from deepspeed.ops.adam import DeepSpeedCPUAdam

            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=0)
        else:
            from torch.optim import AdamW

            optimizer = AdamW(self.parameters(), lr=0)
        return optimizer
