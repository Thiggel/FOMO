import torch
from torch import nn
import torch.nn.functional as F
import lightning.pytorch as L
from sklearn.neighbors import KNeighborsClassifier
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
        self.max_epochs = 1
        self.save_hyperparameters(ignore=["model"])
        self.transform = transform
        self.model = model
        self.batch_size = batch_size
        self.k = k
        self.knn = None
        self.linear = nn.Linear(1, 2)  # Dummy linear layer to satisfy PyTorch Lightning
        self._log_per_class_accuracy = False
        self._test_predictions: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    @property
    def num_workers(self) -> int:
        return max(6, min(6, get_num_workers() // 2))

    def extract_features(self, dataloader):
        features = []
        labels = []
        self.model.eval()
        with torch.no_grad():
            for inputs, label in tqdm(dataloader, desc="Extracting features"):
                dtype = next(self.model.parameters()).dtype
                inputs = inputs.to(device=self.device, dtype=dtype)
                outputs = self.model.extract_features(inputs)
                outputs = F.normalize(outputs, dim=-1)
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
        predictions_tensor = torch.from_numpy(predictions).to(targets.device)
        accuracy = (predictions_tensor == targets).float().mean().item()
        dataset_name = self.__class__.__name__.lower().replace("classifier", "")
        self.log(
            f"{dataset_name}_knn_test_accuracy", accuracy, prog_bar=True, sync_dist=True
        )
        if self._log_per_class_accuracy:
            self._test_predictions.append(predictions_tensor.detach().cpu())
            self._test_targets.append(targets.detach().cpu())
        return torch.tensor(accuracy)

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

        if hasattr(self.knn, "classes_"):
            num_classes = len(self.knn.classes_)
        else:
            num_classes = int(torch.unique(targets).numel())
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

        dataset_name = self.__class__.__name__.lower().replace("classifier", "")
        metric_dict = {
            f"{dataset_name}_knn_test_accuracy_class_{class_idx:03d}": acc
            for class_idx, acc in enumerate(per_class_accuracy.tolist())
        }
        self.log_dict(metric_dict, sync_dist=True)
        self._test_predictions = []
        self._test_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0, momentum=0.9)
        return optimizer
