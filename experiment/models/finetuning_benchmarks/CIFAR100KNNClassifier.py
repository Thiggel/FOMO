from torch.utils.data import DataLoader
from .BaseKNNClassifier import BaseKNNClassifier
from .DatasetRoots import get_data_root
from .CIFARDatasets import get_cifar_dataset
from experiment.utils.mp_context import start_method


class CIFAR100KNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        data_root = get_data_root()
        if stage == "fit" or stage is None:
            self.train_dataset = get_cifar_dataset(
                num_classes=100,
                root=data_root,
                train=True,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = get_cifar_dataset(
                num_classes=100,
                root=data_root,
                train=False,
                transform=self.transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context=start_method(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            multiprocessing_context=start_method(),
        )
