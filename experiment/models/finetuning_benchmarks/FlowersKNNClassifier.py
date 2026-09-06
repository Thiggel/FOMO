from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from .BaseKNNClassifier import BaseKNNClassifier
from .DatasetRoots import get_data_root, allow_data_downloads
from experiment.utils.mp_context import start_method


class FlowersKNNClassifier(BaseKNNClassifier):
    def setup(self, stage=None):
        data_root = get_data_root()
        allow_downloads = allow_data_downloads()
        if stage == "fit" or stage is None:
            self.train_dataset = Flowers102(
                root=data_root,
                split="train",
                download=allow_downloads,
                transform=self.transform,
            )
        if stage == "test" or stage is None:
            self.test_dataset = Flowers102(
                root=data_root,
                split="test",
                download=allow_downloads,
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
