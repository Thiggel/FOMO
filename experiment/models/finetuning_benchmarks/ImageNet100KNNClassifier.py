from torch.utils.data import DataLoader
from experiment.dataset.ImbalancedDataModule import ImbalancedDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.models.finetuning_benchmarks.BaseKNNClassifier import BaseKNNClassifier
from experiment.utils.set_seed import set_seed


class ImageNet100KNNClassifier(BaseKNNClassifier):
    """kNN evaluation on the balanced ImageNet-100 dataset."""

    def __init__(self, *args, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def setup(self, stage=None):
        # Recreate the same dataset splits deterministically
        set_seed(self.seed)
        dm = ImbalancedDataModule(
            dataset_path="clane9/imagenet-100",
            imbalance_method=ImbalanceMethods.NoImbalance,
            transform=self.transform,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = dm.train_dataset
        if stage == "test" or stage is None:
            # Use the held-out test split for evaluation
            self.test_dataset = dm.test_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=False,
            multiprocessing_context="spawn",
        )
