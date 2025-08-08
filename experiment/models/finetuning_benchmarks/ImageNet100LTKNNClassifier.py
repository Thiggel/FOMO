from torch.utils.data import DataLoader
from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.dataset.ImageNetVariants import ImageNetVariants
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods
from experiment.models.finetuning_benchmarks.BaseKNNClassifier import BaseKNNClassifier
from experiment.utils.set_seed import set_seed


class ImageNet100LTKNNClassifier(BaseKNNClassifier):
    """kNN evaluation on the ImageNet-100-LT (long-tailed) dataset."""

    def __init__(self, *args, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def setup(self, stage=None):
        # Recreate the exact long-tailed dataset used during training
        set_seed(self.seed)
        dm = ImbalancedImageNetDataModule(
            dataset_variant=ImageNetVariants.ImageNet100,
            imbalance_method=ImbalanceMethods.PowerLawImbalance,
            transform=self.transform,
        )

        if stage == "fit" or stage is None:
            self.train_dataset = dm.train_dataset
        if stage == "test" or stage is None:
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
