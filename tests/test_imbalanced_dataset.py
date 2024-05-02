from dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods

def test_imbalanced_dataset():
    balanced_datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.NoImbalance,
    )

    balanced_datamodule.setup()

    datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.ExponentiallyIncreasing,
    )

    datamodule.setup()

    assert len(balanced_datamodule.train_dataset) > len(datamodule.train_dataset)
