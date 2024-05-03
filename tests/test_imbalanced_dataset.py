from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods

def test_imbalanced_dataset():
    balanced_datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.NoImbalance,
    )

    balanced_datamodule.setup(stage='train')

    datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.ExponentiallyIncreasing,
    )

    datamodule.setup(stage='train')

    assert (
        len(balanced_datamodule.train_dataset) > len(datamodule.train_dataset),
        "The balanced dataset should have more samples than the imbalanced dataset"
    )
