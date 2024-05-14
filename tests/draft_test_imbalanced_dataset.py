from experiment.dataset.ImbalancedImageNetDataModule import ImbalancedImageNetDataModule
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


def imbalanced_dataset_length():
    balanced_datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.NoImbalance,
        checkpoint_filename="test_balanced_dataset",
        test_mode=True,
    )

    balanced_datamodule.setup(stage="train")

    datamodule = ImbalancedImageNetDataModule(
        imbalance_method=ImbalanceMethods.ExponentiallyIncreasing,
        checkpoint_filename="test_imbalanced_dataset",
        test_mode=True,
    )

    datamodule.setup(stage="train")

    assert len(balanced_datamodule.train_dataset) > len(
        datamodule.train_dataset
    ), "The balanced dataset should have more samples than the imbalanced dataset"
