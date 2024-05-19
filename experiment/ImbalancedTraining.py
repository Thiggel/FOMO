import torch
from torch.utils.data import DataLoader, Subset, random_split
import lightning.pytorch as L
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.ood.ood import OOD
from torchvision import transforms
import copy


class ImbalancedTraining:
    def __init__(
        self,
        args: dict,
        trainer_args: dict,
        ssl_method: L.LightningModule,
        datamodule: L.LightningDataModule,
        checkpoint_callback: L.Callback,
    ):
        self.args = args
        self.trainer_args = trainer_args
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback
        self.n_epochs_per_cycle = args.n_epochs_per_cycle
        self.max_cycles = args.max_cycles
        self.ood_test_split = args.ood_test_split
        self.transform = transforms.Compose(
            [
                transforms.Resize(args.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            self.ssl_method.model.load_state_dict(
                torch.load(self.checkpoint_callback.best_model_path)["state_dict"]
            )

        return self.finetune() if self.args.finetune else {}

    def pretrain_cycle(self, cycle_idx) -> None:
        """
        1. Fit for n epochs
        2. assess OOD samples
        3. generate new data for OOD
        """
        trainer = L.Trainer(**self.trainer_args)

        trainer.fit(
            model=self.ssl_method,
            datamodule=self.datamodule,
        )

        ssl_transform = copy.deepcopy(self.datamodule.train_dataset.dataset.transform)
        self.datamodule.train_dataset.dataset.transform = self.transform

        train_dataset = self.datamodule.train_dataset

        num_ood_test = int(self.ood_test_split * len(train_dataset))
        num_ood_train = len(train_dataset) - num_ood_test

        ood_train_dataset, ood_test_dataset = random_split(
            train_dataset, [num_ood_train, num_ood_test]
        )

        ood = OOD(
            args=self.args,
            train=ood_train_dataset,
            test=ood_test_dataset,
            feature_extractor=self.ssl_method.model.extract_features,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        ood_samples = Subset(ood_train_dataset, ood_indices)

        self.generate_new_data(ood_samples, save_subfolder=f"/{cycle_idx}")

        self.datamodule.update_dataset(
            path=f"{self.args['additional_data_path']}/{cycle_idx}"
        )

        self.datamodule.train_dataset.dataset.transform = ssl_transform

    def pretrain_imbalanced(
        self,
    ) -> None:
        """
        1. Fit for n_epochs_per_cycle epochs,
        2. Use validation set to determine OOD samples
        3. Generate augmentations for OOD samples
        4. Restart
        """
        for cycle_idx in range(self.max_cycles):
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")
            self.pretrain_cycle(cycle_idx)

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.benchmarks
        results = {}

        for benchmark in benchmarks:
            print("\n -- Finetuning benchmark:", benchmark.__name__, "--\n")

            finetuner = benchmark(
                model=self.ssl_method.model, lr=self.args.lr, transform=self.transform
            )

            self.trainer_args["max_epochs"] = finetuner.max_epochs

            self.trainer_args["max_time"] = {
                "minutes": 15,
            }

            trainer = L.Trainer(**self.trainer_args)

            trainer.fit(model=finetuner)

            finetuning_results = trainer.test(model=finetuner)[0]

            results = {**results, **finetuning_results}

        return results

    def generate_new_data(ood_samples):
        pass
