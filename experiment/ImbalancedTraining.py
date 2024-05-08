import torch
from torch.utils.data import DataLoader, Subset, random_split
import lightning.pytorch as L
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from ood.ood import OOD

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
        self.n_epochs_per_cycle = args["n_epochs_per_cycle"]
        self.max_cycles = args["max_cycles"]
        self.ood_test_split = args["ood_test_split"]

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            self.ssl_method.model.load_state_dict(
                torch.load(self.checkpoint_callback.best_model_path)["state_dict"]
            )

        return self.finetune() if self.args.finetune else {}

    def pretrain_cycle(
        self,
        cycle_idx
    ) -> None:
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

        train_dataset = self.datamodule.train_dataset
        val_dataset = self.datamodule.val_dataset

        ood_train_size = int(self.ood_test_split * len(train_dataset))
        ood_test_size = len(train_dataset) - ood_train_size

        ood_train_dataset, ood_test_dataset = random_split(
            train_dataset, [ood_train_size, ood_test_size]
        )

        ood = OOD(
            fe_batch_size=self.args.fe_batch_size,
            k=self.args.k,
            pct_ood=self.args.pct_ood,
            pct_train=self.args.pct_train,
            train_dataset=ood_train_dataset,
            val_dataset=ood_test_dataset,
            model=self.ssl_method.model,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        ood_samples = Subset(ood_train_dataset, ood_indices)

        # Generate new data from OOD samples
        self.generate_new_data(ood_samples, save_subfolder=f"/{cycle_idx}")
        
        # update the existing train dataset with the augmented data
        self.datamodule.update_dataset(path=f"{self.args["additional_data_path"]}/{cycle_idx}")

        # val_loader = self.datamodule.val_dataloader()

        
        # for batch, batch_idx in val_loader:
        #     ood_samples = self.check_ood(batch)

        #     augmented_data = self.generate_new_data(ood_samples)

        #     self.datamodule.update_dataset(augmented_data)

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
            self.pretrain_cycle(cycle_idx)

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.benchmarks
        results = {}

        for benchmark in benchmarks:
            finetuner = benchmark(
                model=self.ssl_method.model,
                lr=self.args.lr,
            )

            self.trainer_args["max_epochs"] = benchmark.max_epochs

            trainer = L.Trainer(**self.trainer_args)

            trainer.fit(model=finetuner)

            results.update(trainer.test(model=finetuner))

        return results
    
    def generate_new_data(ood_samples):
        pass
