import torch
from torch.utils.data import DataLoader
import lightning.pytorch as L


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

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            self.ssl_method.model.load_state_dict(
                torch.load(self.checkpoint_callback.best_model_path)['state_dict']
            )

        return self.finetune() if self.args.finetune else {}

    def pretrain_cycle(
        self,
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

        val_loader = self.datamodule.val_dataloader()

        for batch, batch_idx in val_loader:
            ood_samples = self.check_ood(batch)

            augmented_data = self.generate_new_data(
                ood_samples
            )

            self.datamodule.update_dataset(
                augmented_data
            )


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
            self.pretrain_cycle()

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.benchmarks
        results = {}

        for benchmark in benchmarks:
            finetuner = benchmark(
                model=self.ssl_method.model,
                lr=self.args.lr,
            )

            self.trainer_args['max_epochs'] = benchmark.max_epochs
            
            trainer = L.Trainer(
                **self.trainer_args
            )

            trainer.fit(model=finetuner)

            results.update(trainer.test(model=finetuner))

        return results