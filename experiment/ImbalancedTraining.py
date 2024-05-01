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
        self.trainer = L.Trainer(**trainer_args)
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback

    def run(self) -> dict:
        if self.args.pretrain:
            self.trainer.fit(
                model=self.ssl_method,
                datamodule=self.datamodule
            )

            self.ssl_method.model.load_state_dict(
                torch.load(self.checkpoint_callback.best_model_path)['state_dict']
            )

        return self.finetune() if self.args.finetune else {}

    def pretrain_cycle(
        self,
        trainer: L.Trainer,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> None:
        """
        1. Fit for n epochs
        2. assess OOD samples
        3. generate new data for OOD
        
        Questions:
            - do I have to restart everything every time
            including loading the model and dataset 
            (in that case save checkpoint every time)
        """
        trainer.fit(model=ssl_method, datamodule=datamodule)

        for batch, batch_idx in val_loader:
            ood_samples = check_ood(batch)

            augmented_data = generate_new_data(
                ood_sample
            )

            datamodule.update_dataset(
                augmented_data
            )


    def pretrain_imbalanced(
        self,
        trainer: L.Trainer,
        datamodule: L.LightningDataModule,
        reload_every_n: int,
        max_cycles: int,
    ) -> None:
        """
        1. Fit for reload_every_n epochs, 
        2. Use validation set to determine OOD samples
        3. Generate augmentations for OOD samples
        4. Restart
        """
        val_loader = datamodule.val_dataloader()

        for cycle_idx in range(max_cycles):
            pretrain_cycle(
                trainer, val_loader, reload_every_n
            )

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.benchmarks
        results = {}

        for benchmark in benchmarks:
            finetuner = benchmark(
                model=self.ssl_method.model,
                lr=self.args.lr,
            )
            trainer = L.Trainer(
                **self.trainer_args
            )

            trainer.fit(model=finetuner)

            results.update(trainer.test(model=finetuner))

        return results
