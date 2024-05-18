import torch
from torch.utils.data import DataLoader, Subset, random_split
import lightning.pytorch as L
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.ood.ood import OOD
from diffusers import StableUnCLIPImg2ImgPipeline

from torchvision.transforms.functional import to_pil_image
from PIL import Image
import os

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

        train_dataset = self.datamodule.train_dataset

        ood_train_size = int(self.ood_test_split * len(train_dataset))
        ood_test_size = len(train_dataset) - ood_train_size

        ood_train_dataset, ood_test_dataset = random_split(
            train_dataset, [ood_train_size, ood_test_size]
        )

        ood = OOD(
            args=self.args,
            train_dataset=ood_train_dataset,
            val_dataset=ood_test_dataset,
            model=self.ssl_method.model,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        ood_samples = Subset(ood_train_dataset, ood_indices)

        diffusion_pipe = self.initialize_model()
        self.generate_new_data(ood_samples, 
                               pipe=diffusion_pipe, 
                               batch_size=self.args.sd_batch_size, 
                               save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}")

        self.datamodule.update_dataset(
            aug_path=f"{self.args.additional_data_path}/{cycle_idx}"
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
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")
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

    def initialize_model(self):
        """
        Load the model first to ensure better flow
        """
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16")
        pipe = pipe.to("cuda")
        return pipe

    def generate_new_data(self, ood_samples, pipe, save_subfolder, batch_size=4, nr_to_gen = 1) -> None:
        """
        Generate new data based on out-of-distribution (OOD) samples using StableUnclip Img2Img.

        Args:
        - ood_samples (Dataset): Dataset of OOD samples.
        - pipe (DiffusionPipeline): The diffusion model pipeline to generate new data.
        - save_subfolder (str): Path to the folder where generated images will be saved.
        - batch_size (int): Number of samples per batch.
        - nr_to_gen (int): Number of images to generate per sample.
        """
        if not os.path.exists(save_subfolder):
            os.makedirs(save_subfolder)

        ood_sample_loader = DataLoader(ood_samples, batch_size, shuffle=True)

        for ood_samples, ood_index in ood_sample_loader:
            samples = []
            for sample in ood_samples:
                if not isinstance(sample, Image.Image): #check if sample is already a PIL Image to avoid unnecessary conversion
                    sample = to_pil_image(sample)
                samples.append(sample)

            v_imgs = pipe(samples, num_images_per_prompt=nr_to_gen).images   
            for i, img in enumerate(v_imgs):
                name =f"/ood_variation_{i}.png" #TODO: include index?
                img.save(save_subfolder+name)