import torch
from torch.utils.data import DataLoader, Subset, random_split
import lightning.pytorch as L
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from experiment.ood.ood import OOD
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms
import copy

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
        self.transform = transforms.Compose(
            [
                transforms.Resize((args.crop_size, args.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.initial_train_ds_size = len(self.datamodule.train_dataset)
        self.pipe = self.initialize_model()

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            if not self.args.test_mode:
                self.ssl_method.load_state_dict(
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

        trainer.fit(model=self.ssl_method, datamodule=self.datamodule, ckpt_path="last")

        if not self.args.ood_augmentation:
            return
        
        #Removing the diffusion model and adding the samples from the training set. Selecting random images and adding them to the dataset without caring about the balancedness parameter.
        if self.args.remove_diffusion:
            num_samples_to_generate = int(self.args.pct_ood * self.ood_test_split * len(self.datamodule.train_dataset))
            self.datamodule.add_n_samples_by_index(num_samples_to_generate)
            return 

        ssl_transform = copy.deepcopy(self.datamodule.train_dataset.dataset.transform)

        self.datamodule.train_dataset.dataset.transform = self.transform

        train_dataset = Subset(
            self.datamodule.train_dataset, list(range(self.initial_train_ds_size))
        )

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
            cycle_idx=cycle_idx,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        ood_samples = Subset(ood_train_dataset, ood_indices)

        self.datamodule.train_dataset.dataset.transform = None
        diffusion_pipe = self.pipe
        self.generate_new_data(
            ood_samples,
            pipe=diffusion_pipe,
            batch_size=self.args.sd_batch_size,
            save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
        )

        self.datamodule.update_dataset(
            aug_path=f"{self.args.additional_data_path}/{cycle_idx}"
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
            try:
                self.pretrain_cycle(cycle_idx)
            except Exception as e:
                print(f"Error in cycle {cycle_idx}: {e}")

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.get_benchmarks(
            self.args.finetuning_benchmarks
        )
        results = {}

        self.trainer_args.pop("callbacks")

        for benchmark in benchmarks:
            print("\n -- Finetuning benchmark:", benchmark.__name__, "--\n")

            transform = transforms.Compose(
                [
                    transforms.Resize((self.args.crop_size, self.args.crop_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            finetuner = benchmark(
                model=self.ssl_method.model, lr=self.args.lr, transform=transform
            )

            self.trainer_args["max_epochs"] = finetuner.max_epochs

            self.trainer_args["max_time"] = {
                "minutes": 25,
            }

            trainer = L.Trainer(**self.trainer_args)

            try:
                trainer.fit(model=finetuner)
            except Exception as e:
                print(f"Error in benchmark {benchmark.__name__}: {e}")

            finetuning_results = trainer.test(model=finetuner)[0]

            results = {**results, **finetuning_results}

        return results

    def initialize_model(self):
        """
        Load the model first to ensure better flow
        """
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip",
            torch_dtype=torch.float16,
            variation="fp16",
        )
        pipe = pipe.to("cuda")
        return pipe

    def generate_new_data(
        self, ood_samples, pipe, save_subfolder, batch_size=4, nr_to_gen=1
    ) -> None:
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

        k = 0
        for b_start in range(0, len(ood_samples), batch_size):
            batch = [ood_samples[i+b_start][0] for i in range(min(len(ood_samples)-b_start, batch_size))]

            v_imgs = pipe(batch, num_images_per_prompt=nr_to_gen).images
            for i, img in enumerate(v_imgs):
                name = f"/ood_variation_{k}.png"  # TODO: include index?
                img.save(save_subfolder + name)
                k += 1