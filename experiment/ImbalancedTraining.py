import sys
import io
from tqdm import tqdm
import torch
from torch.utils.data import Subset, random_split, Dataset
from lightning.pytorch.strategies import DeepSpeedStrategy
import lightning.pytorch as L
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
from experiment.ood.ood import OOD
from diffusers import StableUnCLIPImg2ImgPipeline
from torchvision import transforms
import copy
import matplotlib.pyplot as plt

import os
import pickle

from experiment.dataset.ImageStorage import ImageStorage


class ImbalancedTraining:
    def __init__(
        self,
        args: dict,
        trainer_args: dict,
        ssl_method: L.LightningModule,
        datamodule: L.LightningDataModule,
        checkpoint_callback: L.Callback,
        checkpoint_filename: str,
        save_class_distribution: bool = False,
        run_idx: int = 0,
    ):
        self.args = args
        self.run_idx = run_idx
        self.trainer_args = trainer_args
        self.ssl_method = ssl_method
        self.datamodule = datamodule
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_filename = checkpoint_filename
        self.save_class_distribution = save_class_distribution
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
        if self.datamodule is not None:
            self.initial_train_ds_size = len(self.datamodule.train_dataset)
        # self.pipe = self.initialize_model("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        if self.args.pretrain:
            self.pretrain_imbalanced()

            if not self.args.test_mode and os.path.exists(
                self.checkpoint_callback.best_model_path
            ):
                output_path = (
                    self.checkpoint_callback.best_model_path
                    + "_fp32.pt".replace(":", "_").replace(" ", "_")
                )

                convert_zero_checkpoint_to_fp32_state_dict(
                    self.checkpoint_callback.best_model_path,
                    output_path,
                )

                self.ssl_method.load_state_dict(torch.load(output_path)["state_dict"])

        return self.finetune() if self.args.finetune else {}

    def pretrain_cycle(self, cycle_idx) -> None:
        """
        1. Fit for n epochs
        2. assess OOD samples
        3. generate new data for OOD
        """
        trainer = L.Trainer(**self.trainer_args)

        # handle dataloader worker issue -> note: finetuning has the same issue, make sure to se the loaders to None there too.

        trainer.fit(model=self.ssl_method, datamodule=self.datamodule)
        # Set the dataloaders to None for garbage collection
        self.datamodule.set_dataloaders_none()
        # They will be reinstantiate anyway in the next trainer.fit

        if not self.args.ood_augmentation:
            return

        # Removing the diffusion model and adding the samples from the training set. Selecting random images and adding them to the dataset without caring about the balancedness parameter.
        if self.args.remove_diffusion:
            print(f"initial dataset size: {self.initial_train_ds_size}")
            print(f"dataset_size: {len(self.datamodule.train_dataset)}")
            num_samples_to_generate = int(
                self.args.pct_ood * self.ood_test_split * self.initial_train_ds_size
            )
            self.datamodule.add_n_samples_by_index(num_samples_to_generate)
            print(
                f"added {num_samples_to_generate} samples to the training set, dataset size is now {len(self.datamodule.train_dataset.indices)}"
            )
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

        indices_to_be_augmented = (
            self.get_ood_indices(ood_train_dataset, ood_test_dataset, cycle_idx)
            if self.args.use_ood
            else self.get_random_indices(ood_train_dataset)
        )

        ood_samples = Subset(ood_train_dataset, indices_to_be_augmented)

        if cycle_idx < self.max_cycles - 1:
            self.datamodule.train_dataset.dataset.transform = None
            diffusion_pipe = self.initialize_model(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self.generate_new_data(
                ood_samples,
                pipe=diffusion_pipe,
                batch_size=self.args.sd_batch_size,
                save_subfolder=f"{self.args.additional_data_path}/{cycle_idx}",
            )

            self.datamodule.train_dataset.dataset.transform = ssl_transform

    def get_num_samples_to_generate(self) -> int:
        return int(
            self.args.pct_ood * self.ood_test_split * len(self.datamodule.train_dataset)
        )

    def get_random_indices(self, ood_train_dataset) -> list:
        num_samples = self.get_num_samples_to_generate()
        return torch.randperm(len(ood_train_dataset))[:num_samples].tolist()

    def get_ood_indices(self, ood_train_dataset, ood_test_dataset, cycle_idx) -> list:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ssl_method.to(device)
        self.ssl_method.model.to(dtype=self.ssl_method.dtype)
        ood = OOD(
            args=self.args,
            train=ood_train_dataset,
            test=ood_test_dataset,
            feature_extractor=self.ssl_method.model.extract_features,
            cycle_idx=cycle_idx,
            device=self.ssl_method.device,
            dtype=self.ssl_method.dtype,
        )

        ood.extract_features()
        ood_indices, _ = ood.ood()
        return ood_indices

    def save_class_dist(self, dataset: Dataset, filename="class_distribution") -> None:
        """
        Visualize the class distribution of the dataset.
        """
        class_distribution = [0] * self.datamodule.dataset.num_classes

        for index in tqdm(range(len(dataset)), desc="Calculating class distribution"):
            class_distribution[dataset[index][1]] += 1

        with open(filename + ".pkl", "wb") as f:
            pickle.dump(class_distribution, f)

        plt.bar(range(self.datamodule.dataset.num_classes), class_distribution)
        plt.xlabel("Class Index")
        plt.ylabel("Number of Samples")
        plt.savefig(filename + ".pdf", format="pdf")
        plt.close()

    def pretrain_imbalanced(
        self,
    ) -> None:
        """
        1. Fit for n_epochs_per_cycle epochs,
        2. Use validation set to determine OOD samples
        3. Generate augmentations for OOD samples
        4. Restart
        """
        visualization_dir = (
            f"visualizations/class_distributions/{self.checkpoint_filename}"
        )
        os.makedirs(visualization_dir, exist_ok=True)

        if self.save_class_distribution:
            self.save_class_dist(
                self.datamodule.train_dataset, f"{visualization_dir}/initial_class_dist"
            )

        for cycle_idx in range(self.max_cycles):
            print(f"Run {self.run_idx + 1}/{self.args.num_runs}")
            print(f"Pretraining cycle {cycle_idx + 1}/{self.max_cycles}")
            self.pretrain_cycle(cycle_idx)

            if self.save_class_distribution:
                self.save_class_dist(
                    self.datamodule.train_dataset,
                    f"{visualization_dir}/class_dist_after_cycle_{cycle_idx}",
                )

    def finetune(self) -> dict:
        benchmarks = FinetuningBenchmarks.get_benchmarks(
            self.args.finetuning_benchmarks
        )
        results = {}

        self.trainer_args.pop("callbacks")

        torch.multiprocessing.set_sharing_strategy("file_system")

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

            # dataloader is already handled fine here because each loop should set the past loader to None.
            finetuner = benchmark(
                model=self.ssl_method.model, lr=self.args.lr, transform=transform
            )

            self.trainer_args["max_epochs"] = finetuner.max_epochs

            self.trainer_args["max_time"] = {
                "minutes": 25,
            }

            self.trainer_args["accumulate_grad_batches"] = 1

            strategy = DeepSpeedStrategy(
                config={
                    "train_batch_size": 64 * torch.cuda.device_count(),
                    "bf16": {"enabled": True},
                    "zero_optimization": {
                        "stage": 2,
                        "offload_optimizer": {"device": "cpu", "pin_memory": True},
                        "offload_param": {"device": "cpu", "pin_memory": True},
                    },
                }
            )
            self.trainer_args["strategy"] = strategy

            trainer = L.Trainer(**self.trainer_args)

            trainer.fit(model=finetuner)

            finetuning_results = trainer.test(model=finetuner)[0]

            results = {**results, **finetuning_results}

        return results

    def initialize_model(self, device):
        """
        Load the model first to ensure better flow
        """
        if device == "cpu":
            pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip"
            )
        else:
            pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1-unclip",
                torch_dtype=torch.float16,
                variation="bf16",
            )
        pipe = pipe.to(device)
        return pipe

    def generate_new_data(
        self, ood_samples, pipe, save_subfolder, batch_size=4, nr_to_gen=1
    ) -> None:
        cycle_idx = int(save_subfolder.split("/")[-1])  # Extract cycle index from path
        image_storage = ImageStorage(self.args.additional_data_path)

        k = 0
        for b_start in tqdm(
            range(0, len(ood_samples), batch_size), desc="Generating New Data..."
        ):
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            batch = [
                ood_samples[i + b_start][0]
                for i in range(min(len(ood_samples) - b_start, batch_size))
            ]
            v_imgs = pipe(batch, num_images_per_prompt=nr_to_gen).images

            # Save batch of images
            image_storage.save_batch(v_imgs, cycle_idx, k)
            k += len(v_imgs)

            sys.stdout = old_stdout
