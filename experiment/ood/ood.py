import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import faiss
from tqdm import tqdm
import os
from torchvision.utils import save_image
from experiment.utils.get_num_workers import get_num_workers


class OOD:
    def __init__(
        self,
        args: dict,
        train: Dataset,
        test: Dataset,
        feature_extractor,
        cycle_idx: int = None,
    ):
        self.train = train
        self.test = test
        self.num_workers = get_num_workers()
        self.feature_extractor = feature_extractor
        self.batch_size = args.fe_batch_size
        self.K = args.k
        self.pct_ood = args.pct_ood
        self.pct_train = args.pct_train
        self.train_features = []
        self.test_features = []
        self.cycle_idx = cycle_idx

        # Set up multi-GPU environment
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Using {self.num_gpus} GPUs!")
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor.to(self.device)

    def extract_features(self):
        train_loader = DataLoader(
            self.train,
            batch_size=self.batch_size
            * self.num_gpus,  # Increase batch size for multi-GPU
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            self.test,
            batch_size=self.batch_size
            * self.num_gpus,  # Increase batch size for multi-GPU
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        self.feature_extractor.eval()

        with torch.no_grad():
            # Extract features from the train dataset
            for batch, _ in tqdm(train_loader, desc="Extracting train features"):
                batch = batch.to(self.device, non_blocking=True)
                features = self.feature_extractor.extract_features(batch).cpu()
                self.train_features.append(features)

            # Extract features from the test dataset
            for batch, _ in tqdm(test_loader, desc="Extracting test features"):
                batch = batch.to(self.device, non_blocking=True)
                features = self.feature_extractor.extract_features(batch).cpu()
                self.test_features.append(features)

        self.train_features = torch.cat(self.train_features)
        self.test_features = torch.cat(self.test_features)

        # Clear GPU memory
        torch.cuda.empty_cache()

    def normalize(self, x):
        return x / (torch.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)

    def ood(self, normalize=True):
        dim = self.train_features.shape[1]
        train_size = self.train_features.shape[0]

        K = min(self.K, train_size)

        # Normalize features
        if normalize:
            self.train_features = self.normalize(self.train_features)
            self.test_features = self.normalize(self.test_features)

        # Use multiple GPUs for FAISS if available
        resources = [faiss.StandardGpuResources() for i in range(self.num_gpus)]
        config = faiss.GpuMultipleClonerOptions()
        config.shard = True  # Distribute index across GPUs

        cpu_index = faiss.IndexFlatL2(dim)
        gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, cpu_index, config)

        gpu_index.add(self.train_features.numpy())

        # Perform search in batches to avoid GPU memory issues
        batch_size = 10000 * self.num_gpus  # Increase batch size for multi-GPU
        scores_ood = []
        for i in range(0, len(self.test_features), batch_size):
            batch = self.test_features[i : i + batch_size].numpy()
            D, _ = gpu_index.search(batch, K)
            scores_ood.append(D[:, -1])  # extracting dist to k-th nearest neighbor

        scores_ood = np.concatenate(scores_ood)
        threshold = np.percentile(scores_ood, 100 * (1 - self.pct_ood))
        is_ood = scores_ood >= threshold
        ood_indices = [i for i, ood_flag in enumerate(is_ood) if ood_flag]

        if not os.path.exists(f"./ood_logs/{self.cycle_idx}"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}")

        np.save(f"./ood_logs/{self.cycle_idx}/scores_ood.npy", scores_ood)
        if not os.path.exists(f"./ood_logs/{self.cycle_idx}/images"):
            os.makedirs(f"./ood_logs/{self.cycle_idx}/images")
        top_k_indices = np.argsort(scores_ood)[-10:][::-1]
        for i, index in enumerate(top_k_indices):
            image = self.test[index][0]
            if len(image.shape) in [3, 4]:
                distance = scores_ood[index]
                image_path = f"./ood_logs/{self.cycle_idx}/images/ood_{i}_distance_{distance:.3f}.jpg"
                save_image(image, image_path)

        return ood_indices, threshold
