import numpy as np
import torch
from torch.utils.data import Dataset
import faiss
from tqdm import tqdm
from torch.utils.data import DataLoader
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

    def extract_features(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers) #these two are the issue
        test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers) #Because of these two we are using to many dataloader workers 

        # Extract features from the train dataset
        for batch, _ in tqdm(train_loader, desc="Extracting train features"):
            features = self.feature_extractor(batch).cpu().detach()
            self.train_features.append(features)

        # Extract features from the test dataset
        for batch, _ in tqdm(test_loader, desc="Extracting test features"):
            features = self.feature_extractor(batch).cpu().detach()
            self.test_features.append(features)

        self.train_features = torch.cat(self.train_features)
        self.test_features = torch.cat(self.test_features)

        #garbage collection 
        train_loader = None
        test_loader = None

    def normalize(self, x):
        return x / (torch.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)

    def ood(self, normalize=True):
        """
        Since we dont have a set that we know is in-distribution to estimate lambda,
        we will control it with the pct_ood parameter. If we set pct_ood to a conservative value,
        we will always augment the most-OOD samples.
        """
        dim = self.train_features.shape[1]
        train_size = self.train_features.shape[0]

        K = min(self.K, train_size)

        # Normalize features
        if normalize:
            self.train_features = self.normalize(self.train_features)
            self.test_features = self.normalize(self.test_features)

        index = faiss.IndexFlatL2(dim)

        index.add(self.train_features.numpy())

        ################### Using KNN distance Directly ###################
        D, _ = index.search(self.test_features, K)
        scores_ood = D[:, -1]  # extracting dist to k-th nearest neighbor
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
