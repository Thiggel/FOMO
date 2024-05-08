import torch
from torch.utils.data import Dataset
import faiss 
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader


def extract_features(train: Dataset, val: Dataset, feature_extractor: torch.nn.Module, batch_size=32):
    train_features = []
    val_features = []
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)

    # Extract features from the train dataset
    for batch in tqdm(train_loader, desc='Extracting train features'):
        features = feature_extractor(batch).cpu().detach().numpy()
        train_features.append(features)

    # Extract features from the validation dataset
    for batch in tqdm(val_loader, desc='Extracting validation features'):
        features = feature_extractor(batch).cpu().detach().numpy()
        val_features.append(features)

    return train_features, val_features

def ood(train_features, val_features, pct_train=1.0, normalize=True, K=1000, pct_ood=0.1):
    '''
    Since we dont have a set that we know is in-distribution to estimate lambda,
    we will control it with the pct_ood parameter. If we set pct_ood to a conservative value,
    we will always augment the most-OOD samples.
    '''
    dim = train_features.shape[1]
    train_size = train_features.shape[0]

    # Normalize features
    if normalize:
        normalizer = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10)
        train_features = normalizer(train_features)
        val_features = normalizer(val_features)

    rand_ind = np.arange(train_size)#np.random.choice(train_size, int(train_size * pct_train), replace=False)
    index = faiss.IndexFlatL2(dim)
    index.add(train_features[rand_ind])
    
    ################### Using KNN distance Directly ###################
    D, _ = index.search(val_features, K)
    scores_ood = D[:,-1] # extracting dist to k-th nearest neighbor 
    threshold = np.percentile(scores_ood, 100*(1-pct_ood))
    is_ood = scores_ood >= threshold
    return is_ood, threshold
