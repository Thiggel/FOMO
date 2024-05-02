import torch
from torch.utils.data import Dataset
import faiss 
from tqdm import tqdm
import numpy as np

def extract_features(train: Dataset, val: Dataset, feature_extractor: torch.nn.Module, batch_size=32):
    train_features = []
    val_features = []
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

    # Extract features from the train dataset
    for batch in tqdm(train_loader, desc='Extracting train features'):
        features = feature_extractor(batch).cpu().detach().numpy()
        train_features.append(features)

    # Extract features from the validation dataset
    for batch in tqdm(val_loader, desc='Extracting validation features'):
        features = feature_extractor(batch).cpu().detach().numpy()
        val_features.append(features)

    return train_features, val_features

def ood(train_features, val_features, pct_train=1.0, normalize=True):
    dim = train_features.shape[1]
    train_size = train_features.shape[0]

    # Normalize features
    if normalize:
        normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
        train_features = normalizer(train_features)
        val_features = normalizer(val_features)

    for K in [1000]:
        rand_ind = np.random.choice(train_size, int(train_size * pct_train), replace=False)
        index = faiss.IndexFlatL2(dim)
        index.add(train_features[rand_ind])

        ################### Using KNN distance Directly ###################
        D, _ = index.search(val_features, K)
        scores_in = -D[:,-1]
        all_results = []
        for ood_dataset, food in food_all.items():
            D, _ = index.search(food, K)
            scores_ood_test = -D[:,-1]
            results = metrics.cal_metric(scores_in, scores_ood_test)
            all_results.append(results)

        metrics.print_all_results(all_results, args.out_datasets, 'KNN')
        print()