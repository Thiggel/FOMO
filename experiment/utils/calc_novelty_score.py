import torch
from datasets import load_dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score
import faiss
import statistics


class HFDataset(torch.utils.data.Dataset):
    """Wrapper for HuggingFace datasets to apply transforms"""

    def __init__(self, dataset, transform=None, normal_class=None):
        if normal_class is not None:
            # Filter dataset to only include samples of the normal class
            # Use 'fine_label' for CIFAR-100 and 'label' for CIFAR-10
            label_key = "fine_label" if "fine_label" in dataset.features else "label"
            self.dataset = dataset.filter(lambda x: x[label_key] == normal_class)
        else:
            self.dataset = dataset
        self.transform = transform
        # Store the label key for later use
        self.label_key = "fine_label" if "fine_label" in dataset.features else "label"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_key = "image" if "image" in item else "img"
        image = (
            item[image_key]
            if isinstance(item[image_key], Image.Image)
            else Image.fromarray(item[image_key])
        )

        if self.transform:
            image = self.transform(image)

        # Use the correct label key
        label = (
            int(item[self.label_key] != self.normal_class)
            if hasattr(self, "normal_class")
            else item[self.label_key]
        )
        return image, label


class ProgressTracker:
    """Tracks progress and timing for the novelty score calculation process"""

    def __init__(self):
        self.start_time = time.time()
        self.dataset_times = {}
        self.current_dataset = None

    def start_dataset(self, dataset_name, total_samples):
        self.current_dataset = dataset_name
        self.dataset_times[self.current_dataset] = {
            "start_time": time.time(),
            "total_samples": total_samples,
            "feature_extraction_time": 0,
            "knn_time": 0,
        }

    def update_feature_extraction(self, time_taken):
        self.dataset_times[self.current_dataset]["feature_extraction_time"] = time_taken

    def update_knn(self, time_taken):
        self.dataset_times[self.current_dataset]["knn_time"] = time_taken

    def get_summary(self):
        total_time = time.time() - self.start_time
        summary = {
            "total_time": total_time,
            "datasets": self.dataset_times,
            "total_samples_processed": sum(
                d["total_samples"] for d in self.dataset_times.values()
            ),
        }
        return summary


@torch.cuda.amp.autocast("cuda")
def process_batch(batch_data, device, model):
    """Process a single batch with GPU acceleration"""
    batch, _ = batch_data
    batch = batch.to(device, non_blocking=True)
    with torch.no_grad():
        features = model(batch)
    return features


def extract_features(dataloader, model, device, desc="Extracting features"):
    """Extract features from a dataloader"""
    features = []
    labels = []
    with tqdm(total=len(dataloader), desc=desc) as pbar:
        for batch_data in dataloader:
            images, batch_labels = batch_data
            feat = process_batch((images, batch_labels), device, model)
            features.append(feat)
            labels.append(batch_labels)
            pbar.update(1)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features.cpu().numpy(), labels.cpu().numpy()


def knn_score(train_features, test_features, k=5):
    """Calculate KNN distances using FAISS"""
    index = faiss.IndexFlatL2(train_features.shape[1])
    index.add(train_features)
    distances, _ = index.search(test_features, k)
    return np.sum(distances, axis=1)


def calc_novelty_score(args, ssl_method):
    """Calculate novelty scores for different datasets using KNN distances."""
    progress = ProgressTracker()
    print(
        f"Starting novelty score calculation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    model = ssl_method.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.eval()
    model = model.to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    datasets_dict = {
        "cifar10": {
            "train": load_dataset("cifar10", split="train"),
            "test": load_dataset("cifar10", split="test"),
            "num_classes": 10,
        },
        "cifar100": {
            "train": load_dataset("cifar100", split="train"),
            "test": load_dataset("cifar100", split="test"),
            "num_classes": 100,
        },
    }

    dataset_scores = {}  # Store mean AUROC scores per dataset
    class_scores = {}  # Store individual class AUROC scores
    batch_size = 256 if torch.cuda.is_available() else 128
    k = 5

    for dataset_name, dataset_info in datasets_dict.items():
        print(f"\nProcessing dataset: {dataset_name}")
        dataset_auroc_scores = []

        # For each class as the normal class
        for normal_class in range(dataset_info["num_classes"]):
            try:
                print(f"\nEvaluating with class {normal_class} as normal")
                progress.start_dataset(
                    f"{dataset_name}_class_{normal_class}",
                    len(dataset_info["train"]) + len(dataset_info["test"]),
                )

                # Create training dataset with only normal class
                train_dataset = HFDataset(
                    dataset_info["train"],
                    transform=transform,
                    normal_class=normal_class,
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                )

                # Create test dataset with all classes
                test_dataset = HFDataset(dataset_info["test"], transform=transform)
                test_dataset.normal_class = (
                    normal_class  # Set normal class for label generation
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                )

                # Extract features
                feature_start = time.time()
                print("Extracting training features (normal class only)...")
                train_features, _ = extract_features(train_dataloader, model, device)

                print("Extracting test features (all classes)...")
                test_features, test_labels = extract_features(
                    test_dataloader, model, device
                )

                feature_time = time.time() - feature_start
                progress.update_feature_extraction(feature_time)

                # Calculate KNN distances and AUROC
                knn_start = time.time()
                print("Calculating KNN distances...")
                distances = knn_score(train_features, test_features, k)

                # Calculate AUROC
                auc_score = roc_auc_score(test_labels, distances)

                knn_time = time.time() - knn_start
                progress.update_knn(knn_time)

                dataset_auroc_scores.append(auc_score)
                class_scores[f"{dataset_name}_class_{normal_class}"] = float(auc_score)
                print(f"AUROC score for class {normal_class}: {auc_score:.4f}")

            except Exception as e:
                print(f"Error processing {dataset_name} class {normal_class}: {str(e)}")
                class_scores[f"{dataset_name}_class_{normal_class}"] = None
                continue

        # Calculate mean AUROC score for the dataset
        if dataset_auroc_scores:
            mean_auroc = statistics.mean(dataset_auroc_scores)
            dataset_scores[dataset_name] = mean_auroc
            print(f"\nMean AUROC score for {dataset_name}: {mean_auroc:.4f}")

    progress_summary = progress.get_summary()
    print("\nProcessing Complete!")
    print(f"Total time: {progress_summary['total_time']:.2f} seconds")
    print(f"Total samples processed: {progress_summary['total_samples_processed']:,}")

    print("\nDataset AUROC Score Summary:")
    for dataset_name, score in dataset_scores.items():
        print(f"{dataset_name}: {score:.4f}")

    return dataset_scores, class_scores
