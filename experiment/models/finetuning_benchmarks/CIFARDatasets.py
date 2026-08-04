from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from datasets import load_dataset


class HuggingFaceCIFAR(Dataset):
    """CIFAR fallback for clusters that cannot reliably reach Toronto."""

    def __init__(self, num_classes: int, train: bool, transform=None):
        if num_classes not in (10, 100):
            raise ValueError(f"Unsupported CIFAR class count: {num_classes}")
        dataset_id = f"uoft-cs/cifar{num_classes}"
        split = "train" if train else "test"
        self.dataset = load_dataset(dataset_id, split=split)
        self.image_key = "img"
        self.label_key = "label" if num_classes == 10 else "fine_label"
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[int(index)]
        image = sample[self.image_key].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(sample[self.label_key])


def get_cifar_dataset(
    num_classes: int,
    root: str,
    train: bool,
    transform=None,
):
    """Prefer an already verified torchvision copy, then use the HF mirror.

    Calling torchvision with ``download=True`` can spend almost an hour on the
    throttled Toronto host before failing.  Cluster setup stages either the
    extracted torchvision data or the official Hugging Face mirror, so a
    missing/invalid local archive should fall through immediately.
    """

    dataset_type = CIFAR10 if num_classes == 10 else CIFAR100
    try:
        return dataset_type(
            root=root,
            train=train,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        return HuggingFaceCIFAR(
            num_classes=num_classes,
            train=train,
            transform=transform,
        )
