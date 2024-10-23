import torch
from torch import Tensor


def simclr_collate(batch: list) -> tuple[list[Tensor], Tensor]:
    num_images = len(batch[0][0])

    outer_list = []
    for i in range(num_images):
        data = torch.stack([item[0][i] for item in batch])

        outer_list.append(data)

    data = tuple(outer_list)
    labels = [item[1] for item in batch]

    stacked_labels = torch.tensor(labels)

    return data, stacked_labels


def dino_collate(batch: list) -> tuple[list[Tensor], Tensor]:
    """
    Collate function for DINO that handles multiple crops.
    Each item in batch contains [crops, label] where crops is a list of tensors.
    """
    all_crops = []
    num_crops = len(batch[0][0])  # Number of crops (2 global + n local)

    # For each crop position
    for i in range(num_crops):
        # Stack all batch items for this crop position
        crop_batch = torch.stack([item[0][i] for item in batch])
        all_crops.append(crop_batch)

    # Stack labels
    labels = torch.tensor([item[1] for item in batch])

    return all_crops, labels
