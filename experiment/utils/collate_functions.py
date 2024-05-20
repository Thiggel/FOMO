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
