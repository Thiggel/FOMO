import torch
from torch import Tensor

def simclr_collate(batch: list) -> tuple[list[Tensor], Tensor]:
        num_images = len(batch[0][0])

        outer_list = []
        for i in range(num_images):
            data = torch.stack(
                [
                    (
                        item[0][i].repeat(3, 1, 1)
                        if item[0][i].size(0) == 1
                        else item[0][i][:3]
                    )
                    for item in batch
                ]
            )

            outer_list.append(data)

        data = tuple(outer_list)
        labels = [item[1] for item in batch]

        stacked_labels = torch.tensor(labels)

        return data, stacked_labels