import torch
from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class PowerLawImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def normalize_class_dist(self):
        sum_dist = sum(self.class_dist)
        self.class_dist = [dist / sum_dist for dist in self.class_dist]

    def get_imbalance(self, class_indices: torch.Tensor) -> torch.Tensor:
        """
        Following the imbalance evaluated in Assran et al. 2023,
        imbalance is given by [p(tau)]_k prop 1/(k^{tau}).
        according to this paper this should be close to imbalance in real scenarios

        imbalancedness: 1, 0.84, 0.75 etc

        (I think its fine to not use numclasses here)
        """
        imbalance = 1 / ((class_indices + 1) ** 3)

        if isinstance(class_indices, torch.Tensor):
            print("Class indices", class_indices)
            print("Imbalance: ", imbalance)
            exit()
        return imbalance
