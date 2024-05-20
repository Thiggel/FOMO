from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness
from experiment.dataset.imbalancedness.PowerLawImbalancedness import (
    PowerLawImbalancedness,
)


class NoImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

        self.power_law_imbalance = PowerLawImbalancedness(num_classes)
        self.total_imbalance = self.get_total_imbalance()

    def get_power_law_imbalance(self, class_index: int) -> float:
        return self.power_law_imbalance.get_imbalance(class_index)

    def get_total_imbalance(self) -> float:
        """
        We want the same amount of data in both the imbalanced
        and balanced case. Therefore, we calculate the average
        imbalance across classes and take away this data in
        a uniform way.
        """
        total_imbalance = 0
        for class_idx in range(self.num_classes):
            total_imbalance += self.get_power_law_imbalance(class_idx)

        return total_imbalance / self.num_classes

    def get_imbalance(self, class_index: int) -> float:
        return self.total_imbalance
