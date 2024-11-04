from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class AllData(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.total_imbalance = self.get_total_imbalance()

    def get_total_imbalance(self) -> float:
        return 0.0

    def get_imbalance(self, class_index: int) -> float:
        return self.total_imbalance
