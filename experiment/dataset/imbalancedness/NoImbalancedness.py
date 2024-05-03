from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class NoImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def get_imbalance(self, class_index: int) -> float:
        return 0.0
