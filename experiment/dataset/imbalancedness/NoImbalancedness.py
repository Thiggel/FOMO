from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class NoImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def get_imbalance(self, class_index: int) -> float:
        # keep 58.313% of the data
        return 1 - 0.58313
