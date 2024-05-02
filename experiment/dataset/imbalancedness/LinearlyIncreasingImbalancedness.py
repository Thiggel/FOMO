from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class LinearlyIncreasingImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def get_imbalance(self, class_index: int) -> float:
        """
        The imbalance score is a linearly increasing function.
        E.g., class 0 has an imbalance score of 0, class 1 has an imbalance
        score of 0.9 * 1/num_classes, etc.
        and class num_classes - 1 has an imbalance score of 0.9.
        """
        return 0.9 * class_index / self.num_classes

