from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class ExponentiallyIncreasingImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def get_imbalance(self, class_index: int) -> float:
        """
        The imbalance score is an exponentially increasing function.
        E.g., class 0 has an imbalance score of about 0.1
        and class num_classes - 1 has an imbalance score of 0.9.
        """
        return 0.9 * 10 ** (class_index / self.num_classes - 1)

