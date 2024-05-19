from experiment.dataset.imbalancedness.Imbalancedness import Imbalancedness


class PowerLawImbalancedness(Imbalancedness):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def get_imbalance(self, class_index: int) -> float:
        """
        Following the imbalance evaluated in Assran et al. 2023,
        imbalance is given by [p(tau)]_k prop 1/(k^{tau}).
        according to this paper this should be close to imbalance in real scenarios

        imbalancedness: 1, 0.84, 0.75 etc

        (I think its fine to not use numclasses here)
        """
        return 1 / ((class_index + 1) ** (0.25))
