from typing import Callable
from enum import Enum
from dataclasses import dataclass
from experiment.dataset.imbalancedness.LinearlyIncreasingImbalancedness import (
    LinearlyIncreasingImbalancedness,
)
from experiment.dataset.imbalancedness.ExponentiallyIncreasingImbalancedness import (
    ExponentiallyIncreasingImbalancedness,
)
from experiment.dataset.imbalancedness.NoImbalancedness import NoImbalancedness


@dataclass
class ImbalanceMethod:
    name: str
    impl: Callable


class ImbalanceMethods(Enum):
    LinearlyIncreasing = ImbalanceMethod(
        name="linearly_increasing",
        impl=LinearlyIncreasingImbalancedness,
    )

    ExponentiallyIncreasing = ImbalanceMethod(
        name="exponentially_increasing",
        impl=ExponentiallyIncreasingImbalancedness,
    )

    NoImbalance = ImbalanceMethod(
        name="no_imbalance",
        impl=NoImbalancedness,
    )

    @staticmethod
    def init_method(variant: str):
        return {member.value.name: member for member in ImbalanceMethods}[variant]

    @staticmethod
    def get_methods():
        return [member.value.name for member in ImbalanceMethods]

    @staticmethod
    def get_default_method():
        return ImbalanceMethods.get_methods()[0]
