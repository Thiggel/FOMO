from typing import Callable
from enum import Enum
from dataclasses import dataclass
from experiment.dataset.imbalancedness.LinearlyIncreasingImbalancedness import (
    LinearlyIncreasingImbalancedness,
)
from experiment.dataset.imbalancedness.NoImbalancedness import NoImbalancedness
from experiment.dataset.imbalancedness.PowerLawImbalancedness import (
    PowerLawImbalancedness,
)
from experiment.dataset.imbalancedness.AllData import AllData


@dataclass
class ImbalanceMethod:
    name: str
    impl: Callable


class ImbalanceMethods(Enum):
    PowerLawImbalance = ImbalanceMethod(
        name="power_law_imbalance",
        impl=PowerLawImbalancedness,
    )

    AllData = ImbalanceMethod(
        name="all_data",
        impl=AllData,
    )

    LinearlyIncreasing = ImbalanceMethod(
        name="linearly_increasing",
        impl=LinearlyIncreasingImbalancedness,
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
