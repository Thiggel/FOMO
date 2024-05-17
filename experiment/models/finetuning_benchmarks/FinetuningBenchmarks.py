from experiment.models.finetuning_benchmarks.CIFAR10FineTuner import CIFAR10FineTuner
from experiment.models.finetuning_benchmarks.CIFAR100FineTuner import CIFAR100FineTuner
from experiment.models.finetuning_benchmarks.CIFAR10KNNClassifier import (
    CIFAR10KNNClassifier,
)
from experiment.models.finetuning_benchmarks.CIFAR100KNNClassifier import (
    CIFAR100KNNClassifier,
)


class FinetuningBenchmarks:
    benchmarks = [
        CIFAR10KNNClassifier,
        CIFAR100KNNClassifier,
        CIFAR10FineTuner,
        CIFAR100FineTuner,
    ]
