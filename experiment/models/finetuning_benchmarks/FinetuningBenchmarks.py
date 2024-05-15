from experiment.models.finetuning_benchmarks.CIFAR10FineTuner import CIFAR10FineTuner
from experiment.models.finetuning_benchmarks.CIFAR100FineTuner import CIFAR100FineTuner


class FinetuningBenchmarks:
    benchmarks = [
        CIFAR10FineTuner,
        CIFAR100FineTuner,
    ]
