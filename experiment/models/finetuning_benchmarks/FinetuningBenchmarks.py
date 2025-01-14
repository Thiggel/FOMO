from experiment.models.finetuning_benchmarks.CIFAR10FineTuner import CIFAR10FineTuner
from experiment.models.finetuning_benchmarks.CIFAR100FineTuner import CIFAR100FineTuner
from experiment.models.finetuning_benchmarks.CIFAR10KNNClassifier import (
    CIFAR10KNNClassifier,
)
from experiment.models.finetuning_benchmarks.CIFAR100KNNClassifier import (
    CIFAR100KNNClassifier,
)
from experiment.models.finetuning_benchmarks.PetsFineTune import PetsFineTune
from experiment.models.finetuning_benchmarks.CarsFineTune import CarsFineTune
from experiment.models.finetuning_benchmarks.FlowersFineTune import FlowersFineTune
from experiment.models.finetuning_benchmarks.AircraftFineTune import AircraftFineTune
from experiment.models.finetuning_benchmarks.TestFineTuner import TestFineTuner
from experiment.models.finetuning_benchmarks.SecondTestFineTuner import (
    SecondTestFineTuner,
)


class FinetuningBenchmarks:
    benchmarks = [
        CIFAR10FineTuner,
        AircraftFineTune,
        FlowersFineTune,
        CarsFineTune,
        PetsFineTune,
        CIFAR100KNNClassifier,
        CIFAR10KNNClassifier,
        CIFAR100FineTuner,
        TestFineTuner,
        SecondTestFineTuner,
    ]

    test_benchmarks = [TestFineTuner, SecondTestFineTuner]

    @staticmethod
    def get_benchmarks(benchmark_names: list[str]):
        return [
            benchmark
            for benchmark in FinetuningBenchmarks.benchmarks
            if benchmark.__name__ in benchmark_names
        ]

    @staticmethod
    def get_all_benchmark_names():
        return [benchmark.__name__ for benchmark in FinetuningBenchmarks.benchmarks]

    @staticmethod
    def get_default_benchmark_names():
        return [
            benchmark.__name__
            for benchmark in FinetuningBenchmarks.benchmarks
            if benchmark not in FinetuningBenchmarks.test_benchmarks
        ]
