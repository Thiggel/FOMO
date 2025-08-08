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
from experiment.models.finetuning_benchmarks.PetsKNNClassifier import PetsKNNClassifier
from experiment.models.finetuning_benchmarks.AircraftKNNClassifier import (
    AircraftKNNClassifier,
)
from experiment.models.finetuning_benchmarks.CarsKNNClassifier import CarsKNNClassifier
from experiment.models.finetuning_benchmarks.FlowersKNNClassifier import (
    FlowersKNNClassifier,
)
from experiment.models.finetuning_benchmarks.ImageNet100FineTune import (
    ImageNet100FineTune,
)
from experiment.models.finetuning_benchmarks.ImageNet100LTFineTune import (
    ImageNet100LTFineTune,
)
from experiment.models.finetuning_benchmarks.ImageNet100KNNClassifier import (
    ImageNet100KNNClassifier,
)
from experiment.models.finetuning_benchmarks.ImageNet100LTKNNClassifier import (
    ImageNet100LTKNNClassifier,
)


class FinetuningBenchmarks:
    benchmarks = [
        ImageNet100LTKNNClassifier,
        ImageNet100LTFineTune,
        #AircraftFineTune,
        #CarsFineTune,
        #CarsKNNClassifier,
        #AircraftKNNClassifier,
        #FlowersKNNClassifier,
        #PetsKNNClassifier,
        #CIFAR100KNNClassifier,
        #ImageNet100KNNClassifier,
        #CIFAR10FineTuner,
        #FlowersFineTune,
        #PetsFineTune,
        #ImageNet100FineTune,
        #CIFAR10KNNClassifier,
        #CIFAR100FineTuner,
    ]

    test_benchmarks = []

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
