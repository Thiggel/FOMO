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
from experiment.models.finetuning_benchmarks.CIFAR100LTFineTune import (
    CIFAR100LTFineTune,
)



class FinetuningBenchmarks:
    benchmarks = [
        CarsFineTune,
        CarsKNNClassifier,
        AircraftFineTune,
        AircraftKNNClassifier,
        FlowersKNNClassifier,
        PetsKNNClassifier,
        CIFAR100KNNClassifier,
        CIFAR10FineTuner,
        FlowersFineTune,
        PetsFineTune,
        CIFAR10KNNClassifier,
        CIFAR100FineTuner,
        ImageNet100LTKNNClassifier,
        ImageNet100LTFineTune,
        CIFAR100LTFineTune,
    ]

    test_benchmarks = []

    # Paper-facing comparisons must report each downstream dataset separately
    # under both linear probing and kNN.  Keeping the contract here prevents
    # individual launch scripts from silently evaluating different subsets.
    benchmark_suites = {
        "paper_full": [
            CarsFineTune,
            CarsKNNClassifier,
            AircraftFineTune,
            AircraftKNNClassifier,
            FlowersFineTune,
            FlowersKNNClassifier,
            PetsFineTune,
            PetsKNNClassifier,
            CIFAR10FineTuner,
            CIFAR10KNNClassifier,
            CIFAR100FineTuner,
            CIFAR100KNNClassifier,
            ImageNet100LTFineTune,
            ImageNet100LTKNNClassifier,
        ],
    }

    suite_result_metrics = {
        "paper_full": [
            "cars_test_accuracy",
            "carsknn_knn_test_accuracy",
            "aircraft_test_accuracy",
            "aircraftknn_knn_test_accuracy",
            "flowers_test_accuracy",
            "flowersknn_knn_test_accuracy",
            "pets_test_accuracy",
            "petsknn_knn_test_accuracy",
            "cifar10r_test_accuracy",
            "cifar10knn_knn_test_accuracy",
            "cifar100r_test_accuracy",
            "cifar100knn_knn_test_accuracy",
            "imagenet100lt_test_accuracy",
            "imagenet100ltknn_knn_test_accuracy",
        ],
    }

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

    @staticmethod
    def get_benchmark_suite_names(suite_name):
        if suite_name not in FinetuningBenchmarks.benchmark_suites:
            available = ", ".join(sorted(FinetuningBenchmarks.benchmark_suites))
            raise ValueError(
                f"Unknown finetune benchmark suite: {suite_name}. "
                f"Available suites: {available}"
            )
        return [
            benchmark.__name__
            for benchmark in FinetuningBenchmarks.benchmark_suites[suite_name]
        ]

    @staticmethod
    def get_benchmark_suite_result_metrics(suite_name):
        FinetuningBenchmarks.get_benchmark_suite_names(suite_name)
        return list(FinetuningBenchmarks.suite_result_metrics[suite_name])
