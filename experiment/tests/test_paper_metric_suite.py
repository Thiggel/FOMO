from types import SimpleNamespace

from experiment.ImbalancedTraining import ImbalancedTraining
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)


EXPECTED_PAPER_BENCHMARKS = {
    "CarsFineTune",
    "CarsKNNClassifier",
    "AircraftFineTune",
    "AircraftKNNClassifier",
    "FlowersFineTune",
    "FlowersKNNClassifier",
    "PetsFineTune",
    "PetsKNNClassifier",
    "CIFAR10FineTuner",
    "CIFAR10KNNClassifier",
    "CIFAR100FineTuner",
    "CIFAR100KNNClassifier",
    "ImageNet100LTFineTune",
    "ImageNet100LTKNNClassifier",
}

EXPECTED_PAPER_METRICS = {
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
}


def test_paper_full_suite_has_linear_probe_and_knn_for_every_dataset():
    names = FinetuningBenchmarks.get_benchmark_suite_names("paper_full")
    assert len(names) == 14
    assert set(names) == EXPECTED_PAPER_BENCHMARKS
    assert set(
        FinetuningBenchmarks.get_benchmark_suite_result_metrics("paper_full")
    ) == EXPECTED_PAPER_METRICS


def test_latest_checkpoint_is_overwritten_after_each_cycle(tmp_path):
    saved_paths = []

    class FakeTrainer:
        def save_checkpoint(self, path):
            saved_paths.append(path)
            with open(path, "w") as handle:
                handle.write("latest")

    training = ImbalancedTraining.__new__(ImbalancedTraining)
    training.checkpoint_callback = SimpleNamespace(dirpath=str(tmp_path))
    training._persist_latest_checkpoint(FakeTrainer(), cycle_idx=3)

    assert saved_paths == [str(tmp_path / "last.ckpt")]
    assert (tmp_path / "last.ckpt").read_text() == "latest"
