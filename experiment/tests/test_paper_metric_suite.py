from types import SimpleNamespace

from experiment.ImbalancedTraining import ImbalancedTraining
from experiment.models.finetuning_benchmarks.FinetuningBenchmarks import (
    FinetuningBenchmarks,
)
from scripts.report_full_metric_suite import summarize


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


def test_inventory_metrics_match_the_paper_suite():
    # scripts/experiment_inventory.py duplicates the metric list so it can run
    # on login nodes without torch.  The duplicate must not drift.
    from scripts.experiment_inventory import PAPER_METRICS

    assert set(PAPER_METRICS) == EXPECTED_PAPER_METRICS
    assert len(PAPER_METRICS) == len(EXPECTED_PAPER_METRICS)


def test_benchmark_trainer_never_spans_multiple_devices():
    # A multi-device benchmark trainer makes Lightning spawn a second
    # generation of DDP workers, which kills the job after training has
    # already completed.  See _use_single_device_for_benchmarks.
    trainer_args = {
        "strategy": "ddp",
        "num_nodes": 2,
        "accelerator": "cpu",
        "devices": "auto",
        "max_epochs": 7,
    }

    configured = ImbalancedTraining._use_single_device_for_benchmarks(trainer_args)

    assert configured["devices"] == 1
    assert "strategy" not in configured
    assert "num_nodes" not in configured
    assert configured["accelerator"] == "cuda"
    assert configured["max_epochs"] == 7


def test_full_metric_report_uses_percent_mean_and_population_std():
    assert summarize([0.10, 0.12, 0.14]) == "12.00 ± 1.63"
