from experiment.utils.get_training_args import get_training_args
from experiment.__main__ import run_different_seeds
from experiment.utils.print_mean_std import get_mean_std


def test_without_pretraining():
    training_args = get_training_args(get_defaults=True)

    training_args.finetuning_benchmarks = ["TestFineTuner", "SecondTestFineTuner"]
    training_args.pretrain = False
    training_args.test_mode = True

    all_results = run_different_seeds(training_args)
    mean_std = get_mean_std(all_results)

    expected_results = {
        "test1_test_loss": {"mean": 0.0, "std": 0.0},
        "test2_test_loss": {"mean": 0.0, "std": 0.0},
    }

    assert "training_time" in mean_std, f"Expected: 'training_time' in {mean_std}"

    mean_std.pop("training_time")

    assert (
        mean_std == expected_results
    ), f"Expected: {expected_results}, Got: {mean_std}"


def test_full_pipeline():
    training_args = get_training_args(get_defaults=True)

    training_args.ssl_method = "TestSSLMethod"
    training_args.finetuning_benchmarks = ["TestFineTuner", "SecondTestFineTuner"]
    training_args.max_cycles = 1
    training_args.n_epochs_per_cycle = 1
    training_args.test_mode = True

    all_results = run_different_seeds(training_args)
    mean_std = get_mean_std(all_results)

    expected_results = {
        "test1_test_loss": {"mean": 0.0, "std": 0.0},
        "test2_test_loss": {"mean": 0.0, "std": 0.0},
    }

    assert "training_time" in mean_std, f"Expected: 'training_time' in {mean_std}"

    mean_std.pop("training_time")

    assert (
        mean_std == expected_results
    ), f"Expected: {expected_results}, Got: {mean_std}"
