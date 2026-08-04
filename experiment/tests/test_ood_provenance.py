from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, Subset

from experiment.ood.ood import OOD
from experiment.ImbalancedTraining import ImbalancedTraining


class GrowingDataset(Dataset):
    """Minimal stand-in for ImbalancedDataset with synthetic tail entries."""

    def __init__(self):
        self.indices = list(range(5))
        self.dataset = object()

    def __len__(self):
        return 8

    def __getitem__(self, index):
        return torch.tensor(index), 0


def make_ood(dataset):
    args = SimpleNamespace(
        val_batch_size=2,
        k=1,
        num_ood_samples=1,
        every_nth_ood_sample=1,
    )
    return OOD(args, dataset, feature_extractor=lambda batch: batch)


def test_selection_provenance_stops_at_growing_dataset():
    dataset = Subset(GrowingDataset(), [0, 2, 5, 7])

    underlying, original = make_ood(dataset)._selection_provenance([0, 2, 3])

    assert underlying == [0, 5, 7]
    assert original == [True, False, False]


def test_selection_provenance_resolves_nested_subsets():
    base = GrowingDataset()
    training = Subset(base, [0, 2, 5, 7])
    candidates = Subset(training, [1, 2, 3])

    underlying, original = make_ood(candidates)._selection_provenance([0, 1, 2])

    assert underlying == [2, 5, 7]
    assert original == [True, False, False]


def test_oracle_manifest_resolves_positions_in_oracle_anchor_dataset(tmp_path):
    base = GrowingDataset()
    training = Subset(base, [0, 2, 4])
    runner = ImbalancedTraining.__new__(ImbalancedTraining)
    runner.datamodule = SimpleNamespace(train_dataset=training)
    runner.args = SimpleNamespace(
        additional_data_path=str(tmp_path),
        num_generations_per_ood_sample=5,
    )
    images = [(torch.zeros(3, 4, 4), 0), (torch.zeros(3, 4, 4), 0)]

    runner._write_repair_manifest(
        cycle_idx=0,
        selected_positions=[5, 7],
        labels=[0, 0],
        repair_operator="oracle_real_restoration",
        anchor_samples=images,
        dataset=base,
    )

    assert (tmp_path / "repair_manifests" / "cycle_0.json").is_file()


def test_bridge_tada_merge_preserves_budget_and_deduplicates():
    merged = ImbalancedTraining._merge_bridge_tada_indices(
        bridge_indices=[1, 2, 3, 4, 5],
        tada_ranked_indices=[2, 3, 6, 7, 8, 9],
        budget=6,
    )

    assert len(merged) == 6
    assert len(set(merged)) == 6
    assert merged[:3] == [1, 2, 3]
    assert {6, 7, 8}.issubset(merged)


def test_bridge_tada_merge_handles_zero_budget():
    assert (
        ImbalancedTraining._merge_bridge_tada_indices(
            bridge_indices=[1],
            tada_ranked_indices=[2],
            budget=0,
        )
        == []
    )


def test_bridge_tada_outlier_policy_uses_both_rankings():
    runner = ImbalancedTraining.__new__(ImbalancedTraining)
    runner.args = SimpleNamespace(
        sample_selection="bridge_tada",
        num_ood_samples=4,
    )
    runner.datamodule = SimpleNamespace(train_dataset=list(range(10)))
    runner._compute_sample_losses = lambda positions: torch.arange(
        len(positions), dtype=torch.float32
    )

    selected = runner.get_outliers(
        cycle_idx=0,
        precomputed_ood_indices=[0, 1, 2, 3],
    )

    assert selected == [0, 1, 9, 8]
