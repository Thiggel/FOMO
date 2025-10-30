import io
import sys
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, Features, Value
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.dataset.ImbalancedDataset import ImbalancedDataset
from experiment.dataset.imbalancedness.ImbalanceMethods import ImbalanceMethods


def _create_dataset(data: dict[str, list[Any]], features: Features) -> Dataset:
    return Dataset.from_dict(data, features=features)


def _make_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    image = Image.new("RGB", (16, 16), color=color)
    image.save(path)


@pytest.fixture(autouse=True)
def _set_base_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_unlabeled_dataset_uses_all_data(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    _make_image(image_path)

    dataset = _create_dataset(
        {"image_path": [str(image_path)]},
        Features({"image_path": Value("string")}),
    )

    monkeypatch.setattr(
        "experiment.dataset.ImbalancedDataset.load_dataset",
        lambda *args, **kwargs: dataset,
    )

    imbalanced_dataset = ImbalancedDataset(
        dataset_path="dummy/dataset",
        additional_data_path=str(tmp_path / "generated"),
        imbalance_method=ImbalanceMethods.AllData,
        split="train",
        x_key="image_path",
        y_key=None,
        checkpoint_filename="unlabeled",
    )

    sample, label = imbalanced_dataset[0]

    assert label == 0
    assert sample.size == (16, 16)
    assert imbalanced_dataset.num_classes == 1
    assert len(imbalanced_dataset) == 1


def test_string_labels_are_encoded(tmp_path, monkeypatch):
    image_paths = []
    for idx, color in enumerate([(255, 0, 0), (0, 255, 0), (255, 0, 0)]):
        path = tmp_path / f"sample_{idx}.png"
        _make_image(path, color=color)
        image_paths.append(str(path))

    dataset = _create_dataset(
        {
            "image_path": image_paths,
            "category": ["cat", "dog", "cat"],
        },
        Features({
            "image_path": Value("string"),
            "category": Value("string"),
        }),
    )

    monkeypatch.setattr(
        "experiment.dataset.ImbalancedDataset.load_dataset",
        lambda *args, **kwargs: dataset,
    )

    imbalanced_dataset = ImbalancedDataset(
        dataset_path="dummy/dataset",
        additional_data_path=str(tmp_path / "generated"),
        imbalance_method=ImbalanceMethods.AllData,
        split="train",
        x_key="image_path",
        y_key="category",
        checkpoint_filename="string_labels",
    )

    labels = [imbalanced_dataset[i][1] for i in range(len(imbalanced_dataset))]

    assert labels == [0, 1, 0]
    assert imbalanced_dataset.get_class_name(0) == "cat"
    assert imbalanced_dataset.get_class_name(1) == "dog"
    assert imbalanced_dataset.num_classes == 2


def test_url_images_are_cached(tmp_path, monkeypatch):
    dataset = _create_dataset(
        {"url": ["https://example.com/image.png"]},
        Features({"url": Value("string")}),
    )

    monkeypatch.setattr(
        "experiment.dataset.ImbalancedDataset.load_dataset",
        lambda *args, **kwargs: dataset,
    )

    image_bytes = io.BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(image_bytes, format="PNG")
    image_bytes.seek(0)

    class DummyResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:  # pragma: no cover - simple stub
            return None

    calls = {"count": 0}

    def fake_get(url: str, timeout: int):
        calls["count"] += 1
        return DummyResponse(image_bytes.getvalue())

    monkeypatch.setattr("experiment.dataset.ImbalancedDataset.requests.get", fake_get)

    imbalanced_dataset = ImbalancedDataset(
        dataset_path="dummy/dataset",
        additional_data_path=str(tmp_path / "generated"),
        imbalance_method=ImbalanceMethods.AllData,
        split="train",
        x_key="url",
        y_key=None,
        checkpoint_filename="cached",
    )

    _ = imbalanced_dataset[0]
    assert calls["count"] == 1

    _ = imbalanced_dataset[0]
    assert calls["count"] == 1

    cache_root = Path(tmp_path) / "hf_image_cache"
    assert any(cache_root.rglob("*.png"))
