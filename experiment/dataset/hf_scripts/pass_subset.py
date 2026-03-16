"""PASS subset dataset loader with current Zenodo URLs.

This loader intentionally pulls only `PASS.0.tar` to keep the subset practical.
With `split=train[:100000]`, this is sufficient for the configured PASS runs.
"""

import os
from datetime import datetime
from typing import Optional

import datasets
import numpy as np
import pandas as pd
import requests
from filelock import FileLock


_DESCRIPTION = """\
PASS (Pictures without humAns for Self-Supervision) subset.
"""

_CITATION = """\
@Article{asano21pass,
author = "Yuki M. Asano and Christian Rupprecht and Andrew Zisserman and Andrea Vedaldi",
title = "PASS: An ImageNet replacement for self-supervised pretraining without humans",
journal = "NeurIPS Track on Datasets and Benchmarks",
year = "2021"
}
"""

_HOMEPAGE = "https://www.robots.ox.ac.uk/~vgg/research/pass/"
_LICENSE = "Creative Commons Attribution 4.0 International"

_METADATA_DOWNLOAD_URL = "https://zenodo.org/records/6615455/files/pass_metadata.csv?download=1"
_IMAGE_ARCHIVE_DOWNLOAD_URLS = [
    "https://zenodo.org/records/6615455/files/PASS.0.tar?download=1",
]
_DOWNLOAD_TIMEOUT = 60


def _download_dir() -> str:
    base_dir = (
        os.environ.get("FOMO_DATASET_TMPDIR")
        or os.environ.get("TMPDIR")
        or os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("BASE_CACHE_DIR")
    )
    if not base_dir:
        base_dir = os.path.join(os.path.expanduser("~"), ".cache")

    path = os.path.join(base_dir, "pass_subset_downloads")
    os.makedirs(path, exist_ok=True)
    return path


def _download_to_cache(url: str, filename: str) -> str:
    download_dir = _download_dir()
    output_path = os.path.join(download_dir, filename)
    lock = FileLock(output_path + ".lock")

    with lock:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        tmp_path = output_path + ".tmp"
        try:
            response = requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT)
        except requests.exceptions.SSLError:
            response = requests.get(
                url,
                stream=True,
                timeout=_DOWNLOAD_TIMEOUT,
                verify=False,
            )

        with response:
            response.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        os.replace(tmp_path, output_path)

    return output_path


def _parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class PASSSUBSET(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.Value("int64"),
                    "creator_username": datasets.Value("string"),
                    "hash": datasets.Value("string"),
                    "gps_latitude": datasets.Value("float32"),
                    "gps_longitude": datasets.Value("float32"),
                    "date_taken": datasets.Value("timestamp[us]"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        metadata_file = _download_to_cache(
            _METADATA_DOWNLOAD_URL,
            "pass_metadata.csv",
        )
        image_archives = [
            _download_to_cache(url, f"PASS.{idx}.tar")
            for idx, url in enumerate(_IMAGE_ARCHIVE_DOWNLOAD_URLS)
        ]
        metadata = pd.read_csv(metadata_file, encoding="utf-8")
        metadata = metadata.replace(np.nan, pd.NA).where(metadata.notnull(), None)
        metadata = metadata.set_index("hash")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata": metadata,
                    "image_archives": [dl_manager.iter_archive(path) for path in image_archives],
                },
            )
        ]

    def _generate_examples(self, metadata, image_archives):
        for image_archive in image_archives:
            for path, file_obj in image_archive:
                image_hash = os.path.basename(path).split(".")[0]
                if image_hash not in metadata.index:
                    continue

                image_meta = metadata.loc[image_hash]
                yield image_hash, {
                    "image": {"path": path, "bytes": file_obj.read()},
                    "label": 0,
                    "creator_username": image_meta["unickname"],
                    "hash": image_hash,
                    "gps_latitude": image_meta["latitude"],
                    "gps_longitude": image_meta["longitude"],
                    "date_taken": _parse_timestamp(image_meta["datetaken"]),
                }
