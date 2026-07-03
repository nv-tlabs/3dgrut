# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("ncore")

from threedgrut.datasets.dataset_colmap import ColmapDataset
from threedgrut.datasets.protocols import get_dataset_world_transform


class _TransformProvider:
    def __init__(self, transform) -> None:
        self.transform = transform

    def get_world_normalization_transform(self):
        return self.transform


def test_dataset_world_transform_filters_identity_and_copies_nonidentity() -> None:
    assert get_dataset_world_transform(object()) is None
    assert get_dataset_world_transform(_TransformProvider(np.eye(4, dtype=np.float32))) is None

    source = np.eye(4, dtype=np.float32)
    source[:3, 3] = [1.0, 2.0, 3.0]
    result = get_dataset_world_transform(_TransformProvider(source))

    np.testing.assert_array_equal(result, source)
    assert result is not source


@pytest.mark.parametrize(
    "transform,match",
    [
        (np.eye(3), r"shape \(4, 4\)"),
        (np.full((4, 4), np.nan), "finite"),
        (np.eye(4, dtype=np.complex64), "real-valued"),
    ],
)
def test_dataset_world_transform_rejects_invalid_values(transform, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        get_dataset_world_transform(_TransformProvider(transform))


def test_reload_resets_stale_world_transform_when_normalization_is_disabled(monkeypatch) -> None:
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.normalize_world_space = False
    dataset.world_normalization_transform = np.full((4, 4), 7.0, dtype=np.float32)
    dataset._worker_gpu_cache = {}
    dataset._all_exif_exposures = None
    dataset.test_split_interval = 0
    dataset.split = "train"

    def load_intrinsics_and_extrinsics() -> None:
        dataset.cam_intrinsics = {3: object()}
        dataset.cam_extrinsics = [SimpleNamespace(camera_id=3)]

    def load_camera_data() -> None:
        dataset.poses = np.eye(4, dtype=np.float32)[None]
        dataset.image_paths = np.array(["image.png"])
        dataset.mask_paths = np.array(["mask.png"])
        dataset.camera_centers = np.zeros((1, 3), dtype=np.float32)

    monkeypatch.setattr(dataset, "load_intrinsics_and_extrinsics", load_intrinsics_and_extrinsics)
    monkeypatch.setattr(dataset, "_filter_cameras", lambda: [0])
    monkeypatch.setattr(dataset, "load_camera_data", load_camera_data)
    monkeypatch.setattr(
        dataset,
        "compute_spatial_extents",
        lambda: (torch.zeros(3), torch.tensor(0.0), (torch.zeros(3), torch.zeros(3))),
    )

    dataset.reload()

    np.testing.assert_array_equal(dataset.get_world_normalization_transform(), np.eye(4, dtype=np.float32))


def test_world_transform_getter_returns_a_copy() -> None:
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.world_normalization_transform = np.eye(4, dtype=np.float32)

    result = dataset.get_world_normalization_transform()
    result[0, 0] = 9.0

    assert dataset.world_normalization_transform[0, 0] == 1.0
