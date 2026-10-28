# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the frame-to-camera mapping of the NCore dataset."""

from __future__ import annotations

import numpy as np
import pytest

from threedgrut.datasets.datasetNcore import NCoreDataset


def _make_dataset(frames_per_camera: list[int]) -> NCoreDataset:
    """Build an NCoreDataset shell exposing get_camera_idx without loading any sequence."""
    dataset = NCoreDataset.__new__(NCoreDataset)
    dataset.get_n_frames_per_camera = lambda unique_sensors=True: np.array(frames_per_camera, dtype=np.int32)
    return dataset


def test_frames_are_assigned_camera_by_camera():
    dataset = _make_dataset([2, 3, 1])
    assert [dataset.get_camera_idx(i) for i in range(6)] == [0, 0, 1, 1, 1, 2]


def test_cameras_without_frames_are_skipped():
    dataset = _make_dataset([0, 2, 0, 1])
    assert [dataset.get_camera_idx(i) for i in range(3)] == [1, 1, 3]


@pytest.mark.parametrize("frame_idx", [-1, 6])
def test_out_of_range_frame_index_raises(frame_idx):
    with pytest.raises(IndexError):
        _make_dataset([2, 3, 1]).get_camera_idx(frame_idx)
