# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for COLMAP camera filtering by id and folder name."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from threedgrut.datasets.dataset_colmap import ColmapDataset


def _extr(camera_id: int, name: str) -> SimpleNamespace:
    return SimpleNamespace(camera_id=camera_id, name=name)


def _make_filterable_dataset(
    *,
    cam_intrinsics: dict[int, object],
    cam_extrinsics: list[SimpleNamespace],
    camera_names=None,
    camera_ids=None,
) -> ColmapDataset:
    """Build a ColmapDataset shell that exposes _filter_cameras without touching disk."""
    dataset = ColmapDataset.__new__(ColmapDataset)
    dataset.cam_intrinsics = dict(cam_intrinsics)
    dataset.cam_extrinsics = list(cam_extrinsics)
    dataset.camera_names = list(camera_names) if camera_names is not None else None
    dataset.camera_ids = [int(camera_id) for camera_id in camera_ids] if camera_ids is not None else None
    return dataset


def _multi_rig_dataset(camera_names=None, camera_ids=None) -> ColmapDataset:
    """Three rigs, six interleaved frames: ids 1/2/3 -> folders left/right/aux."""
    cam_intrinsics = {1: object(), 2: object(), 3: object()}
    cam_extrinsics = [
        _extr(1, "left/0001.png"),
        _extr(2, "right/0001.png"),
        _extr(3, "aux/0001.png"),
        _extr(1, "left/0002.png"),
        _extr(2, "right/0002.png"),
        _extr(3, "aux/0002.png"),
    ]
    return _make_filterable_dataset(
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        camera_names=camera_names,
        camera_ids=camera_ids,
    )


def test_filter_cameras_returns_all_frames_when_no_filter_set() -> None:
    dataset = _multi_rig_dataset()
    frame_indices = dataset._filter_cameras()
    assert frame_indices == [0, 1, 2, 3, 4, 5]
    assert set(dataset.cam_intrinsics.keys()) == {1, 2, 3}
    assert [extr.name for extr in dataset.cam_extrinsics] == [
        "left/0001.png",
        "right/0001.png",
        "aux/0001.png",
        "left/0002.png",
        "right/0002.png",
        "aux/0002.png",
    ]


def test_filter_cameras_by_camera_id_keeps_matching_frames_only() -> None:
    dataset = _multi_rig_dataset(camera_ids=[2])
    frame_indices = dataset._filter_cameras()
    assert frame_indices == [1, 4]
    assert set(dataset.cam_intrinsics.keys()) == {2}
    assert [extr.camera_id for extr in dataset.cam_extrinsics] == [2, 2]


def test_filter_cameras_by_folder_name_keeps_matching_frames_only() -> None:
    dataset = _multi_rig_dataset(camera_names=["right"])
    frame_indices = dataset._filter_cameras()
    assert frame_indices == [1, 4]
    assert [extr.name for extr in dataset.cam_extrinsics] == ["right/0001.png", "right/0002.png"]


def test_filter_cameras_intersects_id_and_name_filters() -> None:
    dataset = _multi_rig_dataset(camera_names=["right"], camera_ids=[2, 3])
    frame_indices = dataset._filter_cameras()
    assert frame_indices == [1, 4]
    assert set(dataset.cam_intrinsics.keys()) == {2}


def test_filter_cameras_empty_intersection_raises() -> None:
    dataset = _multi_rig_dataset(camera_names=["left"], camera_ids=[2])
    with pytest.raises(ValueError, match="empty"):
        dataset._filter_cameras()


def test_filter_cameras_unknown_camera_id_raises_with_available_list() -> None:
    dataset = _multi_rig_dataset(camera_ids=[42])
    with pytest.raises(ValueError, match=r"camera_ids \[42\].*Available camera_ids: \[1, 2, 3\]"):
        dataset._filter_cameras()


def test_filter_cameras_unknown_camera_name_raises_with_available_list() -> None:
    dataset = _multi_rig_dataset(camera_names=["middle"])
    with pytest.raises(ValueError, match=r"camera_names \['middle'\].*Available camera_names: \['aux', 'left', 'right'\]"):
        dataset._filter_cameras()


def test_filter_cameras_falls_back_to_synthetic_names_when_extrinsic_has_no_folder() -> None:
    """When extrinsics expose bare basenames (no folder), names default to camera_<idx>."""
    cam_intrinsics = {7: object(), 11: object()}
    cam_extrinsics = [_extr(7, "0001.png"), _extr(11, "0002.png")]
    dataset = _make_filterable_dataset(
        cam_intrinsics=cam_intrinsics,
        cam_extrinsics=cam_extrinsics,
        camera_names=["camera_1"],
    )
    frame_indices = dataset._filter_cameras()
    assert frame_indices == [1]
    assert set(dataset.cam_intrinsics.keys()) == {11}
