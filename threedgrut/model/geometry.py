# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import sklearn.neighbors
import torch

from threedgrut.utils.misc import to_np


def nearest_neighbors(pts_src, k=2):

    pts_src_np = to_np(pts_src)

    # distance from a point set to itself
    pts_target = pts_src
    pts_target_np = pts_src_np

    # Build the tree
    kd_tree = sklearn.neighbors.KDTree(pts_target_np)

    # Query it 
    _, neighbors = kd_tree.query(pts_src_np, k=k)

    # Mask out self element
    mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

    # make sure we mask out exactly one element in each row, in rare case of many duplicate points
    mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
    neighbors = neighbors[mask].reshape((neighbors.shape[0], k - 1))

    # recompute distances in torch, so the function is differentiable
    neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
    return neigh_inds

def nearest_neighbor_dist_cpuKD(pts_src, pts_target=None):
    """
    Compute the distance to the nearest neighbor, using a CPU kd-tree
    Passing one arg computes from a point set to itself,
    to args computes distance from each point in src to target
    """

    pts_src_np = to_np(pts_src)

    if pts_target is None:
        # distance from a point set to itself
        on_self = True
        k = 2
        pts_target = pts_src
        pts_target_np = pts_src_np
    else:
        # distance between two point sets
        on_self = False
        k = 1
        pts_target_np = to_np(pts_target)


    # Build the tree
    kd_tree = sklearn.neighbors.KDTree(pts_target_np)

    # Query it 
    _, neighbors = kd_tree.query(pts_src_np, k=k)

    # Mask out self element
    if on_self:
        mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

        # make sure we mask out exactly one element in each row, in rare case of many duplicate points
        mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
        neighbors = neighbors[mask].reshape((neighbors.shape[0],))
    else:
        neighbors = neighbors[:,0]

    # recompute distances in torch, so the function is differentiable
    neigh_inds = torch.tensor(neighbors, device=pts_src.device, dtype=torch.int64)
    dists = torch.linalg.norm(pts_src - pts_target[neigh_inds,:], dim=-1)

    return dists

def safe_normalize(vecs):
    norms = torch.linalg.norm(vecs, dim=-1)
    norms = torch.where(norms > 0., norms, 1.)
    return vecs / norms[...,None]
