
from utils import to_np

import numpy as np
import sklearn.neighbors
import torch

def nearest_neighbor_dist_cpuKD(pts_torch):
    """Compute the distance to the nearest neighbor, using a CPU kd-tree"""

    pts = to_np(pts_torch)

    # Build the tree
    kd_tree = sklearn.neighbors.KDTree(pts)

    # Query it 
    _, neighbors = kd_tree.query(pts, k=2)

    # Mask out self element
    mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

    # make sure we mask out exactly one element in each row, in rare case of many duplicate points
    mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False
    neighbors = neighbors[mask].reshape((neighbors.shape[0],))

    # recompute distances in torch, so the function is differentiable
    neigh_inds = torch.tensor(neighbors, device=pts_torch.device, dtype=torch.int64)
    dists = torch.linalg.norm(pts_torch - pts_torch[neigh_inds], dim=-1)

    return dists


def safe_normalize(vecs):
    norms = torch.linalg.norm(vecs)
    norms = torch.where(norms > 0., norms, 1.)
    return vecs / norms[...,None]
