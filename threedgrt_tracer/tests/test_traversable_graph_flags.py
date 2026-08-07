# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pipeline traversal-declaration <-> traced-graph consistency tests.

Each case runs in its own subprocess: the validation-mode env var is read
once at OptiX context creation, and an aborted launch leaves the process
CUDA context unusable. Validation failures surface only through the OptiX
log callback on stderr (OPTIX_CHECK builds but never emits the message).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("omegaconf")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU (OptiX tracer)"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_conf(primitive_type):
    from omegaconf import OmegaConf

    import threedgrut.utils.misc  # noqa: F401  (registers the div/eq resolvers)

    conf = OmegaConf.load(_REPO_ROOT / "configs" / "base_gs.yaml")
    conf.render = OmegaConf.load(_REPO_ROOT / "configs" / "render" / "3dgrt.yaml")
    conf.render.primitive_type = primitive_type
    conf.render.enable_kernel_timings = False
    return conf


class _TinyGaussians:
    """Minimal duck-typed stand-in for MixtureOfGaussians (values pre-activated)."""

    def __init__(self, n=64, feature_dim=48, device="cuda"):
        gen = torch.Generator().manual_seed(0)
        self.positions = (torch.rand((n, 3), generator=gen) * 0.4 - 0.2).to(device)
        rotation = torch.zeros((n, 4))
        rotation[:, 0] = 1.0  # identity quaternion
        self.rotation = rotation.to(device)
        self.scale = torch.full((n, 3), 0.15, device=device)
        self.density = torch.full((n, 1), 0.9, device=device)
        features = torch.zeros((n, feature_dim))
        features[:, :3] = 1.0  # bright DC term
        self._features = features.to(device)
        self.num_gaussians = n
        self.n_active_features = 3

    @staticmethod
    def rotation_activation(q):
        return q

    @staticmethod
    def scale_activation(s):
        return s

    @staticmethod
    def density_activation(d):
        return d

    def get_rotation(self):
        return self.rotation

    def get_scale(self):
        return self.scale

    def get_density(self):
        return self.density

    def get_features(self):
        return self._features


def _make_batch(h=48, w=48, device="cuda"):
    from threedgrut.datasets.protocols import Batch

    ys, xs = torch.meshgrid(
        torch.linspace(-0.35, 0.35, h),
        torch.linspace(-0.35, 0.35, w),
        indexing="ij",
    )
    rays_dir = torch.nn.functional.normalize(
        torch.stack([xs, ys, torch.ones_like(xs)], dim=-1), dim=-1
    )[None].to(device)
    rays_ori = torch.zeros_like(rays_dir)
    rays_ori[..., 2] = -2.5
    return Batch(
        rays_ori=rays_ori,
        rays_dir=rays_dir,
        T_to_world=torch.eye(4, device=device)[None],
    )


def _probe(primitive_type):
    """Body of one test case; runs in a fresh interpreter (see __main__ hook)."""
    if not torch.cuda.is_available():
        print("PROBE_SKIP: no CUDA device in subprocess", file=sys.stderr)
        return 3
    try:
        from threedgrt_tracer.tracer import Tracer
    except ImportError as exc:
        print(f"PROBE_SKIP: tracer import failed: {exc}", file=sys.stderr)
        return 3

    tracer = Tracer(_make_conf(primitive_type))
    gaussians = _TinyGaussians()
    tracer.build_acc(gaussians, rebuild=True)
    out = tracer.render(gaussians, _make_batch())
    torch.cuda.synchronize()
    print(f"PROBE_OK max_opacity={float(out['pred_opacity'].max()):.6f}")
    return 0


@pytest.mark.parametrize("primitive_type", ["icosahedron", "instances"])
def test_flags_match_traced_graph(primitive_type):
    env = dict(os.environ)
    # Validation mode turns the unspecified outcomes of a mismatched traversal
    # declaration into a deterministic, architecture-independent failure. Set
    # before the interpreter starts: it is read at OptiX context creation.
    env["THREEDGRUT_OPTIX_VALIDATION"] = "1"
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe", primitive_type],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,  # first run JIT-compiles the extension
    )

    if proc.returncode == 3:
        pytest.skip(f"probe environment unavailable:\n{proc.stderr.strip()}")

    # Check the OptiX log first so a validation abort attributes precisely
    # (an aborted launch also makes the returncode nonzero).
    assert "VALIDATION_ERROR" not in proc.stderr, f"OptiX validation error:\n{proc.stderr}"
    assert "Invalid traversable type" not in proc.stderr, f"OptiX validation error:\n{proc.stderr}"
    assert proc.returncode == 0, (
        f"probe exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "PROBE_OK" in proc.stdout, f"probe produced no result:\n{proc.stdout}"
    max_opacity = float(proc.stdout.split("max_opacity=")[1].split()[0])
    assert max_opacity > 0.0, "tracer returned zero opacity everywhere (no hits)"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        sys.exit(_probe(sys.argv[2]))
    print(f"usage: {sys.argv[0]} --probe <primitive_type>", file=sys.stderr)
    sys.exit(2)
