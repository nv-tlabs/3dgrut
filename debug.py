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

import subprocess
import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options
import torch
from pathlib import Path
import numpy as np
import json


OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "Unknown"


def pt_to_raw(file: str) -> None:
    data = torch.load(f"{file}.pt")
    # input parameters
    T_to_world = data["T_to_world"].cpu().numpy()
    rays_ori = data["rays_ori"].cpu().numpy()
    rays_dir = data["rays_dir"].cpu().numpy()
    pred_rgb = data["pred_rgb"].detach().cpu().numpy()
    pred_opacity = data["pred_opacity"].detach().cpu().numpy()
    pred_dist = data["pred_dist"].detach().cpu().numpy()
    pred_normals = data["pred_normals"].detach().cpu().numpy()
    hits_count = data["hits_count"].detach().cpu().numpy()
    mog_visibility = data["mog_visibility"].detach().cpu().numpy()
    # save metadata
    with open(f"{file}_metadata.json", "w") as f:
        json.dump(
            {
                "T_to_world": T_to_world.shape,
                "rays_ori": rays_ori.shape,
                "rays_dir": rays_dir.shape,
                "pred_rgb": pred_rgb.shape,
                "pred_opacity": pred_opacity.shape,
                "pred_dist": pred_dist.shape,
                "pred_normals": pred_normals.shape,
                "hits_count": hits_count.shape,
                "mog_visibility": mog_visibility.shape,
            },
            f,
            indent=4,
        )
    # convert to file
    T_to_world.tofile(f"{file}_T_to_world.raw")
    rays_ori.tofile(f"{file}_rays_ori.raw")
    rays_dir.tofile(f"{file}_rays_dir.raw")
    pred_rgb.tofile(f"{file}_pred_rgb.raw")
    pred_opacity.tofile(f"{file}_pred_opacity.raw")
    pred_dist.tofile(f"{file}_pred_dist.raw")
    pred_normals.tofile(f"{file}_pred_normals.raw")
    hits_count.tofile(f"{file}_hits_count.raw")
    mog_visibility.tofile(f"{file}_mog_visibility.raw")


def load_raw(file: str) -> None:
    with open(f"{file}_metadata.json", "r") as f:
        shapes = json.load(f)

    T_to_world = np.fromfile(f"{file}_T_to_world.raw", dtype=np.float32).reshape(
        *shapes["T_to_world"]
    )
    rays_ori = np.fromfile(f"{file}_rays_ori.raw", dtype=np.float32).reshape(
        *shapes["rays_ori"]
    )
    rays_dir = np.fromfile(f"{file}_rays_dir.raw", dtype=np.float32).reshape(
        *shapes["rays_dir"]
    )
    pred_rgb = np.fromfile(f"{file}_pred_rgb.raw", dtype=np.float32).reshape(
        *shapes["pred_rgb"]
    )
    pred_opacity = np.fromfile(f"{file}_pred_opacity.raw", dtype=np.float32).reshape(
        *shapes["pred_opacity"]
    )
    pred_dist = np.fromfile(f"{file}_pred_dist.raw", dtype=np.float32).reshape(
        *shapes["pred_dist"]
    )
    pred_normals = np.fromfile(f"{file}_pred_normals.raw", dtype=np.float32).reshape(
        *shapes["pred_normals"]
    )
    hits_count = np.fromfile(f"{file}_hits_count.raw", dtype=np.float32).reshape(
        *shapes["hits_count"]
    )
    mog_visibility = np.fromfile(
        f"{file}_mog_visibility.raw", dtype=np.float32
    ).reshape(*shapes["mog_visibility"])
    return {
        "T_to_world": torch.from_numpy(T_to_world).to(device="cuda"),
        "rays_ori": torch.from_numpy(rays_ori).to(device="cuda"),
        "rays_dir": torch.from_numpy(rays_dir).to(device="cuda"),
        "pred_rgb": torch.from_numpy(pred_rgb).to(device="cuda"),
        "pred_opacity": torch.from_numpy(pred_opacity).to(device="cuda"),
        "pred_dist": torch.from_numpy(pred_dist).to(device="cuda"),
        "pred_normals": torch.from_numpy(pred_normals).to(device="cuda"),
        "hits_count": torch.from_numpy(hits_count).to(device="cuda"),
        "mog_visibility": torch.from_numpy(mog_visibility).to(device="cuda"),
    }


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    logger.info(f"Git hash: {get_git_revision_hash()}")
    logger.info(f"Compiling native code..")

    from threedgrut.trainer import Trainer3DGRUT
    from threedgrut.export.ply_exporter import PLYExporter
    from threedgrt_tracer.tracer import Tracer

    trainer = Trainer3DGRUT(conf)

    # exporter = PLYExporter()
    # exporter.export(
    #     trainer.model, Path("debug.ply"), dataset=trainer.train_dataset, conf=conf
    # )
    # trainer.run_train_pass(conf)

    gaussians = trainer.model
    gaussians.build_acc(rebuild=True)
    gaussians.train()
    renderer = trainer.model.renderer

    # pt_to_raw("debug")
    debug_data = load_raw("debug/debug")

    T_to_world = debug_data["T_to_world"]
    rays_ori = debug_data["rays_ori"]
    rays_dir = debug_data["rays_dir"]

    (
        pred_rgb_generated,
        pred_opacity_generated,
        pred_dist_generated,
        pred_normals_generated,
        hits_count_generated,
        mog_visibility_generated,
    ) = Tracer._Autograd.apply(
        renderer.tracer_wrapper,
        0,
        # input parameters
        T_to_world.contiguous(),
        rays_ori.contiguous(),
        rays_dir.contiguous(),
        # model parameters
        gaussians.positions.contiguous(),
        gaussians.get_rotation().contiguous(),
        gaussians.get_scale().contiguous(),
        gaussians.get_density().contiguous(),
        gaussians.get_features().contiguous(),
        # render options
        Tracer.RenderOpts.DEFAULT,
        gaussians.n_active_features,
        conf.render.min_transmittance,
    )
    # import pdb; pdb.set_trace()

    pred_rgb_reference = debug_data["pred_rgb"]
    pred_opacity_reference = debug_data["pred_opacity"]
    pred_dist_reference = debug_data["pred_dist"]
    pred_normals_reference = debug_data["pred_normals"]
    hits_count_reference = debug_data["hits_count"]
    mog_visibility_reference = debug_data["mog_visibility"]

    assert torch.allclose(pred_rgb_generated, pred_rgb_reference)
    assert torch.allclose(pred_opacity_generated, pred_opacity_reference)
    assert torch.allclose(pred_dist_generated, pred_dist_reference)
    assert torch.allclose(pred_normals_generated, pred_normals_reference)
    assert torch.allclose(hits_count_generated, hits_count_reference)
    assert torch.allclose(mog_visibility_generated, mog_visibility_reference)

    criterion = torch.nn.functional.mse_loss

    loss = criterion(pred_rgb_generated, 1e3 * torch.ones_like(pred_rgb_generated))
    loss.backward()

    print("Positions grad:")
    print(gaussians.positions.grad.mean())
    print("Rotation grad:")
    print(gaussians.rotation.grad.mean())
    print("Scale grad:")
    print(gaussians.scale.grad.mean())
    print("Density grad:")
    print(gaussians.density.grad.mean())
    print("Features albedo grad:")
    print(gaussians.features_albedo.grad.mean())
    print("Features specular grad:")
    print(gaussians.features_specular.grad.mean())


if __name__ == "__main__":
    main()


# Run the program with
# ```bash
# python debug.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgrt \
#   import_ply.enabled=true import_ply.path=model_0.ply
# ```
