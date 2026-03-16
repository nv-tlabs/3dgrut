import gzip
from pathlib import Path
from typing import Any, Dict, Tuple

import msgpack
import numpy as np
import torch

from threedgrut.export.base import ExportableModel
from threedgrut.export.nurec_templates import fill_3dgut_template
from threedgrut.utils.logger import logger


def _build_template_params(conf: Dict[str, Any], method: str) -> Dict[str, Any]:
    template_params = {
        "density_kernel_degree": conf.render.particle_kernel_degree,
        "density_activation": conf.model.density_activation,
        "scale_activation": conf.model.scale_activation,
        "rotation_activation": "normalize",
        "density_kernel_density_clamping": conf.render.particle_kernel_density_clamping,
        "density_kernel_min_response": conf.render.particle_kernel_min_response,
        "radiance_sph_degree": conf.render.particle_radiance_sph_degree,
        "transmittance_threshold": conf.render.min_transmittance,
    }

    if method == "3dgut":
        template_params.update(
            {
                "global_z_order": conf.render.splat.global_z_order,
                "n_rolling_shutter_iterations": conf.render.splat.n_rolling_shutter_iterations,
                "ut_alpha": conf.render.splat.ut_alpha,
                "ut_beta": conf.render.splat.ut_beta,
                "ut_kappa": conf.render.splat.ut_kappa,
                "ut_require_all_sigma_points": conf.render.splat.ut_require_all_sigma_points_valid,
                "image_margin_factor": conf.render.splat.ut_in_image_margin_factor,
                "rect_bounding": conf.render.splat.rect_bounding,
                "tight_opacity_bounding": conf.render.splat.tight_opacity_bounding,
                "tile_based_culling": conf.render.splat.tile_based_culling,
                "k_buffer_size": conf.render.splat.k_buffer_size,
            }
        )
    else:
        logger.warning("Using 3DGUT NuRec template for 3DGRT data, may see slight loss of quality.")

    return template_params


def crop_model_to_bbox(model: ExportableModel, min_bounds: np.ndarray, max_bounds: np.ndarray) -> Tuple[int, int]:
    min_b = torch.as_tensor(min_bounds, dtype=model.get_positions().dtype, device=model.get_positions().device)
    max_b = torch.as_tensor(max_bounds, dtype=model.get_positions().dtype, device=model.get_positions().device)

    positions = model.get_positions()
    mask = ((positions >= min_b) & (positions <= max_b)).all(dim=1)
    kept = int(mask.sum().item())
    total = int(mask.shape[0])

    if kept == 0:
        return kept, total

    model.positions = torch.nn.Parameter(model.positions[mask])
    model.rotation = torch.nn.Parameter(model.rotation[mask])
    model.scale = torch.nn.Parameter(model.scale[mask])
    model.density = torch.nn.Parameter(model.density[mask])
    model.features_albedo = torch.nn.Parameter(model.features_albedo[mask])
    model.features_specular = torch.nn.Parameter(model.features_specular[mask])

    return kept, total


def export_model_to_nurec(model: ExportableModel, output_path: Path, conf: Dict[str, Any], method: str) -> None:
    positions = model.get_positions().detach().cpu().numpy()
    rotations = model.get_rotation(preactivation=True).detach().cpu().numpy()
    scales = model.get_scale(preactivation=True).detach().cpu().numpy()
    densities = model.get_density(preactivation=True).detach().cpu().numpy()
    features_albedo = model.get_features_albedo().detach().cpu().numpy()
    features_specular = model.get_features_specular().detach().cpu().numpy()
    n_active_features = model.get_n_active_features()

    template_params = _build_template_params(conf, method)
    template = fill_3dgut_template(
        positions=positions,
        rotations=rotations,
        scales=scales,
        densities=densities,
        features_albedo=features_albedo,
        features_specular=features_specular,
        n_active_features=n_active_features,
        **template_params,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, mode="wb", compresslevel=0) as file:
        file.write(msgpack.packb(template))
