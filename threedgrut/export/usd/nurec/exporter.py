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

"""
NuRec Exporter for Omniverse-compatible USDZ files.

This exporter produces USDZ files in the NuRec format for rendering in
Omniverse Kit and Isaac Sim. For standard OpenUSD export, use USDExporter.
"""

import gzip
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import msgpack
import numpy as np
import torch
from pxr import Sdf, Usd

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.transforms import estimate_normalizing_transform
from threedgrut.export.usd.camera_copy import (
    merge_source_prim_at_same_path,
    merge_source_prims_and_collect_sidecars,
    stage_has_ppisp_post_processing_effects,
)
from threedgrut.export.usd.exporter import (
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    _build_camera_frame_mapping_from_grouping,
    _build_camera_time_mapping_from_grouping,
    _build_frame_time_codes_from_grouping,
    _extract_camera_grouping,
    _extract_camera_params_from_dataset,
    _extract_camera_resolutions,
    _is_ppisp_post_processing,
    _normalize_positive_finite_float,
    _normalize_ppisp_responsivity,
    _resolve_enable_ppisp_controller_export,
    _suffix_camera_names,
    normalize_ppisp_integration_mode,
)
from threedgrut.export.usd.nurec.serializer import (
    serialize_nurec_usd,
    serialize_usd_default_layer,
    write_to_usdz,
)
from threedgrut.export.usd.nurec.templates import NamedSerialized, fill_3dgut_template
from threedgrut.export.usd.ppisp_spg import (
    ppisp_has_controller,
    resolve_ppisp_controller_export_enabled,
    select_spg_files_for_export,
)
from threedgrut.export.usd.writers.camera import export_cameras_to_usd
from threedgrut.export.usd.writers.render_product import create_render_products
from threedgrut.utils.logger import logger

_DEFAULT_RENDER_PRODUCT_VAR = "LdrColor"
_PPISP_INPUT_RENDER_PRODUCT_VAR = "HdrColor"
_NUREC_REFERENCED_WORLD_PATH = "/World/gauss"


def _remap_render_product_camera_targets(
    stage: Usd.Stage,
    *,
    source_prefix: str = "/World/Cameras",
    target_prefix: str = f"{_NUREC_REFERENCED_WORLD_PATH}/Cameras",
) -> None:
    render_scope = stage.GetPrimAtPath("/Render")
    if not render_scope.IsValid():
        return

    for prim in Usd.PrimRange(render_scope):
        if prim.GetTypeName() != "RenderProduct":
            continue
        camera_rel = prim.GetRelationship("camera")
        if not camera_rel:
            continue
        remapped_targets = []
        changed = False
        for target in camera_rel.GetTargets():
            target_text = str(target)
            if target_text == source_prefix or target_text.startswith(f"{source_prefix}/"):
                suffix = target_text[len(source_prefix) :]
                remapped_targets.append(Sdf.Path(f"{target_prefix}{suffix}"))
                changed = True
            else:
                remapped_targets.append(target)
        if changed:
            camera_rel.SetTargets(remapped_targets)


def _get_default_nurec_conf() -> SimpleNamespace:
    """Minimal config for NuRec export when no training config is available (e.g. transcode from PLY)."""
    conf = SimpleNamespace()
    conf.model = SimpleNamespace(density_activation="sigmoid", scale_activation="exp")
    conf.render = SimpleNamespace(
        method="3dgut",
        particle_kernel_degree=2,
        particle_kernel_density_clamping=True,
        particle_kernel_min_response=0.0113,
        particle_radiance_sph_degree=3,
        min_transmittance=0.0001,
        splat=SimpleNamespace(
            global_z_order=True,
            n_rolling_shutter_iterations=5,
            ut_alpha=1.0,
            ut_beta=2.0,
            ut_kappa=0.0,
            ut_require_all_sigma_points_valid=False,
            ut_in_image_margin_factor=0.1,
            rect_bounding=True,
            tight_opacity_bounding=True,
            tile_based_culling=True,
            k_buffer_size=0,
        ),
    )
    conf.export_usd = SimpleNamespace(apply_normalizing_transform=True)
    return conf


class NuRecExporter(ModelExporter):
    """Exporter for NuRec/Omniverse USDZ format files.

    Implements export functionality for Gaussian models in the NuRec USDZ format,
    which allows for rendering in Omniverse Kit and Isaac Sim.

    This is the legacy format. For standard OpenUSD export with ParticleField3DGaussianSplat
    schema, use USDExporter instead.
    """

    def __init__(
        self,
        *,
        export_cameras: bool = True,
        export_post_processing: bool = True,
        ppisp_integration_mode: str | None = None,
        ppisp_reference_camera_id: int | None = None,
        ppisp_reference_frame_id: int | None = None,
        ppisp_responsivity: float = 1.0,
        enable_ppisp_controller_export: bool | None = None,
        sh_optimization_num_iterations: int | None = None,
        scene_radiance_scale: float | None = None,
    ) -> None:
        self.export_cameras = export_cameras
        self.export_post_processing = export_post_processing
        self.ppisp_integration_mode = normalize_ppisp_integration_mode(ppisp_integration_mode)
        self.ppisp_reference_camera_id = None if ppisp_reference_camera_id is None else int(ppisp_reference_camera_id)
        self.ppisp_reference_frame_id = None if ppisp_reference_frame_id is None else int(ppisp_reference_frame_id)
        self.ppisp_responsivity = _normalize_ppisp_responsivity(ppisp_responsivity)
        self.enable_ppisp_controller_export = _resolve_enable_ppisp_controller_export(enable_ppisp_controller_export)
        if (
            self.enable_ppisp_controller_export is True
            and self.ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED
        ):
            raise ValueError(
                "enable_ppisp_controller_export is incompatible with ppisp_integration_mode='sh-optimized'"
            )
        self.sh_optimization_num_iterations = 3000
        if sh_optimization_num_iterations is not None:
            if isinstance(sh_optimization_num_iterations, bool) or not isinstance(sh_optimization_num_iterations, int):
                raise TypeError(
                    "sh_optimization_num_iterations must be int, "
                    f"got {type(sh_optimization_num_iterations).__name__}"
                )
            if sh_optimization_num_iterations < 1:
                raise ValueError(f"sh_optimization_num_iterations must be >= 1, got {sh_optimization_num_iterations}.")
            self.sh_optimization_num_iterations = int(sh_optimization_num_iterations)
        self.scene_radiance_scale = _normalize_positive_finite_float(
            "scene_radiance_scale",
            scene_radiance_scale,
            default=1.0,
        )

    @torch.no_grad()
    def export(
        self, model: ExportableModel, output_path: Path, dataset=None, conf: Dict[str, Any] = None, **kwargs
    ) -> None:
        """Export the model to a NuRec USDZ file.

        Args:
            model: The model to export (must implement ExportableModel)
            output_path: Path where the USDZ file will be saved
            dataset: Optional dataset to get camera poses for upright transform
            conf: Configuration parameters for the renderer
            **kwargs: Additional parameters for export
        """
        logger.info(f"exporting nurec usdz file to {output_path}...")

        if conf is None:
            conf = _get_default_nurec_conf()
        if conf.render.method not in ["3dgut", "3dgrt"]:
            raise ValueError(
                f"NuRec export requires render.method to be '3dgut' or '3dgrt', got '{conf.render.method}'"
            )

        post_processing = kwargs.get("post_processing")
        validation_dataset = kwargs.get("validation_dataset")
        has_ppisp_module = _is_ppisp_post_processing(post_processing)
        uses_baked_post_processing_export = (
            post_processing is not None
            and self.export_post_processing
            and self.ppisp_integration_mode == PPISP_INTEGRATION_MODE_SH_OPTIMIZED
        )
        uses_omni_native_post_processing_export = (
            post_processing is not None
            and self.export_post_processing
            and self.ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
        )
        enable_ppisp_controller_export = False
        if uses_omni_native_post_processing_export:
            if self.enable_ppisp_controller_export is True and not ppisp_has_controller(post_processing):
                raise ValueError("enable_ppisp_controller_export=True requires a PPISP module with trained controllers")
            enable_ppisp_controller_export = resolve_ppisp_controller_export_enabled(
                requested=self.enable_ppisp_controller_export,
                ppisp_module=post_processing,
                ppisp_integration_mode=self.ppisp_integration_mode,
            )
            if enable_ppisp_controller_export and (
                self.ppisp_reference_camera_id is not None or self.ppisp_reference_frame_id is not None
            ):
                logger.warning("PPISP controller export is enabled; ppisp_reference_camera_id/frame_id are ignored.")
        if self.export_cameras and dataset is None:
            raise ValueError(
                "export_cameras=True requires a dataset so camera poses, intrinsics, "
                "and RenderProducts can be authored. Pass a dataset or set export_cameras=False."
            )

        if uses_baked_post_processing_export:
            from threedgrut.export.usd.post_processing_sh_bake import (
                MODE_PPISP_BAKE_VIGNETTING_NONE,
                PPISPPostProcessingBakeAdapter,
                bake_post_processing_into_sh,
            )

            if not has_ppisp_module:
                raise ValueError("sh-optimized post-processing export currently supports PPISP post-processing only.")
            bake_camera_id = 0 if self.ppisp_reference_camera_id is None else self.ppisp_reference_camera_id
            bake_frame_id = 0 if self.ppisp_reference_frame_id is None else self.ppisp_reference_frame_id
            adapter = PPISPPostProcessingBakeAdapter(
                camera_id=bake_camera_id,
                frame_id=bake_frame_id,
                vignetting_mode=MODE_PPISP_BAKE_VIGNETTING_NONE,
            )
            logger.info(
                "Baking post-processing into NuRec Gaussian SH coefficients before export "
                f"(camera={bake_camera_id}, frame={bake_frame_id})"
            )
            model = bake_post_processing_into_sh(
                model=model,
                post_processing=post_processing,
                train_dataset=dataset,
                conf=conf,
                adapter=adapter,
                num_iterations=self.sh_optimization_num_iterations,
            )
        if uses_omni_native_post_processing_export and not has_ppisp_module:
            raise ValueError("spg-runtime post-processing export currently supports PPISP post-processing only.")

        if self.scene_radiance_scale != 1.0:
            from threedgrut.export.usd.post_processing_sh_bake import scale_sh_output

            scale_sh_output(model, self.scene_radiance_scale)

        copy_source_usd = kwargs.get("copy_source_usd")
        if copy_source_usd is None:
            copy_source_usd = kwargs.get("copy_cameras_source")
        copied_source_stage = None
        copied_source_has_post_processing = False
        if copy_source_usd is not None:
            stage_path, _ = copy_source_usd
            try:
                copied_source_stage = Usd.Stage.Open(str(stage_path))
                if copied_source_stage:
                    copied_source_has_post_processing = stage_has_ppisp_post_processing_effects(copied_source_stage)
            except Exception as exc:
                logger.warning("Could not inspect source USD for NuRec post-processing effects: %s", exc)

        # Use accessor to get model data
        accessor = GaussianExportAccessor(model, conf)
        attrs = accessor.get_attributes(preactivation=True)
        caps = accessor.get_capabilities()

        # Apply normalizing transform if enabled and dataset is provided
        normalizing_transform = np.eye(4)
        export_usd = getattr(conf, "export_usd", None)
        apply_transform = getattr(export_usd if export_usd is not None else conf, "apply_normalizing_transform", True)
        if apply_transform and dataset is not None:
            try:
                poses = dataset.get_poses()
                normalizing_transform = estimate_normalizing_transform(poses)
                logger.info("Applying normalizing transform to NuRec export")
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to apply normalizing transform: {e}")
                normalizing_transform = np.eye(4)

        # Set up common parameters
        template_params = {
            "positions": attrs.positions,
            "rotations": attrs.rotations,
            "scales": attrs.scales,
            "densities": attrs.densities,
            "features_albedo": attrs.albedo,
            "features_specular": attrs.specular,
            "n_active_features": caps.sh_degree,
            "density_kernel_degree": conf.render.particle_kernel_degree,
            # Common renderer configuration parameters
            "density_activation": conf.model.density_activation,
            "scale_activation": conf.model.scale_activation,
            "rotation_activation": "normalize",  # Always normalize for rotations
            "density_kernel_density_clamping": conf.render.particle_kernel_density_clamping,
            "density_kernel_min_response": conf.render.particle_kernel_min_response,
            "radiance_sph_degree": conf.render.particle_radiance_sph_degree,
            "transmittance_threshold": conf.render.min_transmittance,
        }

        if conf.render.method == "3dgut":
            # 3DGUT-specific splatting parameters
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
            # For 3DGRT renderer, fall back to default splatting parameters
            logger.warning("Using 3DGUT NuRec template for 3DGRT data, may see slight loss of quality.")

        template = fill_3dgut_template(**template_params)

        # Compress the data
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=0) as f:
            packed = msgpack.packb(template)
            f.write(packed)

        model_file = NamedSerialized(filename=output_path.stem + ".nurec", serialized=buffer.getvalue())

        apply_coordinate_transform = kwargs.get("apply_coordinate_transform", False)

        # Create USD representations
        gauss_usd = serialize_nurec_usd(
            model_file,
            attrs.positions,
            normalizing_transform,
            apply_coordinate_transform=apply_coordinate_transform,
            source_gaussian_transform=kwargs.get("source_gaussian_transform"),
            author_render_settings=True,
            invert_registered_compositing=not (
                uses_omni_native_post_processing_export or copied_source_has_post_processing
            ),
            skip_gaussian_tonemapping=uses_omni_native_post_processing_export or copied_source_has_post_processing,
        )
        train_cameras = None
        validation_cameras = None
        if self.export_cameras:
            train_cameras = self._export_dataset_cameras(
                gauss_usd.stage,
                dataset=dataset,
                normalizing_transform=normalizing_transform,
                apply_transform=apply_transform,
            )
            if train_cameras is None:
                raise ValueError(
                    "export_cameras=True could not export NuRec cameras. "
                    "Check that dataset poses and intrinsics are available."
                )
            if validation_dataset is not None:
                validation_cameras = self._export_dataset_cameras(
                    gauss_usd.stage,
                    dataset=validation_dataset,
                    normalizing_transform=normalizing_transform,
                    apply_transform=apply_transform,
                    validation=True,
                )
                if validation_cameras is None:
                    raise ValueError(
                        "export_cameras=True could not export NuRec validation cameras. "
                        "Check that validation dataset poses and intrinsics are available."
                    )
        extra_files: list[NamedSerialized] = []
        if self.export_cameras and not uses_omni_native_post_processing_export:
            render_products = self._create_camera_render_products(
                gauss_usd.stage, train_cameras, (_DEFAULT_RENDER_PRODUCT_VAR,)
            )
            if not render_products:
                raise ValueError(
                    "export_cameras=True created NuRec cameras but no RenderProducts. "
                    "Check that dataset intrinsics include native image resolution."
                )
            if validation_cameras is not None:
                validation_render_products = self._create_camera_render_products(
                    gauss_usd.stage, validation_cameras, (_DEFAULT_RENDER_PRODUCT_VAR,)
                )
                if not validation_render_products:
                    raise ValueError(
                        "export_cameras=True created NuRec validation cameras but no validation RenderProducts. "
                        "Check that validation dataset intrinsics include native image resolution."
                    )
        elif self.export_cameras:
            render_products = self._create_camera_render_products(
                gauss_usd.stage, train_cameras, (_PPISP_INPUT_RENDER_PRODUCT_VAR,)
            )
            if not render_products:
                raise ValueError(
                    "export_cameras=True could not create NuRec PPISP RenderProducts. "
                    "Check that dataset cameras and intrinsics are available."
                )
            if render_products is not None:
                if validation_cameras is not None:
                    validation_render_products = self._create_camera_render_products(
                        gauss_usd.stage, validation_cameras, (_PPISP_INPUT_RENDER_PRODUCT_VAR,)
                    )
                    if not validation_render_products:
                        raise ValueError(
                            "export_cameras=True could not create NuRec validation PPISP RenderProducts. "
                            "Check that validation dataset cameras and intrinsics are available."
                        )
                self._export_ppisp(
                    stage=gauss_usd.stage,
                    dataset=dataset,
                    camera_info=train_cameras,
                    post_processing=post_processing,
                    files=extra_files,
                    enable_ppisp_controller_export=enable_ppisp_controller_export,
                )
                if validation_cameras is not None:
                    validation_mapping = _build_camera_frame_mapping_from_grouping(
                        validation_cameras["camera_names"],
                        validation_cameras["frame_to_camera"],
                    )
                    validation_time_mapping = _build_camera_time_mapping_from_grouping(
                        validation_cameras["camera_names"],
                        validation_cameras["frame_to_camera"],
                    )
                    self._export_ppisp(
                        stage=gauss_usd.stage,
                        dataset=validation_dataset,
                        camera_info=validation_cameras,
                        post_processing=post_processing,
                        files=extra_files,
                        camera_frame_mapping=validation_mapping,
                        camera_time_mapping=validation_time_mapping,
                        neutral_frame_params=(
                            self.ppisp_reference_camera_id is None and self.ppisp_reference_frame_id is None
                        ),
                        enable_ppisp_controller_export=enable_ppisp_controller_export,
                    )
        default_usd = serialize_usd_default_layer(gauss_usd)
        if uses_omni_native_post_processing_export:
            merge_source_prim_at_same_path(default_usd.stage, gauss_usd.stage, "/Render")
            _remap_render_product_camera_targets(default_usd.stage)
        if copy_source_usd is not None:
            stage_path, res_root = copy_source_usd
            try:
                src_stage = copied_source_stage or Usd.Stage.Open(str(stage_path))
                if not src_stage:
                    logger.warning("Could not open source USD for NuRec prim merge: %s", stage_path)
                else:
                    merge_source_prims_and_collect_sidecars(
                        dest_stage=default_usd.stage,
                        source_stage=src_stage,
                        res_root=res_root,
                        source_stage_path=Path(stage_path),
                        files=extra_files,
                        skip_source_subtrees=kwargs.get("copy_source_skip_subtrees"),
                    )
            except Exception as exc:
                logger.warning("Failed to merge source USD prims into NuRec output: %s", exc)

        # Write the final USDZ file
        write_to_usdz(output_path, model_file, gauss_usd, default_usd, extra_files if extra_files else None)

    def _export_dataset_cameras(
        self,
        stage,
        *,
        dataset,
        normalizing_transform: np.ndarray,
        apply_transform: bool,
        validation: bool = False,
    ) -> dict | None:
        if dataset is None:
            return None

        try:
            poses = dataset.get_poses()
            if apply_transform:
                poses = np.einsum("ij,njk->nik", normalizing_transform, poses)

            camera_names, frame_to_camera = _extract_camera_grouping(dataset)
            if validation:
                camera_names = _suffix_camera_names(camera_names)

            camera_params = _extract_camera_params_from_dataset(dataset)
            previous_start = stage.GetStartTimeCode()
            previous_end = stage.GetEndTimeCode()
            camera_paths = export_cameras_to_usd(
                stage=stage,
                poses=poses,
                camera_names=camera_names,
                frame_to_camera=frame_to_camera,
                camera_params=camera_params,
                frame_time_codes=_build_frame_time_codes_from_grouping(camera_names, frame_to_camera),
                root_path="/World/Cameras",
                visible=False,
            )
            stage.SetStartTimeCode(min(previous_start, stage.GetStartTimeCode()))
            stage.SetEndTimeCode(max(previous_end, stage.GetEndTimeCode()))
            camera_kind = "validation " if validation else ""
            logger.info(f"Exported {len(camera_paths)} {camera_kind}camera(s) to NuRec USD from {len(poses)} frames")
            return {
                "camera_names": camera_names,
                "frame_to_camera": frame_to_camera,
                "camera_params": camera_params,
                "camera_prim_paths": camera_paths,
            }
        except (AttributeError, KeyError, ValueError) as exc:
            camera_kind = "validation " if validation else ""
            logger.warning(f"Failed to export {camera_kind}cameras to NuRec USD: {exc}")
            return None

    def _create_camera_render_products(
        self, stage, camera_info: dict | None, render_vars: tuple[str, ...]
    ) -> dict | None:
        if camera_info is None:
            return None
        camera_prim_paths = camera_info["camera_prim_paths"]
        if not camera_prim_paths:
            logger.warning("No NuRec camera prims available for RenderProduct wiring, skipping")
            return None

        resolutions = _extract_camera_resolutions(
            camera_info["camera_params"],
            camera_info["camera_names"],
            camera_info["frame_to_camera"],
        )
        camera_entries = {}
        for camera_name, camera_path in camera_prim_paths.items():
            if camera_name not in resolutions:
                logger.warning(f"No native resolution found for NuRec camera {camera_name}; skipping RenderProduct")
                continue
            width, height = resolutions[camera_name]
            camera_entries[camera_name] = (camera_path, width, height)
        if not camera_entries:
            logger.warning("No NuRec RenderProducts created because no native camera resolutions were found")
            return None

        create_render_products(stage=stage, camera_entries=camera_entries, render_vars=render_vars)
        return camera_entries

    def _export_ppisp(
        self,
        *,
        stage,
        dataset,
        camera_info: dict,
        post_processing,
        files: list[NamedSerialized],
        camera_frame_mapping: dict | None = None,
        camera_time_mapping: dict | None = None,
        neutral_frame_params: bool = False,
        enable_ppisp_controller_export: bool = False,
    ) -> None:
        try:
            from ppisp import PPISP  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("ppisp package not available, skipping NuRec PPISP export")
            return

        if not isinstance(post_processing, PPISP):
            logger.warning(
                f"NuRec PPISP export requested but post_processing is {type(post_processing).__name__}; skipping"
            )
            return

        from threedgrut.export.usd.writers.ppisp_writer import (
            add_ppisp_to_all_render_products,
            build_camera_frame_mapping,
            build_camera_time_mapping,
        )

        if camera_frame_mapping is None:
            _, camera_frame_mapping = build_camera_frame_mapping(dataset)
        if camera_time_mapping is None:
            _, camera_time_mapping = build_camera_time_mapping(dataset)

        fixed_camera_id = None if enable_ppisp_controller_export else self.ppisp_reference_camera_id
        fixed_frame_id = None if enable_ppisp_controller_export else self.ppisp_reference_frame_id
        if not enable_ppisp_controller_export and fixed_frame_id is not None and fixed_camera_id is None:
            raise ValueError(
                "ppisp_reference_frame_id was set without ppisp_reference_camera_id "
                "in spg-runtime export mode. Set ppisp_reference_camera_id as well, "
                "or leave both unset for time-sampled SPG authoring."
            )

        add_ppisp_to_all_render_products(
            stage=stage,
            ppisp=post_processing,
            camera_names=camera_info["camera_names"],
            camera_frame_mapping=camera_frame_mapping,
            camera_time_mapping=camera_time_mapping,
            fixed_camera_index=fixed_camera_id,
            fixed_frame_index=fixed_frame_id,
            use_controller=enable_ppisp_controller_export,
            responsivity=self.ppisp_responsivity,
            neutral_frame_params=neutral_frame_params and not enable_ppisp_controller_export,
        )

        spg_files = select_spg_files_for_export(
            enable_ppisp_controller_export=enable_ppisp_controller_export,
            ppisp_module=post_processing if enable_ppisp_controller_export else None,
            camera_indices=(
                range(len(getattr(post_processing, "controllers", []))) if enable_ppisp_controller_export else None
            ),
        )

        for spg_file in spg_files:
            if not any(file.filename == spg_file.filename for file in files):
                files.append(spg_file)

        logger.info(
            f"NuRec PPISP export complete: {len(files)} sidecar(s) added "
            f"(controller={enable_ppisp_controller_export})"
        )
