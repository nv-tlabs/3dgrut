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
USD Exporter for 3DGRUT Gaussian Splatting models.

Main exporter using the ParticleField3DGaussianSplat schema (OpenUSD standard).
Produces USDZ files by default for maximum compatibility.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)
from pxr import Usd

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.transforms import (
    estimate_normalizing_transform,
    get_3dgrut_to_usdz_coordinate_transform,
)
from threedgrut.export.usd.camera_copy import (
    merge_source_prims_and_collect_sidecars,
    save_serialized_files,
)
from threedgrut.export.usd.particle_field_hints import (
    DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
    normalize_particle_field_sorting_mode_hint,
)
from threedgrut.export.usd.post_processing_sh_bake import (
    MODE_PPISP_BAKE_VIGNETTING_NONE,
    scale_sh_output,
)
from threedgrut.export.usd.ppisp_spg import (
    ppisp_has_controller,
    resolve_ppisp_controller_export_enabled,
    select_spg_files_for_export,
)
from threedgrut.export.usd.stage_utils import (
    NamedSerialized,
    NamedUSDStage,
    create_gaussian_model_root,
    initialize_usd_stage,
    write_to_usdz,
)
from threedgrut.export.usd.writers.background import export_background_to_usd
from threedgrut.export.usd.writers.base import create_gaussian_writer
from threedgrut.export.usd.writers.camera import export_cameras_to_usd
from threedgrut.export.usd.writers.render_product import create_render_products

logger = logging.getLogger(__name__)


_DEFAULT_RENDER_SCOPE_PATH = "/Render"
_DEFAULT_RENDER_PRODUCT_VAR = "LdrColor"
_PPISP_INPUT_RENDER_PRODUCT_VAR = "HdrColor"
_VALIDATION_CAMERA_SUFFIX = "_val"
_RENDER_SETTING_TONEMAP_OP = "rtx:post:tonemap:op"
_RENDER_SETTING_SKIP_GAUSSIAN_TONEMAPPING = "rtx:rtpt:gaussian:skipTonemapping:enabled"
PPISP_INTEGRATION_MODE_SPG_RUNTIME = "spg-runtime"
PPISP_INTEGRATION_MODE_SH_OPTIMIZED = "sh-optimized"
VALID_PPISP_INTEGRATION_MODES = (
    PPISP_INTEGRATION_MODE_SPG_RUNTIME,
    PPISP_INTEGRATION_MODE_SH_OPTIMIZED,
)


def _is_ppisp_post_processing(post_processing: Any) -> bool:
    post_processing_type = type(post_processing)
    return (
        post_processing_type.__name__ == "PPISP"
        and post_processing_type.__module__.split(".", maxsplit=1)[0] == "ppisp"
    )


def normalize_ppisp_integration_mode(mode: str | None) -> str:
    normalized = PPISP_INTEGRATION_MODE_SPG_RUNTIME if mode is None else str(mode).strip().lower()
    if normalized not in VALID_PPISP_INTEGRATION_MODES:
        raise ValueError(
            f"Unsupported PPISP integration mode '{mode}'. " f"Expected one of: {list(VALID_PPISP_INTEGRATION_MODES)}"
        )
    return normalized


def compute_runtime_post_processing(
    model_has_postprocessing: bool,
    ppisp_module: Any | None,
    ppisp_integration_mode: str,
) -> bool:
    """Return True iff a runtime PPISP SPG stage will be authored."""
    return (
        model_has_postprocessing
        and ppisp_module is not None
        and ppisp_integration_mode == PPISP_INTEGRATION_MODE_SPG_RUNTIME
    )


def _get_export_config_value(export_conf, hyphen_name: str, attr_name: str, default: Any) -> Any:
    if hasattr(export_conf, "get"):
        return export_conf.get(hyphen_name, getattr(export_conf, attr_name, default))
    return getattr(export_conf, attr_name, default)


def _normalize_positive_finite_float(name: str, value: Any, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise TypeError(f"{name} must be a real number, got None")
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    value = float(value)
    if not (value > 0.0):
        raise ValueError(f"{name} must be strictly positive, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _resolve_enable_ppisp_controller_export(enable_ppisp_controller_export: bool | None) -> bool | None:
    if enable_ppisp_controller_export is None:
        return None
    return bool(enable_ppisp_controller_export)


def _normalize_ppisp_responsivity(value: Any) -> float:
    if value is None:
        return 1.0
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"ppisp_responsivity must be a single achromatic value, got {value!r}")
        value = value[0]
    return _normalize_positive_finite_float("ppisp_responsivity", value)


def _extract_camera_params_from_dataset(dataset) -> Optional[List]:
    """
    Extract per-frame camera parameters from a dataset.

    Handles different dataset types:
    - ColmapDataset: intrinsics dict keyed by camera_id, use get_intrinsics_idx() per frame
    - NeRFDataset: single intrinsics list [fx, fy, cx, cy] for all frames

    Returns:
        List of CameraModelParameters (one per frame), or None if extraction fails
    """
    num_frames = len(dataset)
    camera_params = []

    # Check if dataset has ColmapDataset-style intrinsics (dict with camera params)
    if hasattr(dataset, "intrinsics") and isinstance(dataset.intrinsics, dict):
        # ColmapDataset: intrinsics is dict[camera_id] = (params_dict, rays_o, rays_d, camera_name)
        for frame_idx in range(num_frames):
            try:
                # Get camera ID for this frame
                camera_id = dataset.get_intrinsics_idx(frame_idx)
                params_tuple = dataset.intrinsics.get(camera_id)

                if params_tuple is None:
                    logger.warning(f"No intrinsics found for frame {frame_idx}, camera_id {camera_id}")
                    camera_params.append(None)
                    continue

                params_dict, _, _, camera_name, *_ = params_tuple

                # Reconstruct CameraModelParameters from dict
                if camera_name == "OpenCVPinholeCameraModelParameters":
                    params = OpenCVPinholeCameraModelParameters(
                        resolution=np.array(params_dict["resolution"], dtype=np.uint64),
                        shutter_type=ShutterType[params_dict["shutter_type"]],
                        principal_point=np.array(params_dict["principal_point"], dtype=np.float32),
                        focal_length=np.array(params_dict["focal_length"], dtype=np.float32),
                        radial_coeffs=np.array(params_dict["radial_coeffs"], dtype=np.float32),
                        tangential_coeffs=np.array(params_dict["tangential_coeffs"], dtype=np.float32),
                        thin_prism_coeffs=np.array(params_dict["thin_prism_coeffs"], dtype=np.float32),
                    )
                elif camera_name == "OpenCVFisheyeCameraModelParameters":
                    params = OpenCVFisheyeCameraModelParameters(
                        resolution=np.array(params_dict["resolution"], dtype=np.uint64),
                        shutter_type=ShutterType[params_dict["shutter_type"]],
                        principal_point=np.array(params_dict["principal_point"], dtype=np.float32),
                        focal_length=np.array(params_dict["focal_length"], dtype=np.float32),
                        radial_coeffs=np.array(params_dict["radial_coeffs"], dtype=np.float32),
                        max_angle=float(params_dict["max_angle"]),
                    )
                else:
                    logger.warning(f"Unknown camera model type: {camera_name}")
                    camera_params.append(None)
                    continue

                camera_params.append(params)

            except Exception as e:
                logger.warning(f"Failed to extract camera params for frame {frame_idx}: {e}")
                camera_params.append(None)

        return camera_params

    # Check for NeRFDataset-style intrinsics (simple list [fx, fy, cx, cy])
    elif hasattr(dataset, "intrinsics") and isinstance(dataset.intrinsics, list):
        # NeRFDataset: single intrinsics for all frames
        intrinsics_list = dataset.intrinsics
        if len(intrinsics_list) != 4:
            logger.warning(f"Expected intrinsics list [fx, fy, cx, cy], got {len(intrinsics_list)} elements")
            return None

        fx, fy, cx, cy = intrinsics_list

        # Get resolution from dataset (NeRFDataset stores image_w, image_h)
        if hasattr(dataset, "image_w") and hasattr(dataset, "image_h"):
            width, height = dataset.image_w, dataset.image_h
        else:
            # Try to infer from K matrix
            if hasattr(dataset, "K"):
                # K is 3x3, principal point should be roughly at center
                width = int(cx * 2)
                height = int(cy * 2)
            else:
                logger.warning("Cannot determine image resolution for NeRF dataset")
                return None

        # Create a single OpenCVPinhole params (no distortion for NeRF synthetic)
        params = OpenCVPinholeCameraModelParameters(
            resolution=np.array([width, height], dtype=np.uint64),
            shutter_type=ShutterType.GLOBAL,
            principal_point=np.array([cx, cy], dtype=np.float32),
            focal_length=np.array([fx, fy], dtype=np.float32),
            radial_coeffs=np.zeros(6, dtype=np.float32),
            tangential_coeffs=np.zeros(2, dtype=np.float32),
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

        # Same params for all frames
        return [params] * num_frames

    return None


def _extract_camera_grouping(dataset):
    """Extract camera grouping info from a dataset.

    Returns:
        (camera_names, frame_to_camera) where camera_names is a list of logical
        camera names and frame_to_camera maps frame_idx → camera_idx.
    """
    camera_names = None
    frame_to_camera = None

    if hasattr(dataset, "get_camera_names"):
        camera_names = dataset.get_camera_names()
    if hasattr(dataset, "get_camera_idx"):
        frame_to_camera = [dataset.get_camera_idx(i) for i in range(len(dataset))]

    if camera_names is None:
        camera_names = ["camera_0000"]
    if frame_to_camera is None:
        frame_to_camera = [0] * len(dataset)

    return camera_names, frame_to_camera


def _extract_camera_resolutions(camera_params: List, camera_names: List[str], frame_to_camera: List[int]):
    """Extract per-camera resolution from the first valid frame of each camera."""
    result = {}
    num_cameras = len(camera_names)
    first_frame: Dict[int, int] = {}
    for frame_idx, cam_idx in enumerate(frame_to_camera):
        if cam_idx not in first_frame and 0 <= cam_idx < num_cameras:
            first_frame[cam_idx] = frame_idx

    for cam_idx, cam_name in enumerate(camera_names):
        frame_idx = first_frame.get(cam_idx)
        if frame_idx is None or camera_params is None:
            continue
        params = camera_params[frame_idx] if frame_idx < len(camera_params) else None
        if params is None:
            continue
        if hasattr(params, "resolution"):
            w, h = int(params.resolution[0]), int(params.resolution[1])
            result[cam_name] = (w, h)

    return result


def _suffix_camera_names(camera_names: List[str], suffix: str = _VALIDATION_CAMERA_SUFFIX) -> List[str]:
    return [f"{name}{suffix}" for name in camera_names]


def _build_camera_frame_mapping_from_grouping(
    camera_names: List[str], frame_to_camera: List[int]
) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {name: [] for name in camera_names}
    for frame_idx, cam_idx in enumerate(frame_to_camera):
        if 0 <= cam_idx < len(camera_names):
            mapping[camera_names[cam_idx]].append(frame_idx)
    return mapping


def _build_camera_time_mapping_from_grouping(
    camera_names: List[str],
    frame_to_camera: List[int],
) -> Dict[str, List[float]]:
    """Per-camera USD time codes using GLOBAL dataset frame indices.

    Each camera's time code list is the subsequence of global frame indices
    where ``frame_to_camera[i] == cam_idx``. Sparse but uniquely identifies
    the originating dataset frame, so ``(camera, time_code)`` round-trips
    back to a single ``frame_idx`` without any sidecar.

    Required by the OVRTX-vs-PyTorch comparator: PyTorch ``render.py`` writes
    PNG ``<global_frame_idx:05d>.png``; OVRTX writes ``<time_code:06d>.png``;
    matching by integer key works directly across multi-rig validation sets.
    """
    mapping: Dict[str, List[float]] = {name: [] for name in camera_names}
    for global_idx, cam_idx in enumerate(frame_to_camera):
        if 0 <= cam_idx < len(camera_names):
            mapping[camera_names[cam_idx]].append(float(global_idx))
    return mapping


def _build_frame_time_codes_from_grouping(camera_names: List[str], frame_to_camera: List[int]) -> List[float]:
    """Per-frame USD time codes using GLOBAL dataset frame indices.

    Returns ``[0.0, 1.0, ..., N-1]`` matching the dataloader iteration order
    used by ``threedgrut/render.py``. Camera xform time samples and PPISP
    shader time samples both use this convention so they stay aligned at
    runtime.
    """
    return [float(i) for i in range(len(frame_to_camera))]


def _particle_field_render_settings(*, has_runtime_ppisp: bool) -> Dict[str, Any]:
    """Render settings shared with the nre-borel Gaussian USD export."""
    render_settings: Dict[str, Any] = {_RENDER_SETTING_TONEMAP_OP: 2}
    if has_runtime_ppisp:
        render_settings[_RENDER_SETTING_SKIP_GAUSSIAN_TONEMAPPING] = False
    return render_settings


class USDExporter(ModelExporter):
    """
    Exporter for OpenUSD format using ParticleField3DGaussianSplat schema.

    This is the default USD exporter for 3DGRUT. It produces USDZ files
    containing Gaussian splatting data in the standard OpenUSD format.

    Features:
    - ParticleField3DGaussianSplat schema (standard OpenUSD)
    - One Camera prim per physical camera with time-sampled transforms
    - Background/environment export as DomeLight
    - Optional baked-SH post-processing export or PPISP Omniverse native export
    - USDZ packaging (default output)

    For NuRec compatibility, use NuRecExporter instead.
    """

    def __init__(
        self,
        half_precision: bool = False,
        half_geometry: bool = False,
        half_features: bool = False,
        export_cameras: bool = True,
        export_background: bool = True,
        apply_normalizing_transform: bool = True,
        sorting_mode_hint: str = DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT,
        export_post_processing: bool = True,
        ppisp_integration_mode: str | None = None,
        ppisp_reference_camera_id: int | None = None,
        ppisp_reference_frame_id: int | None = None,
        ppisp_responsivity: Any = 1.0,
        enable_ppisp_controller_export: bool | None = None,
        sh_optimization_num_iterations: int | None = None,
        scene_radiance_scale: float | None = None,
        frames_per_second: float = 1.0,
    ):
        """
        Initialize the USD exporter.

        Args:
            half_precision: If True, use half for both geometry and features (backward compat).
            half_geometry: Use half precision for positions, orientations, scales (LightField).
            half_features: Use half precision for opacities and SH coefficients (LightField).
            export_cameras: Include camera poses in export.
            export_background: Include background/environment in export.
            apply_normalizing_transform: Apply transform to normalize scene orientation.
            sorting_mode_hint: Sorting hint for rendering ("zDepth", "cameraDistance", "rayHitDistance").
            export_post_processing: If True, export the checkpoint post-processing
                module with the selected export mode.
            ppisp_integration_mode: "spg-runtime" authors PPISP as Omniverse SPG
                shaders applied at render time. "sh-optimized" bakes one fixed
                PPISP transform into Gaussian SH coefficients. Matches nre-borel.
            ppisp_reference_camera_id: Optional fixed PPISP camera index.
                sh-optimized defaults unset values to 0; spg-runtime keeps
                per-camera RenderProduct behavior when unset. Matches nre-borel.
            ppisp_reference_frame_id: Optional fixed PPISP frame index.
                sh-optimized defaults unset values to 0; spg-runtime keeps
                animated frame inputs when unset. Matches nre-borel.
            ppisp_responsivity: Achromatic input HDR multiplier authored on
                PPISP spg-runtime shaders as a user-overridable default.
            enable_ppisp_controller_export: nre-borel tri-state controller
                export setting. None exports controllers only when the PPISP
                module has them; True requires controller export; False forces
                the static/time-sampled PPISP fallback.
            sh_optimization_num_iterations: nre-borel SH-match step budget
                for sh-optimized export. Defaults to 3000.
            scene_radiance_scale: nre-borel asset-level multiplicative scale
                applied to exported SH radiance.
            frames_per_second: Sets stage.timeCodesPerSecond. Multi-camera
                camera exports use compact per-camera-local frame time codes, so
                this controls playback speed. Default 1.0 means 1 frame per
                second of real time.
        """
        if half_precision:
            half_geometry = True
            half_features = True
        self.half_geometry = half_geometry
        self.half_features = half_features
        self.export_cameras = export_cameras
        self.export_background = export_background
        self.apply_normalizing_transform = apply_normalizing_transform
        self.sorting_mode_hint = normalize_particle_field_sorting_mode_hint(sorting_mode_hint)
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
        self.frames_per_second = frames_per_second

    def _create_default_stage(
        self,
        referenced_stages: List[NamedUSDStage],
        *,
        has_runtime_ppisp: bool = False,
    ) -> NamedUSDStage:
        """
        Create a default.usda that references the data stages.
        """
        stage = initialize_usd_stage(up_axis="Y")
        stage.SetTimeCodesPerSecond(self.frames_per_second)
        stage.SetMetadataByDictKey(
            "customLayerData",
            "renderSettings",
            _particle_field_render_settings(has_runtime_ppisp=has_runtime_ppisp),
        )
        authored_ranges = [
            (ref_stage.stage.GetStartTimeCode(), ref_stage.stage.GetEndTimeCode())
            for ref_stage in referenced_stages
            if getattr(ref_stage.stage, "HasAuthoredTimeCodeRange", None) and ref_stage.stage.HasAuthoredTimeCodeRange()
        ]
        if authored_ranges:
            stage.SetStartTimeCode(min(start for start, _ in authored_ranges))
            stage.SetEndTimeCode(max(end for _, end in authored_ranges))

        for ref_stage in referenced_stages:
            filename_stem = Path(ref_stage.filename).stem
            prim_path = f"/World/{filename_stem}"
            prim = stage.OverridePrim(prim_path)
            prim.GetReferences().AddReference(ref_stage.filename)

        return NamedUSDStage(filename="default.usda", stage=stage)

    @torch.no_grad()
    def export(
        self,
        model: ExportableModel,
        output_path: Path,
        dataset=None,
        conf: Dict[str, Any] = None,
        background=None,
        **kwargs,
    ) -> None:
        """
        Export the model to a USDZ file.

        Args:
            model: The model to export (must implement ExportableModel).
            output_path: Path where the USDZ file will be saved.
            dataset: Optional dataset for camera poses.
            conf: Configuration parameters.
            background: Optional background model for environment export.
            **kwargs:
                post_processing: checkpoint post-processing module to bake or export natively.
                validate_usd (default True): run OpenUSD stage validators.
                apply_coordinate_transform (bool): apply 3DGRUT→USDZ coordinate flip.
                copy_source_usd: (stage_path, res_root) for prim merge.
                copy_source_skip_subtrees: subtrees to skip during prim merge.
        """
        output_path = Path(output_path)
        logger.info(f"Exporting USD file to {output_path}...")
        post_processing = kwargs.get("post_processing")
        validation_dataset = kwargs.get("validation_dataset")
        has_ppisp_module = _is_ppisp_post_processing(post_processing)
        runtime_post_processing = compute_runtime_post_processing(
            model_has_postprocessing=post_processing is not None and self.export_post_processing,
            ppisp_module=post_processing if has_ppisp_module else None,
            ppisp_integration_mode=self.ppisp_integration_mode,
        )
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
                "Baking post-processing into Gaussian SH coefficients before export "
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

        # User-requested constant radiance scale, applied uniformly to the
        # SH output regardless of bake / colour-space mode. The DC offset
        # baked into RGB2SH is compensated so a forward eval reproduces
        # scene_radiance_scale * (original SH-evaluated RGB).
        if self.scene_radiance_scale != 1.0:
            scale_sh_output(model, self.scene_radiance_scale)

        # Get model data via accessor
        accessor = GaussianExportAccessor(model, conf)
        attrs = accessor.get_attributes(preactivation=False)
        caps = accessor.get_capabilities()

        logger.info(f"Schema: LightField (post-activation)")
        logger.info(f"Exporting {attrs.num_gaussians} Gaussians, SH degree {caps.sh_degree}")

        # Compute normalizing transform if enabled
        normalizing_transform = np.eye(4)
        if self.apply_normalizing_transform and dataset is not None:
            try:
                poses = dataset.get_poses()
                normalizing_transform = estimate_normalizing_transform(poses)
                logger.info("Computed normalizing transform from camera poses")
            except (AttributeError, ValueError) as e:
                logger.warning(f"Failed to compute normalizing transform: {e}")

        # Create main USD stage with the configured time code rate
        stage = initialize_usd_stage(up_axis="Y")
        stage.SetTimeCodesPerSecond(self.frames_per_second)

        apply_coordinate_transform = kwargs.get("apply_coordinate_transform", False)
        coordinate_transform = get_3dgrut_to_usdz_coordinate_transform() if apply_coordinate_transform else None

        # Create Gaussian content root
        gaussians_root = create_gaussian_model_root(
            stage,
            flip_x_axis=False,
            flip_y_axis=False,
            flip_z_axis=False,
            root_path="/World/Gaussians",
            normalizing_transform=normalizing_transform if self.apply_normalizing_transform else None,
            coordinate_transform=coordinate_transform,
            source_transform_samples=kwargs.get("source_gaussian_transform"),
        )

        # Write Gaussians
        writer = create_gaussian_writer(
            stage=stage,
            capabilities=caps,
            content_root_path=gaussians_root,
            half_geometry=self.half_geometry,
            half_features=self.half_features,
            sorting_mode_hint=self.sorting_mode_hint,
            linear_srgb=runtime_post_processing,
            omni_usd=runtime_post_processing,
            has_post_processing=runtime_post_processing,
        )
        writer.create_prim(attrs.num_gaussians)
        writer.write_attributes(attrs)
        writer.finalize(attrs.positions)

        suffix = output_path.suffix.lower()
        package_as_usdz = suffix == ".usdz" or suffix not in (".usd", ".usda", ".usdc")

        gaussians_stage = NamedUSDStage(filename="gaussians.usdc", stage=stage)
        default_stage_wrapped: Optional[NamedUSDStage] = None
        if package_as_usdz:
            default_stage_wrapped = self._create_default_stage(
                [gaussians_stage],
                has_runtime_ppisp=runtime_post_processing,
            )
        scene_stage = default_stage_wrapped.stage if default_stage_wrapped is not None else stage
        if not package_as_usdz:
            scene_stage.SetMetadataByDictKey(
                "customLayerData",
                "renderSettings",
                _particle_field_render_settings(has_runtime_ppisp=runtime_post_processing),
            )

        files: List[NamedSerialized] = []

        copy_source_usd = kwargs.get("copy_source_usd")
        if copy_source_usd is None:
            copy_source_usd = kwargs.get("copy_cameras_source")
        if copy_source_usd is not None:
            stage_path, res_root = copy_source_usd
            try:
                src_stage = Usd.Stage.Open(str(stage_path))
                if not src_stage:
                    logger.warning("Could not open source USD for prim merge: %s", stage_path)
                else:
                    skip = kwargs.get("copy_source_skip_subtrees")
                    merge_source_prims_and_collect_sidecars(
                        dest_stage=scene_stage,
                        source_stage=src_stage,
                        res_root=res_root,
                        source_stage_path=Path(stage_path),
                        files=files,
                        skip_source_subtrees=skip,
                    )
            except Exception as e:
                logger.warning("Failed to merge source USD prims: %s", e)

        # Extract camera grouping from dataset (used by both camera export and PPISP)
        camera_names = None
        frame_to_camera = None
        camera_prim_paths: Dict[str, str] = {}
        camera_params = None
        camera_time_mapping = None
        validation_camera_names = None
        validation_frame_to_camera = None
        validation_camera_prim_paths: Dict[str, str] = {}
        validation_camera_params = None
        validation_camera_time_mapping = None

        if dataset is not None:
            camera_names, frame_to_camera = _extract_camera_grouping(dataset)
            camera_time_mapping = _build_camera_time_mapping_from_grouping(camera_names, frame_to_camera)
        if validation_dataset is not None:
            validation_camera_names, validation_frame_to_camera = _extract_camera_grouping(validation_dataset)
            validation_camera_names = _suffix_camera_names(validation_camera_names)
            validation_camera_time_mapping = _build_camera_time_mapping_from_grouping(
                validation_camera_names,
                validation_frame_to_camera,
            )

        # Export cameras — one prim per physical camera with time-sampled transforms
        if self.export_cameras and dataset is not None:
            try:
                poses = dataset.get_poses()

                if self.apply_normalizing_transform:
                    poses = np.einsum("ij,njk->nik", normalizing_transform, poses)

                camera_params = _extract_camera_params_from_dataset(dataset)
                if camera_params is not None:
                    logger.info(f"Extracted camera params for {len(camera_params)} frames")
                else:
                    logger.warning("Could not extract camera intrinsics from dataset, using default")

                camera_prim_paths = export_cameras_to_usd(
                    stage=scene_stage,
                    poses=poses,
                    camera_names=camera_names,
                    frame_to_camera=frame_to_camera,
                    camera_params=camera_params,
                    frame_time_codes=_build_frame_time_codes_from_grouping(camera_names, frame_to_camera),
                    root_path="/World/Cameras",
                    visible=False,
                )
                logger.info(f"Exported {len(camera_prim_paths)} camera(s) from {len(poses)} frames")
            except (AttributeError, KeyError, ValueError) as e:
                raise ValueError(f"Failed to export cameras: {e}") from e

        if self.export_cameras and validation_dataset is not None:
            try:
                validation_poses = validation_dataset.get_poses()

                if self.apply_normalizing_transform:
                    validation_poses = np.einsum("ij,njk->nik", normalizing_transform, validation_poses)

                validation_camera_params = _extract_camera_params_from_dataset(validation_dataset)
                if validation_camera_params is not None:
                    logger.info(f"Extracted validation camera params for {len(validation_camera_params)} frames")
                else:
                    logger.warning("Could not extract validation camera intrinsics, skipping validation RenderProducts")

                previous_start = scene_stage.GetStartTimeCode()
                previous_end = scene_stage.GetEndTimeCode()
                validation_camera_prim_paths = export_cameras_to_usd(
                    stage=scene_stage,
                    poses=validation_poses,
                    camera_names=validation_camera_names,
                    frame_to_camera=validation_frame_to_camera,
                    camera_params=validation_camera_params,
                    frame_time_codes=_build_frame_time_codes_from_grouping(
                        validation_camera_names,
                        validation_frame_to_camera,
                    ),
                    root_path="/World/Cameras",
                    visible=False,
                )
                scene_stage.SetStartTimeCode(min(previous_start, scene_stage.GetStartTimeCode()))
                scene_stage.SetEndTimeCode(max(previous_end, scene_stage.GetEndTimeCode()))
                logger.info(
                    f"Exported {len(validation_camera_prim_paths)} validation camera(s) "
                    f"from {len(validation_poses)} frames"
                )
            except (AttributeError, KeyError, ValueError) as e:
                raise ValueError(f"Failed to export validation cameras: {e}") from e

        if (
            self.export_cameras
            and dataset is not None
            and camera_prim_paths
            and not uses_omni_native_post_processing_export
        ):
            camera_render_products = self._create_camera_render_products(
                stage=scene_stage,
                camera_names=camera_names,
                frame_to_camera=frame_to_camera,
                camera_prim_paths=camera_prim_paths,
                camera_params=camera_params,
                render_vars=(_DEFAULT_RENDER_PRODUCT_VAR,),
            )
            if not camera_render_products:
                raise ValueError(
                    "export_cameras=True created cameras but no RenderProducts. "
                    "Check that dataset intrinsics include native image resolution."
                )
        if (
            self.export_cameras
            and validation_dataset is not None
            and validation_camera_prim_paths
            and not uses_omni_native_post_processing_export
        ):
            validation_render_products = self._create_camera_render_products(
                stage=scene_stage,
                camera_names=validation_camera_names,
                frame_to_camera=validation_frame_to_camera,
                camera_prim_paths=validation_camera_prim_paths,
                camera_params=validation_camera_params,
                render_vars=(_DEFAULT_RENDER_PRODUCT_VAR,),
            )
            if not validation_render_products:
                raise ValueError(
                    "export_cameras=True created validation cameras but no validation RenderProducts. "
                    "Check that validation dataset intrinsics include native image resolution."
                )

        # Export background if requested
        envmap_bytes = None
        if self.export_background and background is not None:
            try:
                _, envmap_bytes = export_background_to_usd(
                    stage=scene_stage,
                    background=background,
                    conf=conf,
                    root_path="/World/Environment",
                    envmap_filename="envmap.png",
                )
                if envmap_bytes is not None:
                    files.append(NamedSerialized(filename="envmap.png", serialized=envmap_bytes))
                    logger.info("Exported background as DomeLight")
            except (AttributeError, ValueError, ImportError) as e:
                logger.warning(f"Failed to export background: {e}")

        if not self.export_post_processing and _is_ppisp_post_processing(post_processing):
            logger.warning(
                "PPISP post-processing module is present but export_usd.export_post_processing=false; "
                "PPISP effects will not be exported. Set export_usd.export_post_processing=true to export them."
            )
        if self.export_post_processing and post_processing is None:
            logger.info("Post-processing export requested but no post_processing module is available; skipping bake")

        if uses_omni_native_post_processing_export:
            render_product_entries = self._create_ppisp_render_products(
                stage=scene_stage,
                dataset=dataset,
                camera_names=camera_names,
                frame_to_camera=frame_to_camera,
                camera_prim_paths=camera_prim_paths,
                camera_params=camera_params,
            )
            if not render_product_entries:
                raise ValueError(
                    "export_cameras=True could not create PPISP RenderProducts. "
                    "Check that dataset cameras and intrinsics are available."
                )
            if render_product_entries is not None:
                if validation_dataset is not None and validation_camera_prim_paths:
                    validation_render_products = self._create_camera_render_products(
                        stage=scene_stage,
                        camera_names=validation_camera_names,
                        frame_to_camera=validation_frame_to_camera,
                        camera_prim_paths=validation_camera_prim_paths,
                        camera_params=validation_camera_params,
                        render_vars=(_PPISP_INPUT_RENDER_PRODUCT_VAR,),
                    )
                    if not validation_render_products:
                        raise ValueError(
                            "export_cameras=True could not create validation PPISP RenderProducts. "
                            "Check that validation dataset cameras and intrinsics are available."
                        )
                self._export_ppisp(
                    stage=scene_stage,
                    dataset=dataset,
                    camera_names=camera_names,
                    post_processing=post_processing,
                    files=files,
                    fixed_camera_id=None if enable_ppisp_controller_export else self.ppisp_reference_camera_id,
                    fixed_frame_id=None if enable_ppisp_controller_export else self.ppisp_reference_frame_id,
                    responsivity=self.ppisp_responsivity,
                    camera_time_mapping=camera_time_mapping,
                    enable_ppisp_controller_export=enable_ppisp_controller_export,
                )
                if validation_dataset is not None and validation_camera_names is not None:
                    validation_mapping = _build_camera_frame_mapping_from_grouping(
                        validation_camera_names,
                        validation_frame_to_camera,
                    )
                    self._export_ppisp(
                        stage=scene_stage,
                        dataset=validation_dataset,
                        camera_names=validation_camera_names,
                        post_processing=post_processing,
                        files=files,
                        fixed_camera_id=None if enable_ppisp_controller_export else self.ppisp_reference_camera_id,
                        fixed_frame_id=None if enable_ppisp_controller_export else self.ppisp_reference_frame_id,
                        responsivity=self.ppisp_responsivity,
                        camera_frame_mapping=validation_mapping,
                        camera_time_mapping=validation_camera_time_mapping,
                        neutral_frame_params=(
                            self.ppisp_reference_camera_id is None and self.ppisp_reference_frame_id is None
                        ),
                        enable_ppisp_controller_export=enable_ppisp_controller_export,
                    )

        # Package
        if suffix == ".usdz":
            if default_stage_wrapped is None:
                default_stage_wrapped = self._create_default_stage(
                    [gaussians_stage],
                    has_runtime_ppisp=runtime_post_processing,
                )
            write_to_usdz(output_path, [default_stage_wrapped, gaussians_stage], files if files else None)
            written_path = output_path
        elif suffix in [".usda", ".usd", ".usdc"]:
            stage.Export(str(output_path))
            if files:
                save_serialized_files(files, output_path.parent)
            written_path = output_path
        else:
            usdz_path = output_path.with_suffix(".usdz")
            if default_stage_wrapped is None:
                default_stage_wrapped = self._create_default_stage(
                    [gaussians_stage],
                    has_runtime_ppisp=runtime_post_processing,
                )
            write_to_usdz(usdz_path, [default_stage_wrapped, gaussians_stage], files if files else None)
            written_path = usdz_path

        if kwargs.get("validate_usd", True):
            from threedgrut.export.usd.validation import validate_exported_usd_stage

            validate_exported_usd_stage(written_path)

        logger.info(f"USD export complete: {output_path}")

    def _create_ppisp_render_products(
        self,
        stage,
        dataset,
        camera_names,
        frame_to_camera,
        camera_prim_paths: Dict[str, str],
        camera_params,
    ):
        """Create /Render RenderProducts for PPISP spg-runtime export."""
        if dataset is None or not camera_prim_paths:
            logger.warning("No camera prims available for PPISP RenderProduct wiring, skipping")
            return None

        resolutions = _extract_camera_resolutions(camera_params, camera_names, frame_to_camera)
        camera_entries = {}
        for cam_name, cam_path in camera_prim_paths.items():
            if cam_name not in resolutions:
                logger.warning("No native resolution found for camera %s; skipping RenderProduct", cam_name)
                continue
            w, h = resolutions[cam_name]
            camera_entries[cam_name] = (cam_path, w, h)
        if not camera_entries:
            logger.warning("No camera RenderProducts created because no native camera resolutions were found")
            return None

        try:
            create_render_products(
                stage=stage,
                camera_entries=camera_entries,
                render_vars=(_PPISP_INPUT_RENDER_PRODUCT_VAR,),
            )
        except Exception as e:
            logger.warning(f"Failed to create RenderProducts: {e}")
            return None

        return camera_entries

    def _create_camera_render_products(
        self,
        stage,
        camera_names,
        frame_to_camera,
        camera_prim_paths: Dict[str, str],
        camera_params,
        render_vars=(_DEFAULT_RENDER_PRODUCT_VAR,),
    ):
        """Create per-camera RenderProducts for exported cameras."""
        if not camera_prim_paths:
            logger.warning("No camera prims available for RenderProduct wiring, skipping")
            return None

        resolutions = _extract_camera_resolutions(camera_params, camera_names, frame_to_camera)
        camera_entries = {}
        for cam_name, cam_path in camera_prim_paths.items():
            if cam_name not in resolutions:
                logger.warning("No native resolution found for camera %s; skipping RenderProduct", cam_name)
                continue
            w, h = resolutions[cam_name]
            camera_entries[cam_name] = (cam_path, w, h)
        if not camera_entries:
            logger.warning("No camera RenderProducts created because no native camera resolutions were found")
            return None

        try:
            create_render_products(
                stage=stage,
                camera_entries=camera_entries,
                render_vars=render_vars,
            )
        except Exception as e:
            logger.warning(f"Failed to create RenderProducts: {e}")
            return None

        return camera_entries

    def _export_ppisp(
        self,
        stage,
        dataset,
        camera_names,
        post_processing,
        files: List[NamedSerialized],
        fixed_camera_id: int | None = None,
        fixed_frame_id: int | None = None,
        responsivity: float = 1.0,
        camera_frame_mapping: Dict[str, List[int]] | None = None,
        camera_time_mapping: Dict[str, List[float]] | None = None,
        neutral_frame_params: bool = False,
        enable_ppisp_controller_export: bool = False,
    ) -> None:
        """Attach PPISP SPG shaders to existing RenderProducts."""
        try:
            from ppisp import PPISP  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("ppisp package not available, skipping PPISP export")
            return

        if not isinstance(post_processing, PPISP):
            logger.warning(
                f"export_post_processing=True but post_processing is {type(post_processing).__name__}, "
                "expected ppisp.PPISP - skipping"
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

        if not enable_ppisp_controller_export and fixed_frame_id is not None and fixed_camera_id is None:
            raise ValueError(
                "ppisp_reference_frame_id was set without ppisp_reference_camera_id "
                "in spg-runtime export mode. Set ppisp_reference_camera_id as well, "
                "or leave both unset for time-sampled SPG authoring."
            )

        try:
            add_ppisp_to_all_render_products(
                stage=stage,
                ppisp=post_processing,
                camera_names=camera_names,
                camera_frame_mapping=camera_frame_mapping,
                camera_time_mapping=camera_time_mapping,
                fixed_camera_index=fixed_camera_id,
                fixed_frame_index=fixed_frame_id,
                use_controller=enable_ppisp_controller_export,
                responsivity=responsivity,
                neutral_frame_params=neutral_frame_params and not enable_ppisp_controller_export,
            )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Failed to add PPISP shaders: {e}")
            return

        spg_files = select_spg_files_for_export(
            enable_ppisp_controller_export=enable_ppisp_controller_export,
            ppisp_module=post_processing if enable_ppisp_controller_export else None,
            camera_indices=(
                range(len(getattr(post_processing, "controllers", []))) if enable_ppisp_controller_export else None
            ),
        )

        for spg_file in spg_files:
            if not any(f.filename == spg_file.filename for f in files):
                files.append(spg_file)

        logger.info(
            "PPISP spg-runtime export complete: %d sidecar(s) added (controller=%s)",
            len(files),
            enable_ppisp_controller_export,
        )

    @classmethod
    def from_config(cls, conf) -> "USDExporter":
        """
        Create USDExporter from configuration.
        """
        export_conf = getattr(conf, "export_usd", None) or conf
        half_precision = getattr(export_conf, "half_precision", False)
        half_geometry = getattr(export_conf, "half_geometry", False)
        half_features = getattr(export_conf, "half_features", False)
        if half_precision:
            half_geometry = True
            half_features = True
        return cls(
            half_geometry=half_geometry,
            half_features=half_features,
            export_cameras=getattr(export_conf, "export_cameras", True),
            export_background=getattr(export_conf, "export_background", True),
            apply_normalizing_transform=getattr(export_conf, "apply_normalizing_transform", True),
            sorting_mode_hint=getattr(export_conf, "sorting_mode_hint", DEFAULT_PARTICLE_FIELD_SORTING_MODE_HINT),
            export_post_processing=_get_export_config_value(
                export_conf,
                "export-post-processing",
                "export_post_processing",
                True,
            ),
            ppisp_integration_mode=_get_export_config_value(
                export_conf,
                "ppisp-integration-mode",
                "ppisp_integration_mode",
                None,
            ),
            ppisp_reference_camera_id=_get_export_config_value(
                export_conf,
                "ppisp-reference-camera-id",
                "ppisp_reference_camera_id",
                None,
            ),
            ppisp_reference_frame_id=_get_export_config_value(
                export_conf,
                "ppisp-reference-frame-id",
                "ppisp_reference_frame_id",
                None,
            ),
            ppisp_responsivity=_get_export_config_value(
                export_conf,
                "ppisp-responsivity",
                "ppisp_responsivity",
                1.0,
            ),
            enable_ppisp_controller_export=_get_export_config_value(
                export_conf,
                "enable-ppisp-controller-export",
                "enable_ppisp_controller_export",
                None,
            ),
            sh_optimization_num_iterations=_get_export_config_value(
                export_conf,
                "sh-optimization-num-iterations",
                "sh_optimization_num_iterations",
                None,
            ),
            scene_radiance_scale=_get_export_config_value(
                export_conf,
                "scene-radiance-scale",
                "scene_radiance_scale",
                None,
            ),
            frames_per_second=_get_export_config_value(
                export_conf,
                "frames-per-second",
                "frames_per_second",
                1.0,
            ),
        )
