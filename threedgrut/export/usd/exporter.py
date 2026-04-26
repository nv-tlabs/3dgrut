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
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pxr import Usd
from ncore.data import (
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ShutterType,
)

from threedgrut.export.accessor import GaussianExportAccessor
from threedgrut.export.base import ExportableModel, ModelExporter
from threedgrut.export.transforms import (
    estimate_normalizing_transform,
    get_3dgrut_to_usdz_coordinate_transform,
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
from threedgrut.export.usd.camera_copy import (
    collect_transitive_sidecars_for_subtree,
    copy_authored_time_settings_from_source,
    merge_source_prim_at_same_path,
    merge_source_world_at_same_paths,
)
from threedgrut.export.usd.writers.ov_post_processing import (
    MODE_PPISP_OMNI_FALLBACK_NONE,
    MODE_PPISP_OMNI_FALLBACK_SPG_PLUS_FITTED_POST_PROCESSING,
    normalize_ov_post_processing_mode,
)
from threedgrut.export.usd.writers.camera import export_cameras_to_usd

logger = logging.getLogger(__name__)


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

                params_dict, _, _, camera_name = params_tuple

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
    """Extract per-camera resolution from the first valid frame of each camera.

    Returns:
        {camera_name: (width, height)} or empty dict on failure.
    """
    result = {}
    num_cameras = len(camera_names)
    # Build first-frame-per-camera map
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


class USDExporter(ModelExporter):
    """
    Exporter for OpenUSD format using ParticleField3DGaussianSplat schema.

    This is the default USD exporter for 3DGRUT. It produces USDZ files
    containing Gaussian splatting data in the standard OpenUSD format.

    Features:
    - ParticleField3DGaussianSplat schema (standard OpenUSD)
    - One Camera prim per physical camera with time-sampled transforms
    - Background/environment export as DomeLight
    - Optional PPISP SPG shader on per-camera RenderProducts
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
        sorting_mode_hint: str = "cameraDistance",
        linear_srgb: bool = False,
        export_ppisp: bool = False,
        ov_post_processing: str = MODE_PPISP_OMNI_FALLBACK_NONE,
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
            sorting_mode_hint: Sorting hint for rendering ("cameraDistance", "zDepth").
            linear_srgb: If True, set prim color space to lin_rec709_scene.
            export_ppisp: If True, export PPISP using SPG or the selected
                Omniverse USD fallback mode. Requires post_processing kwarg to
                be a ppisp.PPISP instance.
            ov_post_processing: PPISP export implementation selector. "none"
                uses the full SPG path when export_ppisp is enabled.
            frames_per_second: Sets stage.timeCodesPerSecond. Time codes are always
                bare frame indices (float(frame_idx)), so this controls playback speed.
                Default 1.0 means 1 frame per second of real time.
        """
        if half_precision:
            half_geometry = True
            half_features = True
        self.half_geometry = half_geometry
        self.half_features = half_features
        self.export_cameras = export_cameras
        self.export_background = export_background
        self.apply_normalizing_transform = apply_normalizing_transform
        self.sorting_mode_hint = sorting_mode_hint
        self.linear_srgb = linear_srgb
        self.export_ppisp = export_ppisp
        self.ov_post_processing = normalize_ov_post_processing_mode(ov_post_processing)
        if not self.export_ppisp and self.ov_post_processing != MODE_PPISP_OMNI_FALLBACK_NONE:
            raise ValueError(
                "export_usd.ov-post-processing requires export_usd.export_ppisp=true. "
                "Set export_ppisp=true to export PPISP through an Omniverse USD fallback, "
                "or set ov-post-processing=none."
            )
        self.frames_per_second = frames_per_second

    def _create_default_stage(self, referenced_stages: List[NamedUSDStage]) -> NamedUSDStage:
        """
        Create a default.usda that references the data stages.
        """
        stage = initialize_usd_stage(up_axis="Y")

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
                post_processing: ppisp.PPISP instance for SPG export (used when
                    export_ppisp=True or ov-post-processing is enabled).
                validate_usd (default True): run OpenUSD stage validators.
                apply_coordinate_transform (bool): apply 3DGRUT→USDZ coordinate flip.
                copy_source_usd: (stage_path, res_root) for prim merge.
                copy_source_skip_subtrees: subtrees to skip during prim merge.
        """
        output_path = Path(output_path)
        logger.info(f"Exporting USD file to {output_path}...")

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
        )

        # Write Gaussians
        writer = create_gaussian_writer(
            stage=stage,
            capabilities=caps,
            content_root_path=gaussians_root,
            half_geometry=self.half_geometry,
            half_features=self.half_features,
            sorting_mode_hint=self.sorting_mode_hint,
            linear_srgb=self.linear_srgb,
        )
        writer.create_prim(attrs.num_gaussians)
        writer.write_attributes(attrs)
        writer.finalize(attrs.positions)

        suffix = output_path.suffix.lower()
        package_as_usdz = suffix == ".usdz" or suffix not in (".usd", ".usda", ".usdc")

        gaussians_stage = NamedUSDStage(filename="gaussians.usdc", stage=stage)
        default_stage_wrapped: Optional[NamedUSDStage] = None
        if package_as_usdz:
            default_stage_wrapped = self._create_default_stage([gaussians_stage])

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
                    merge_target = default_stage_wrapped.stage if default_stage_wrapped is not None else stage
                    merge_source_world_at_same_paths(merge_target, src_stage, skip_source_subtrees=skip)
                    merge_source_prim_at_same_path(merge_target, src_stage, "/Render")
                    copy_authored_time_settings_from_source(src_stage, merge_target)
                    if package_as_usdz and res_root is not None and res_root.is_dir():
                        for path_prefix in ("/World", "/Render"):
                            sidecars = collect_transitive_sidecars_for_subtree(
                                merge_target.GetRootLayer(),
                                res_root,
                                path_prefix=path_prefix,
                                extra_skip_names={Path(stage_path).name},
                            )
                            for entry in sidecars:
                                if not any(f.filename == entry.filename for f in files):
                                    files.append(entry)
            except Exception as e:
                logger.warning("Failed to merge source USD prims: %s", e)

        # Extract camera grouping from dataset (used by both camera export and PPISP)
        camera_names = None
        frame_to_camera = None
        camera_prim_paths: Dict[str, str] = {}
        camera_params = None

        if dataset is not None:
            camera_names, frame_to_camera = _extract_camera_grouping(dataset)

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
                    stage=stage,
                    poses=poses,
                    camera_names=camera_names,
                    frame_to_camera=frame_to_camera,
                    camera_params=camera_params,
                    root_path="/World/Cameras",
                    visible=False,
                )
                logger.info(f"Exported {len(camera_prim_paths)} camera(s) from {len(poses)} frames")
            except (AttributeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to export cameras: {e}")

        # Export background if requested
        envmap_bytes = None
        if self.export_background and background is not None:
            try:
                _, envmap_bytes = export_background_to_usd(
                    stage=stage,
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

        render_product_entries = None
        export_spg_ppisp = self.export_ppisp and (
            self.ov_post_processing == MODE_PPISP_OMNI_FALLBACK_NONE
            or self.ov_post_processing == MODE_PPISP_OMNI_FALLBACK_SPG_PLUS_FITTED_POST_PROCESSING
        )
        export_omni_ppisp_fallback = (
            self.export_ppisp and self.ov_post_processing != MODE_PPISP_OMNI_FALLBACK_NONE
        )
        needs_ppisp_render_products = self.export_ppisp
        if needs_ppisp_render_products:
            render_product_entries = self._create_ppisp_render_products(
                stage=stage,
                dataset=dataset,
                camera_names=camera_names,
                frame_to_camera=frame_to_camera,
                camera_prim_paths=camera_prim_paths,
                camera_params=camera_params,
            )

        # Export PPISP as SPG shaders on RenderProducts
        if export_spg_ppisp and render_product_entries is not None:
            self._export_ppisp(
                stage=stage,
                dataset=dataset,
                camera_names=camera_names,
                post_processing=kwargs.get("post_processing"),
                files=files,
            )

        # Export fitted PPISP approximation as Omniverse USD post-processing settings
        if export_omni_ppisp_fallback and render_product_entries is not None:
            self._export_ov_post_processing(
                stage=stage,
                camera_names=camera_names,
                camera_prim_paths=camera_prim_paths,
                render_product_entries=render_product_entries,
                dataset=dataset,
                post_processing=kwargs.get("post_processing"),
            )

        # Package
        if suffix == ".usdz":
            if default_stage_wrapped is None:
                default_stage_wrapped = self._create_default_stage([gaussians_stage])
            write_to_usdz(output_path, [default_stage_wrapped, gaussians_stage], files if files else None)
            written_path = output_path
        elif suffix in [".usda", ".usd", ".usdc"]:
            stage.Export(str(output_path))
            if envmap_bytes is not None:
                envmap_path = output_path.parent / "envmap.png"
                with open(envmap_path, "wb") as f:
                    f.write(envmap_bytes)
            written_path = output_path
        else:
            usdz_path = output_path.with_suffix(".usdz")
            if default_stage_wrapped is None:
                default_stage_wrapped = self._create_default_stage([gaussians_stage])
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
        """Create /Render RenderProducts shared by SPG and OV PPISP exports."""
        if dataset is None or not camera_prim_paths:
            logger.warning("No camera prims available for PPISP RenderProduct wiring, skipping")
            return None

        from threedgrut.export.usd.writers.render_product import create_render_products

        resolutions = _extract_camera_resolutions(camera_params, camera_names, frame_to_camera)
        camera_entries = {}
        for cam_name, cam_path in camera_prim_paths.items():
            w, h = resolutions.get(cam_name, (0, 0))
            camera_entries[cam_name] = (cam_path, w, h)

        try:
            create_render_products(stage=stage, camera_entries=camera_entries)
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
    ) -> None:
        """Attach PPISP SPG shaders to existing RenderProducts."""
        try:
            from ppisp import PPISP  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("ppisp package not available, skipping PPISP export")
            return

        if not isinstance(post_processing, PPISP):
            logger.warning(
                f"export_ppisp=True but post_processing is {type(post_processing).__name__}, "
                "expected ppisp.PPISP — skipping"
            )
            return

        from threedgrut.export.usd.writers.ppisp_writer import (
            add_ppisp_to_all_render_products,
            build_camera_frame_mapping,
        )
        from threedgrut.export.usd.ppisp_spg import get_ppisp_spg_files

        _, camera_frame_mapping = build_camera_frame_mapping(dataset)

        try:
            add_ppisp_to_all_render_products(
                stage=stage,
                ppisp=post_processing,
                camera_names=camera_names,
                camera_frame_mapping=camera_frame_mapping,
            )
        except Exception as e:
            logger.warning(f"Failed to add PPISP shaders: {e}")
            return

        # Add SPG sidecars to the USDZ package
        spg_files = get_ppisp_spg_files()
        for spg_file in spg_files:
            if not any(f.filename == spg_file.filename for f in files):
                files.append(spg_file)

        logger.info(f"PPISP SPG export complete: {len(spg_files)} sidecar(s) added")

    def _export_ov_post_processing(
        self,
        stage,
        camera_names,
        camera_prim_paths,
        render_product_entries,
        dataset,
        post_processing,
    ) -> None:
        """Attach Omniverse USD post-processing fallback attributes to RenderProducts."""
        try:
            from ppisp import PPISP  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("ppisp package not available, skipping OV post-processing export")
            return

        if not isinstance(post_processing, PPISP):
            logger.warning(
                f"ov-post-processing={self.ov_post_processing} but post_processing is "
                f"{type(post_processing).__name__}, expected ppisp.PPISP — skipping"
            )
            return

        from threedgrut.export.usd.writers.ov_post_processing import add_ov_post_processing
        from threedgrut.export.usd.writers.ppisp_writer import build_camera_frame_mapping

        _, camera_frame_mapping = build_camera_frame_mapping(dataset)
        try:
            add_ov_post_processing(
                stage=stage,
                camera_names=camera_names,
                camera_prim_paths=camera_prim_paths,
                camera_frame_mapping=camera_frame_mapping,
                render_product_entries=render_product_entries,
                post_processing=post_processing,
                mode=self.ov_post_processing,
            )
        except Exception as e:
            logger.warning(f"Failed to add OV post-processing fallback: {e}")

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
            sorting_mode_hint=getattr(export_conf, "sorting_mode_hint", "cameraDistance"),
            linear_srgb=getattr(export_conf, "linear_srgb", False),
            export_ppisp=getattr(export_conf, "export_ppisp", False),
            ov_post_processing=export_conf.get("ov-post-processing", MODE_PPISP_OMNI_FALLBACK_NONE)
            if hasattr(export_conf, "get")
            else getattr(export_conf, "ov_post_processing", MODE_PPISP_OMNI_FALLBACK_NONE),
            frames_per_second=getattr(export_conf, "frames_per_second", 1.0),
        )
