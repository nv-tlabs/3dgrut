"""BBox filtering and export helpers for the GLFW viewer.

The exporter isolates gaussian visibility filtering, temporary export model
construction, and the USDZ/NUREC serialization flow used by the viewer UI.
"""

import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from threedgrut.utils.logger import logger

class Exporter:
    """Apply BBox filtering and export the current scene selection.

    Args:
        engine: Active viewer engine whose gaussian model will be exported.
    """

    BBOX_EMPTY_DENSITY = -100000.0
    EXPORT_TO_OV_MATRIX = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    def __init__(self, engine):
        """Store the engine reference and snapshot original densities.

        Args:
            engine: Active viewer engine whose gaussian model will be exported.
        """
        self.engine = engine
        self._density_backup = self.engine.scene_mog.density.detach().clone()

    @torch.no_grad()
    def apply_bbox_filter(self, min_vals, max_vals):
        """Hide gaussians outside the requested bounding box.

        Args:
            min_vals: Minimum BBox corner in world coordinates.
            max_vals: Maximum BBox corner in world coordinates.

        Returns:
            tuple: ``(ok, kept, total)`` describing the filter result.
        """
        positions = self.engine.scene_mog.positions
        density = self.engine.scene_mog.density
        min_t = torch.as_tensor(min_vals, device=positions.device, dtype=positions.dtype)
        max_t = torch.as_tensor(max_vals, device=positions.device, dtype=positions.dtype)
        is_inside = ((positions >= min_t) & (positions <= max_t)).all(dim=1)
        kept = int(is_inside.sum().item())
        total = int(is_inside.shape[0])
        if kept == 0:
            return False, kept, total
        density.data.copy_(self._density_backup)
        density.data[~is_inside] = self.BBOX_EMPTY_DENSITY
        self.engine.is_materials_dirty = True
        return True, kept, total

    @torch.no_grad()
    def disable_bbox_filter(self):
        """Restore the original gaussian density tensor.

        Returns:
            None: Density values are restored in place.
        """
        self.engine.scene_mog.density.data.copy_(self._density_backup)
        self.engine.is_materials_dirty = True

    def infer_export_method(self, default_config: str) -> str:
        """Infer the export method family from the selected config path.

        Args:
            default_config: Config path or name passed to the viewer.

        Returns:
            str: Either ``3dgut`` or ``3dgrt``.
        """
        return "3dgut" if "3dgut" in str(default_config).lower() else "3dgrt"

    def get_camera_eye_target(self, camera_state: dict):
        """Convert orbit-camera state into eye and target positions.

        Args:
            camera_state: Orbit camera state dictionary from the viewer.

        Returns:
            tuple[np.ndarray, np.ndarray]: Eye and target world positions.
        """
        camera_distance = float(camera_state["distance"])
        camera_phi = float(camera_state["phi"])
        camera_theta = float(camera_state["theta"])
        camera_pan_x = float(camera_state["pan_x"])
        camera_pan_y = float(camera_state["pan_y"])
        camera_pan_z = float(camera_state["pan_z"])

        cam_x = camera_distance * np.sin(camera_phi) * np.cos(camera_theta)
        cam_y = camera_distance * np.cos(camera_phi)
        cam_z = camera_distance * np.sin(camera_phi) * np.sin(camera_theta)

        eye = np.array([cam_x + camera_pan_x, cam_y + camera_pan_y, cam_z + camera_pan_z], dtype=np.float64)
        target = np.array([camera_pan_x, camera_pan_y, camera_pan_z], dtype=np.float64)
        return eye, target

    def build_export_camera_settings(self, camera_state: dict, camera_fov: float):
        """Build Omniverse-style camera settings for USDZ export.

        Args:
            camera_state: Orbit camera state dictionary from the viewer.
            camera_fov: Vertical field of view in degrees.

        Returns:
            dict: Camera settings payload consumed by the USDZ exporter.
        """
        eye, target = self.get_camera_eye_target(camera_state)
        eye_h = np.array([eye[0], eye[1], eye[2], 1.0], dtype=np.float64)
        target_h = np.array([target[0], target[1], target[2], 1.0], dtype=np.float64)
        eye_usd = (self.EXPORT_TO_OV_MATRIX @ eye_h)[:3]
        target_usd = (self.EXPORT_TO_OV_MATRIX @ target_h)[:3]
        distance = float(np.linalg.norm(eye_usd - target_usd))

        return {
            "Perspective": {
                "position": tuple(float(v) for v in eye_usd),
                "target": tuple(float(v) for v in target_usd),
                "radius": distance,
                "fov": float(camera_fov),
            },
            "boundCamera": "/OmniverseKit_Persp",
        }

    def create_export_model_snapshot(self):
        """Clone the current gaussian model into an export-only snapshot.

        Returns:
            object: Lightweight model wrapper exposing the export interface
            expected by downstream export utilities.
        """
        source = self.engine.scene_mog
        n_active_features = int(source.get_n_active_features())

        class _ExportModelSnapshot:
            """Detached gaussian-model snapshot used only during export.

            Note:
                This lightweight wrapper exposes just the methods expected by
                downstream export helpers, while keeping all tensors detached
                from the live engine state.
            """

            def __init__(self):
                """Clone export-relevant tensors from the live gaussian model.

                Returns:
                    None: Detached export tensors are stored on the snapshot.
                """
                self.positions = torch.nn.Parameter(source.positions.detach().clone(), requires_grad=False)
                self.rotation = torch.nn.Parameter(source.rotation.detach().clone(), requires_grad=False)
                self.scale = torch.nn.Parameter(source.scale.detach().clone(), requires_grad=False)
                self.density = torch.nn.Parameter(source.density.detach().clone(), requires_grad=False)
                self.features_albedo = torch.nn.Parameter(source.features_albedo.detach().clone(), requires_grad=False)
                self.features_specular = torch.nn.Parameter(source.features_specular.detach().clone(), requires_grad=False)
                self._n_active_features = n_active_features

            def get_positions(self):
                """Return detached gaussian positions for export.

                Returns:
                    torch.nn.Parameter: Snapshot positions tensor.
                """
                return self.positions

            def get_rotation(self, preactivation=True):
                """Return detached gaussian rotations for export.

                Args:
                    preactivation: Unused compatibility flag kept for API parity.

                Returns:
                    torch.nn.Parameter: Snapshot rotation tensor.
                """
                return self.rotation

            def get_scale(self, preactivation=True):
                """Return detached gaussian scales for export.

                Args:
                    preactivation: Unused compatibility flag kept for API parity.

                Returns:
                    torch.nn.Parameter: Snapshot scale tensor.
                """
                return self.scale

            def get_density(self, preactivation=True):
                """Return detached gaussian densities for export.

                Args:
                    preactivation: Unused compatibility flag kept for API parity.

                Returns:
                    torch.nn.Parameter: Snapshot density tensor.
                """
                return self.density

            def get_features_albedo(self):
                """Return detached gaussian albedo features for export.

                Returns:
                    torch.nn.Parameter: Snapshot albedo feature tensor.
                """
                return self.features_albedo

            def get_features_specular(self):
                """Return detached gaussian specular features for export.

                Returns:
                    torch.nn.Parameter: Snapshot specular feature tensor.
                """
                return self.features_specular

            def get_n_active_features(self):
                """Return the number of active feature channels in the snapshot.

                Returns:
                    int: Active feature count copied from the live model.
                """
                return self._n_active_features

        return _ExportModelSnapshot()

    def patch_gauss_usda_orientation(self, gauss_usda_bytes):
        """Rewrite the exported gauss transform into the expected USD basis.

        Args:
            gauss_usda_bytes: Raw ``gauss.usda`` file contents.

        Returns:
            bytes: Patched file contents, or the original bytes if unchanged.
        """
        try:
            content = gauss_usda_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return gauss_usda_bytes

        pattern = r"^\s*matrix4d xformOp:transform\s*=\s*\(.*\)$"
        match = re.search(pattern, content, flags=re.MULTILINE)
        if match is None:
            return gauss_usda_bytes

        old_line = match.group(0)
        number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        values = [float(v) for v in re.findall(number_pattern, old_line)]
        if len(values) != 16:
            return gauss_usda_bytes

        tx, ty, tz = values[3], values[7], values[11]
        new_matrix = np.array(self.EXPORT_TO_OV_MATRIX, dtype=np.float64)
        new_matrix[0, 3] = tx
        new_matrix[1, 3] = ty
        new_matrix[2, 3] = tz

        row0 = f"{new_matrix[0,0]:.12g}, {new_matrix[0,1]:.12g}, {new_matrix[0,2]:.12g}, {new_matrix[0,3]:.12g}"
        row1 = f"{new_matrix[1,0]:.12g}, {new_matrix[1,1]:.12g}, {new_matrix[1,2]:.12g}, {new_matrix[1,3]:.12g}"
        row2 = f"{new_matrix[2,0]:.12g}, {new_matrix[2,1]:.12g}, {new_matrix[2,2]:.12g}, {new_matrix[2,3]:.12g}"
        row3 = f"{new_matrix[3,0]:.12g}, {new_matrix[3,1]:.12g}, {new_matrix[3,2]:.12g}, {new_matrix[3,3]:.12g}"
        new_line = f"        matrix4d xformOp:transform = ( ({row0}), ({row1}), ({row2}), ({row3}) )"

        content = content[:match.start()] + new_line + content[match.end():]
        return content.encode("utf-8")

    def patch_exported_usdz_orientation(self, usdz_path: Path):
        """Patch ``gauss.usda`` inside an exported USDZ archive when present.

        Args:
            usdz_path: Path to the USDZ archive to patch.

        Returns:
            None: The archive is rewritten in place only when a patch is needed.
        """
        if not usdz_path.exists():
            return

        with zipfile.ZipFile(usdz_path, "r") as zin:
            names = zin.namelist()
            files = {name: zin.read(name) for name in names}

        if "gauss.usda" not in files:
            return

        patched = self.patch_gauss_usda_orientation(files["gauss.usda"])
        if patched == files["gauss.usda"]:
            return
        files["gauss.usda"] = patched

        with zipfile.ZipFile(usdz_path, "w", compression=zipfile.ZIP_STORED) as zout:
            for name in names:
                zout.writestr(name, files[name])

    def export_usdz(self, gs_object, bbox_filter_enabled, bbox_min=None, bbox_max=None, default_config="apps/colmap_3dgrt.yaml", camera_state=None):
        """Export the current gaussian scene selection to USDZ and NUREC.

        Args:
            gs_object: Source gaussian asset path.
            bbox_filter_enabled: Whether export should be cropped to an active BBox.
            bbox_min: Optional minimum export BBox corner.
            bbox_max: Optional maximum export BBox corner.
            default_config: Config path used to initialize export settings.
            camera_state: Optional orbit camera state used to bind an export camera.

        Returns:
            tuple: ``(ok, output_usdz, status_message)`` describing export success.
        """
        from threedgrut.export.edit_nurec import crop_model_to_bbox, export_model_to_nurec
        from threedgrut.export.scripts.ply_to_usd import load_default_config
        from threedgrut.export.usdz_exporter import USDZExporter

        conf = load_default_config(config_name=default_config)
        method = self.infer_export_method(default_config)
        conf.render.method = method
        model_for_export = self.create_export_model_snapshot()

        if bbox_filter_enabled and bbox_min is not None and bbox_max is not None:
            kept, total = crop_model_to_bbox(model_for_export, bbox_min, bbox_max)
            if kept == 0:
                return False, None, "Export failed: empty bbox"

        src_path = Path(gs_object)
        output_dir = src_path.resolve().parent / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_usdz = output_dir / f"{src_path.stem}_view_{timestamp}.usdz"
        output_nurec = output_dir / f"{src_path.stem}_view_{timestamp}.nurec"

        export_model_to_nurec(model_for_export, output_nurec, conf, method)

        camera_settings = None
        if camera_state is not None:
            camera_settings = self.build_export_camera_settings(camera_state, self.engine.camera_fov)

        exporter = USDZExporter()
        exporter.export(
            model_for_export,
            output_usdz,
            dataset=None,
            conf=conf,
            camera_settings=camera_settings,
        )
        self.patch_exported_usdz_orientation(output_usdz)

        logger.info(f"USDZ export done: {output_usdz}")
        logger.info(f"NUREC export done: {output_nurec}")
        return True, output_usdz, f"USDZ exported: {output_usdz.name}"
