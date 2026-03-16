"""Scene bounds, home-camera state, and HUD projection helpers.

This module keeps scene-wide derived values out of the main viewer class,
including bounds estimation, default camera initialization, and the state
bundle consumed by HUD guide rendering.
"""

from __future__ import annotations

import numpy as np

from threedgrut.utils.logger import logger


class SceneStateController:
    """Track derived scene bounds and build view-related helper state.

    Args:
        viewer: High-level viewer instance that owns the engine and camera.
    """

    def __init__(self, viewer):
        """Initialize default scene-derived values.

        Args:
            viewer: High-level viewer instance that owns the engine and camera.
        """
        self.viewer = viewer
        self.scene_bbox_min = None
        self.scene_bbox_max = None
        self.scene_diagonal = 1.0
        self.scene_center = np.array([0.0, 0.0, 0.0])
        self.scene_bbox_min_fixed = np.array([-1.5, -1.5, -1.5])
        self.scene_bbox_max_fixed = np.array([1.5, 1.5, 1.5])

    def compute_scene_bounds(self):
        """Estimate scene bounds from gaussian positions.

        Returns:
            None: Derived bounds and related scene metrics are stored in place.

        Note:
            The fixed bounds currently mirror the original viewer behavior used
            for home-view framing and guide rendering.
        """
        try:
            positions = self.viewer.engine.scene_mog.positions.detach().cpu().numpy()
            bbox_min = positions.min(axis=0)
            bbox_max = positions.max(axis=0)
            bbox_size = bbox_max - bbox_min
            scene_diagonal = np.linalg.norm(bbox_size)

            self.scene_bbox_min = bbox_min
            self.scene_bbox_max = bbox_max
            self.scene_diagonal = scene_diagonal
            self.scene_center = np.array([0.0, 0.0, 0.0])
            self.scene_bbox_min_fixed = np.array([-1.5, -1.5, -1.5])
            self.scene_bbox_max_fixed = np.array([1.5, 1.5, 1.5])

            logger.info(f"Scene bounds (actual): min={bbox_min}, max={bbox_max}")
            logger.info(
                f"Scene bounds (Polyscope fixed): min={self.scene_bbox_min_fixed}, max={self.scene_bbox_max_fixed}"
            )
            logger.info(f"Scene diagonal: {scene_diagonal:.4f}")
        except Exception as error:
            logger.warning(f"Failed to compute scene bounds: {error}")
            self.scene_center = np.array([0.0, 0.0, 0.0])
            self.scene_bbox_min_fixed = np.array([-1.5, -1.5, -1.5])
            self.scene_bbox_max_fixed = np.array([1.5, 1.5, 1.5])
            self.scene_diagonal = 1.0

    def initialize_camera_from_bounds(self):
        """Reset the orbit camera to the scene home pose.

        Returns:
            None: The viewer camera controller is updated in place.
        """
        self.viewer.camera_controller.pan_x = 0.0
        self.viewer.camera_controller.pan_y = 0.0
        self.viewer.camera_controller.pan_z = 0.0

        bbox_size = 3.0
        fov_rad = np.deg2rad(self.viewer.engine.camera_fov)
        self.viewer.camera_controller.distance = bbox_size / (2.0 * np.tan(fov_rad / 2.0))
        self.viewer.camera_controller.theta = 0.0
        self.viewer.camera_controller.phi = np.pi / 4.0
        self.viewer.camera_controller.roll = 0.0
        self.viewer.camera_controller._update_camera_matrix()

    def build_hud_scene_state(self):
        """Assemble the scene-state bundle required by HUD guide rendering.

        Returns:
            dict: Scene center, extents, view matrix, and projection metadata.
        """
        return {
            "pan_x": self.viewer.camera_controller.pan_x,
            "pan_y": self.viewer.camera_controller.pan_y,
            "pan_z": self.viewer.camera_controller.pan_z,
            "view_matrix": self.viewer.camera_controller.get_view_matrix(),
            "fov_deg": self.viewer.engine.camera_fov,
            "base_extent": float(np.max(self.scene_bbox_max_fixed - self.scene_bbox_min_fixed))
            if self.scene_bbox_max_fixed is not None and self.scene_bbox_min_fixed is not None
            else 3.0,
        }
