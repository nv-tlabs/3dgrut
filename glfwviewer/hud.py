"""HUD, guide-overlay, and BBox UI drawing utilities.

This module centralizes all 2D overlay rendering performed on top of the
engine RGB output, including runtime text, scene guides, and the BBox
editing panel used for filtering and exporting visible gaussians.
"""

import time

import numpy as np
import cv2
import torch

from .utils import quat_wxyz_to_euler_deg, format_runtime_memory_usage

class HUD:
    """Compose viewer overlay text and draw 2D guide elements.

    Note:
        All drawing happens in image space using OpenCV so the render loop
        can remain agnostic to overlay semantics.
    """

    HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
    HUD_FONT_SCALE = 0.46
    HUD_TEXT_THICKNESS = 1
    HUD_LINE_HEIGHT = 18
    HUD_MARGIN = 10
    HUD_TEXT_BASELINE_OFFSET = 8
    HUD_BOX_X = 8
    HUD_BOX_Y = 8
    HUD_BACKGROUND_DARKEN = 0.35
    HUD_STROKE_COLOR = (0.0, 0.0, 0.0)
    HUD_TEXT_COLOR = (0.95, 0.95, 0.95)
    AXIS_X_COLOR = (0.2, 0.2, 1.0)
    AXIS_Y_COLOR = (0.2, 1.0, 0.2)
    AXIS_Z_COLOR = (1.0, 0.4, 0.2)
    CENTER_DOT_COLOR = (0.1, 0.1, 1.0)
    CENTER_DOT_RADIUS = 10
    ORBIT_SEGMENTS = 72
    ORBIT_LINE_THICKNESS = 2
    PLANE_ALPHA = 0.14
    CAMERA_NEAR = 0.01
    CAMERA_FAR = 100.0

    AXIS_INDICATOR_SIZE = 52
    AXIS_INDICATOR_MARGIN = 14
    AXIS_INDICATOR_ARROW_LEN = 40
    AXIS_INDICATOR_LINE_THICKNESS = 2
    AXIS_INDICATOR_TIP_RADIUS = 4
    AXIS_INDICATOR_FONT_SCALE = 0.48
    AXIS_INDICATOR_FONT_THICKNESS = 1

    def __init__(self):
        """Initialize cached object statistics used by the HUD.

        Returns:
            None: Cache fields and refresh timers are stored on the instance.
        """
        self.object_state_cache = {
            "center": np.zeros(3, dtype=np.float32),
            "extent": np.zeros(3, dtype=np.float32),
            "mean_scale": np.zeros(3, dtype=np.float32),
            "mean_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "mean_euler_deg": np.zeros(3, dtype=np.float32),
        }
        self.object_state_update_interval = 0.5
        self.object_state_last_update = 0.0
        self._orbit_t = np.linspace(0.0, 2.0 * np.pi, self.ORBIT_SEGMENTS + 1)
        self._orbit_cos = np.cos(self._orbit_t)
        self._orbit_sin = np.sin(self._orbit_t)

    def initialize_hud_state_cache(self, engine):
        """Reset and eagerly populate the cached object-state summary.

        Args:
            engine: Active viewer engine that owns the gaussian model.

        Returns:
            None: The cache is refreshed in place.
        """
        self.object_state_cache = {
            "center": np.zeros(3, dtype=np.float32),
            "extent": np.zeros(3, dtype=np.float32),
            "mean_scale": np.zeros(3, dtype=np.float32),
            "mean_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "mean_euler_deg": np.zeros(3, dtype=np.float32),
        }
        self.object_state_update_interval = 0.5
        self.object_state_last_update = 0.0
        self.update_object_state_cache(engine, force=True)

    def should_refresh_object_state_cache(self, now: float) -> bool:
        """Decide whether the cached object statistics should be recomputed.

        Args:
            now: Current wall-clock time in seconds.

        Returns:
            bool: ``True`` when the refresh interval has elapsed.
        """
        return (now - self.object_state_last_update) >= self.object_state_update_interval

    def collect_object_state_snapshot(self, engine):
        """Collect a fresh summary of scene-wide gaussian statistics.

        Args:
            engine: Active viewer engine that owns the gaussian model.

        Returns:
            dict: Cached values describing center, extent, scale, and mean orientation.
        """
        mog = engine.scene_mog
        positions = mog.positions

        center = positions.mean(dim=0)
        bbox_min = positions.min(dim=0).values
        bbox_max = positions.max(dim=0).values
        extent = bbox_max - bbox_min

        mean_scale = mog.get_scale().mean(dim=0)
        mean_quat = mog.get_rotation().mean(dim=0)
        mean_quat = mean_quat / torch.clamp(torch.linalg.norm(mean_quat), min=1e-8)
        mean_quat_np = mean_quat.detach().cpu().numpy().astype(np.float32)

        return {
            "center": center.detach().cpu().numpy().astype(np.float32),
            "extent": extent.detach().cpu().numpy().astype(np.float32),
            "mean_scale": mean_scale.detach().cpu().numpy().astype(np.float32),
            "mean_quat": mean_quat_np,
            "mean_euler_deg": quat_wxyz_to_euler_deg(mean_quat_np),
        }

    def update_object_state_cache(self, engine, force: bool = False):
        """Refresh cached object statistics when needed.

        Args:
            engine: Active viewer engine that owns the gaussian model.
            force: When ``True``, bypasses the time-based refresh guard.

        Returns:
            None: Cache contents are updated in place when a refresh occurs.
        """
        now = time.time()
        if not force and not self.should_refresh_object_state_cache(now):
            return

        self.object_state_cache = self.collect_object_state_snapshot(engine)
        self.object_state_last_update = now

    def get_mouse_action_label(self, interaction_state: dict) -> str:
        """Translate raw input flags into a human-readable interaction label.

        Args:
            interaction_state: Mouse and modifier key state snapshot.

        Returns:
            str: One of ``Idle``, ``Rotate``, ``Pan``, or ``Zoom``.
        """
        is_zoom = (
            interaction_state["ctrl_pressed"]
            and interaction_state["shift_pressed"]
            and interaction_state["mouse_left_pressed"]
        )
        is_pan = (
            interaction_state["shift_pressed"] and interaction_state["mouse_left_pressed"]
        ) or interaction_state["mouse_right_pressed"]
        is_rotate = (
            interaction_state["mouse_left_pressed"]
            and not interaction_state["shift_pressed"]
            and not interaction_state["ctrl_pressed"]
        )

        if is_zoom:
            return "Zoom"
        if is_pan:
            return "Pan"
        if is_rotate:
            return "Rotate"
        return "Idle"

    def build_hud_lines(self, engine, runtime_state: dict, interaction_state: dict, camera_state: dict, bbox_state: dict):
        """Build the text lines displayed in the top-left HUD block.

        Args:
            engine: Active viewer engine used for runtime statistics.
            runtime_state: Lightweight FPS and render-style snapshot.
            interaction_state: Mouse and keyboard interaction snapshot.
            camera_state: Orbit camera parameters needed for display.
            bbox_state: BBox filter status text and enable state.

        Returns:
            list[str]: Ordered HUD text lines ready for drawing.
        """
        self.update_object_state_cache(engine)
        state = self.object_state_cache

        center = state["center"]
        extent = state["extent"]
        mean_scale = state["mean_scale"]
        mean_quat = state["mean_quat"]
        mean_euler = state["mean_euler_deg"]

        mode = self.get_mouse_action_label(interaction_state)

        return [
            f"FPS: {runtime_state['fps']:5.1f} | Style: {runtime_state['render_style']}",
            format_runtime_memory_usage(engine, engine.device),
            f"BBox Filter: {'ON' if bbox_state['enabled'] else 'OFF'} | {bbox_state['status_message']}",
            "Mouse: L=Rotate | Shift+L / R=Pan | Ctrl+Shift+L / Wheel=Zoom",
            f"Mouse Pos: ({interaction_state['mouse_pos'][0]:.0f}, {interaction_state['mouse_pos'][1]:.0f}) | Mode: {mode}",
            (
                f"Buttons[L,R,M]=[{int(interaction_state['mouse_left_pressed'])},{int(interaction_state['mouse_right_pressed'])},"
                f"{int(interaction_state['mouse_middle_pressed'])}] | Mods[Shift,Ctrl]=[{int(interaction_state['shift_pressed'])},{int(interaction_state['ctrl_pressed'])}]"
            ),
            (
                f"Camera Orbit(deg): theta={np.degrees(camera_state['theta']):.1f}, "
                f"phi={np.degrees(camera_state['phi']):.1f}, roll={np.degrees(camera_state['roll']):.1f} "
                f"| dist={camera_state['distance']:.3f}"
            ),
            f"Object Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})",
            f"Object Extent: ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f})",
            f"Object Scale(mean xyz): ({mean_scale[0]:.4f}, {mean_scale[1]:.4f}, {mean_scale[2]:.4f})",
            (
                f"Object Rot quat(wxyz): ({mean_quat[0]:.4f}, {mean_quat[1]:.4f}, "
                f"{mean_quat[2]:.4f}, {mean_quat[3]:.4f})"
            ),
            (
                f"Object Orient rpy(deg): ({mean_euler[0]:.1f}, "
                f"{mean_euler[1]:.1f}, {mean_euler[2]:.1f})"
            ),
        ]

    def draw_hud_overlay(self, rgb, lines, bbox_ui_state: dict | None = None, scene_state: dict | None = None, view_matrix: np.ndarray | None = None):
        """Draw runtime HUD, guides, axis indicator, and BBox UI on a rendered RGB image.

        Args:
            rgb: Float RGB image in viewer render orientation.
            lines: HUD text lines generated by ``build_hud_lines``.
            bbox_ui_state: Optional mutable BBox UI state for drawing widgets.
            scene_state: Optional scene projection state for guide rendering.
                When ``None``, the orbit-plane guide overlay is skipped.
            view_matrix: Optional 4×4 world-to-view matrix used to draw the
                bottom-left axis orientation indicator.  When ``None`` the
                indicator is skipped.

        Returns:
            tuple: A pair of ``(rgb_with_overlay, bbox_ui_state)``.
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb, bbox_ui_state
        hud = np.ascontiguousarray(np.flipud(rgb))
        if scene_state is not None:
            self._draw_scene_guides(hud, scene_state)
        if view_matrix is not None:
            self._draw_axis_indicator(hud, view_matrix)
        box_x, box_y, box_w, box_h = self._compute_hud_box_geometry(hud, lines)
        if box_w <= 2 or box_h <= 2:
            return hud, bbox_ui_state
        self._apply_hud_background(hud, box_x, box_y, box_w, box_h)
        self._draw_hud_lines(hud, lines, box_x, box_y, box_h)
        if bbox_ui_state is not None:
            self._draw_bbox_ui_overlay(hud, bbox_ui_state)
        return np.ascontiguousarray(np.flipud(hud)), bbox_ui_state

    def _build_projection_matrix(self, width: int, height: int, fov_deg: float) -> np.ndarray:
        """Build a perspective projection matrix for HUD world-to-screen projection.

        Args:
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            fov_deg: Vertical field of view in degrees.

        Returns:
            np.ndarray: A ``4 x 4`` projection matrix.
        """
        aspect = max(width / max(height, 1), 1e-6)
        fov_rad = np.deg2rad(fov_deg)
        f = 1.0 / np.tan(fov_rad / 2.0)
        near = self.CAMERA_NEAR
        far = self.CAMERA_FAR

        proj = np.zeros((4, 4), dtype=np.float64)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    def _project_world_to_screen(self, point_world: np.ndarray, width: int, height: int, view_matrix: np.ndarray, fov_deg: float):
        """Project a world-space point onto the HUD canvas.

        Args:
            point_world: World-space position to project.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            view_matrix: World-to-view matrix used by the camera.
            fov_deg: Vertical field of view in degrees.

        Returns:
            tuple[int, int] | None: Screen-space pixel coordinates, or ``None`` if clipped.
        """
        proj = self._build_projection_matrix(width, height, fov_deg)
        p = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
        clip = proj @ (view_matrix @ p)
        w = clip[3]
        if abs(w) < 1e-8:
            return None

        ndc = clip[:3] / w
        if ndc[2] < -1.2 or ndc[2] > 1.2:
            return None

        sx = int((ndc[0] * 0.5 + 0.5) * width)
        sy = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
        return sx, sy

    def _project_world_to_screen_vp(self, point_world: np.ndarray, width: int, height: int, view_proj: np.ndarray):
        """Project a world-space point using a precomputed view-projection matrix.

        Args:
            point_world: World-space position to project.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            view_proj: Combined projection-view matrix.

        Returns:
            tuple[int, int] | None: Screen-space pixel coordinates, or ``None`` if clipped.
        """
        p = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=np.float64)
        clip = view_proj @ p
        w = clip[3]
        if abs(w) < 1e-8:
            return None

        ndc = clip[:3] / w
        if ndc[2] < -1.2 or ndc[2] > 1.2:
            return None

        sx = int((ndc[0] * 0.5 + 0.5) * width)
        sy = int((1.0 - (ndc[1] * 0.5 + 0.5)) * height)
        return sx, sy

    def _draw_projected_polyline(self, hud: np.ndarray, points_3d: np.ndarray, color, thickness: int, scene_state: dict, view_proj: np.ndarray | None = None):
        """Project and draw a 3D polyline segment-by-segment on the HUD.

        Args:
            hud: Target RGB canvas in OpenCV format.
            points_3d: Polyline points expressed in world coordinates.
            color: OpenCV RGB color tuple.
            thickness: Line thickness in pixels.
            scene_state: Projection-related scene information.

        Returns:
            None: Drawing occurs directly on ``hud``.
        """
        h, w = hud.shape[:2]
        if view_proj is None:
            proj = self._build_projection_matrix(w, h, scene_state["fov_deg"])
            view_proj = proj @ scene_state["view_matrix"]
        segment = []
        for point in points_3d:
            uv = self._project_world_to_screen_vp(point, w, h, view_proj)
            if uv is None:
                if len(segment) >= 2:
                    cv2.polylines(hud, [np.array(segment, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
                segment = []
                continue
            segment.append(uv)

        if len(segment) >= 2:
            cv2.polylines(hud, [np.array(segment, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def _draw_projected_plane(self, hud: np.ndarray, corners_3d: np.ndarray, color, alpha: float, scene_state: dict, view_proj: np.ndarray | None = None):
        """Project and draw a filled 3D plane patch on the HUD canvas.

        Args:
            hud: Target RGB canvas in OpenCV format.
            corners_3d: Plane corner points in world coordinates.
            color: OpenCV RGB color tuple.
            alpha: Fill alpha used for compositing.
            scene_state: Projection-related scene information.

        Returns:
            None: Drawing occurs directly on ``hud``.
        """
        h, w = hud.shape[:2]
        if view_proj is None:
            proj = self._build_projection_matrix(w, h, scene_state["fov_deg"])
            view_proj = proj @ scene_state["view_matrix"]
        corners_2d = []
        for point in corners_3d:
            uv = self._project_world_to_screen_vp(point, w, h, view_proj)
            if uv is None:
                return
            corners_2d.append(uv)

        polygon = np.array(corners_2d, dtype=np.int32)
        min_x = max(0, int(np.min(polygon[:, 0])) - 1)
        max_x = min(w, int(np.max(polygon[:, 0])) + 2)
        min_y = max(0, int(np.min(polygon[:, 1])) - 1)
        max_y = min(h, int(np.max(polygon[:, 1])) + 2)
        if max_x <= min_x or max_y <= min_y:
            return
        roi = hud[min_y:max_y, min_x:max_x]
        roi_overlay = roi.copy()
        shifted_polygon = polygon - np.array([min_x, min_y], dtype=np.int32)
        cv2.fillConvexPoly(roi_overlay, shifted_polygon, color, lineType=cv2.LINE_AA)
        cv2.addWeighted(roi_overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
        cv2.polylines(hud, [polygon], True, color, 1, cv2.LINE_AA)

    def _draw_scene_guides(self, hud: np.ndarray, scene_state: dict):
        """Draw orbit planes, axis rings, and the scene center marker.

        Args:
            hud: Target RGB canvas in OpenCV format.
            scene_state: Projection and scene extents used for guide generation.

        Returns:
            None: Drawing occurs directly on ``hud``.
        """
        center = np.array([
            scene_state["pan_x"],
            scene_state["pan_y"],
            scene_state["pan_z"],
        ], dtype=np.float64)
        h, w = hud.shape[:2]

        base_extent = float(scene_state.get("base_extent", 3.0))
        orbit_radius = max(0.3, 0.45 * base_extent)
        plane_half = orbit_radius * 1.05

        xy_corners = np.array([
            center + np.array([-plane_half, -plane_half, 0.0]),
            center + np.array([plane_half, -plane_half, 0.0]),
            center + np.array([plane_half, plane_half, 0.0]),
            center + np.array([-plane_half, plane_half, 0.0]),
        ], dtype=np.float64)
        yz_corners = np.array([
            center + np.array([0.0, -plane_half, -plane_half]),
            center + np.array([0.0, -plane_half, plane_half]),
            center + np.array([0.0, plane_half, plane_half]),
            center + np.array([0.0, plane_half, -plane_half]),
        ], dtype=np.float64)
        xz_corners = np.array([
            center + np.array([-plane_half, 0.0, -plane_half]),
            center + np.array([plane_half, 0.0, -plane_half]),
            center + np.array([plane_half, 0.0, plane_half]),
            center + np.array([-plane_half, 0.0, plane_half]),
        ], dtype=np.float64)

        proj = self._build_projection_matrix(w, h, scene_state["fov_deg"])
        view_proj = proj @ scene_state["view_matrix"]

        self._draw_projected_plane(hud, xy_corners, self.AXIS_Z_COLOR, self.PLANE_ALPHA, scene_state, view_proj=view_proj)
        self._draw_projected_plane(hud, yz_corners, self.AXIS_X_COLOR, self.PLANE_ALPHA, scene_state, view_proj=view_proj)
        self._draw_projected_plane(hud, xz_corners, self.AXIS_Y_COLOR, self.PLANE_ALPHA, scene_state, view_proj=view_proj)

        t_cos = self._orbit_cos
        t_sin = self._orbit_sin
        orbit_x = np.stack([
            np.full_like(t_cos, center[0]),
            center[1] + orbit_radius * t_cos,
            center[2] + orbit_radius * t_sin,
        ], axis=1)
        orbit_y = np.stack([
            center[0] + orbit_radius * t_cos,
            np.full_like(t_cos, center[1]),
            center[2] + orbit_radius * t_sin,
        ], axis=1)
        orbit_z = np.stack([
            center[0] + orbit_radius * t_cos,
            center[1] + orbit_radius * t_sin,
            np.full_like(t_cos, center[2]),
        ], axis=1)

        self._draw_projected_polyline(hud, orbit_x, self.AXIS_X_COLOR, self.ORBIT_LINE_THICKNESS, scene_state, view_proj=view_proj)
        self._draw_projected_polyline(hud, orbit_y, self.AXIS_Y_COLOR, self.ORBIT_LINE_THICKNESS, scene_state, view_proj=view_proj)
        self._draw_projected_polyline(hud, orbit_z, self.AXIS_Z_COLOR, self.ORBIT_LINE_THICKNESS, scene_state, view_proj=view_proj)

        center_uv = self._project_world_to_screen_vp(center, w, h, view_proj)
        if center_uv is not None:
            cv2.circle(hud, center_uv, self.CENTER_DOT_RADIUS, self.CENTER_DOT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(hud, center_uv, self.CENTER_DOT_RADIUS + 2, (1.0, 1.0, 1.0), 1, cv2.LINE_AA)

    def _draw_bbox_ui_overlay(self, hud: np.ndarray, state: dict):
        """Draw the BBox input panel, action buttons, and slider rows.

        Args:
            hud: Target RGB canvas in OpenCV format.
            state: Mutable BBox UI state used for widget geometry and display.

        Returns:
            None: Drawing occurs directly on ``hud`` and ``state`` is updated.
        """
        h, w = hud.shape[:2]
        state["canvas_size"] = (w, h)
        panel_w = min(420, max(300, w - 24))
        panel_h = 250
        panel_x = max(8, w - panel_w - 8)
        panel_y = 8

        panel_color = (0.08, 0.08, 0.08)
        panel_alpha = 0.82
        roi = hud[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        roi_overlay = roi.copy()
        cv2.rectangle(roi_overlay, (0, 0), (panel_w, panel_h), panel_color, -1)
        cv2.addWeighted(roi_overlay, panel_alpha, roi, 1.0 - panel_alpha, 0.0, dst=roi)
        cv2.rectangle(hud, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0.5, 0.5, 0.5), 1)

        state["rects"] = {
            "bbox_panel": (panel_x, panel_y, panel_x + panel_w, panel_y + panel_h)
        }

        cv2.putText(hud, "BBox Controls", (panel_x + 12, panel_y + 24), self.HUD_FONT, 0.62, (0.95, 0.95, 0.95), 1, cv2.LINE_AA)

        status_color = (0.5, 1.0, 0.5) if state["filter_enabled"] else (1.0, 0.75, 0.3)
        cv2.putText(
            hud,
            state["status_message"][:48],
            (panel_x + 12, panel_y + 45),
            self.HUD_FONT,
            0.45,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # ---- Slider rows (only active bbox controls) ----
        axis_names = ["x", "y", "z"]
        slider_row_top = panel_y + 64
        slider_h = 28
        slider_gap = 10
        slider_label_w = 56   # width reserved for "min"/ "max" label
        slider_track_x = panel_x + slider_label_w + 6
        slider_track_w = panel_w - slider_label_w - 20

        slider_state = state.get("slider_values", {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]})
        slider_center_state = state.get("slider_center_values", {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]})
        active_slider = state.get("active_slider", None)

        for row_idx, bound_type in enumerate(("min", "max")):
            row_y = slider_row_top + row_idx * (slider_h + slider_gap)
            cv2.putText(
                hud, bound_type,
                (panel_x + 8, row_y + 19),
                self.HUD_FONT, 0.54, (0.9, 0.9, 0.9), 1, cv2.LINE_AA,
            )
            # Axis header on first row only
            if row_idx == 0:
                per_w = slider_track_w // 3
                for ai, an in enumerate(axis_names):
                    ax = slider_track_x + ai * per_w + per_w // 2 - 4
                    cv2.putText(hud, an, (ax, row_y - 2), self.HUD_FONT, 0.40, (0.8, 0.8, 1.0), 1, cv2.LINE_AA)

            per_w = slider_track_w // 3
            for axis_idx in range(3):
                sx = slider_track_x + axis_idx * per_w
                s_w = per_w - 6
                slider_rect = (sx, row_y, sx + s_w, row_y + slider_h)
                val = slider_state[bound_type][axis_idx]
                center_val = slider_center_state[bound_type][axis_idx]
                is_active = (active_slider == (bound_type, axis_idx))
                self._draw_bbox_slider(
                    hud, slider_rect,
                    bound_type, axis_idx,
                    val, center_val, is_active, state,
                )

        # ---- Buttons ----
        button_w = panel_w - 24
        button_h = 34
        button_x = panel_x + 12
        button_y = panel_y + panel_h - button_h * 2 - 18
        button_rect = (button_x, button_y, button_x + button_w, button_y + button_h)
        state["rects"]["bbox_toggle_button"] = button_rect

        button_color = (0.16, 0.6, 0.2) if not state["filter_enabled"] else (0.52, 0.24, 0.16)
        cv2.rectangle(hud, (button_rect[0], button_rect[1]), (button_rect[2], button_rect[3]), button_color, -1)
        cv2.rectangle(hud, (button_rect[0], button_rect[1]), (button_rect[2], button_rect[3]), (0.88, 0.88, 0.88), 1)

        button_text = "Show BBox Only" if not state["filter_enabled"] else "Disable BBox Filter"
        cv2.putText(
            hud,
            button_text,
            (button_x + max(12, button_w // 2 - 92), button_y + 23),
            self.HUD_FONT,
            0.6,
            (1.0, 1.0, 1.0),
            1,
            cv2.LINE_AA,
        )

        export_button_y = button_y + button_h + 8
        export_button_rect = (button_x, export_button_y, button_x + button_w, export_button_y + button_h)
        state["rects"]["bbox_export_button"] = export_button_rect
        cv2.rectangle(hud, (export_button_rect[0], export_button_rect[1]), (export_button_rect[2], export_button_rect[3]), (0.16, 0.36, 0.72), -1)
        cv2.rectangle(hud, (export_button_rect[0], export_button_rect[1]), (export_button_rect[2], export_button_rect[3]), (0.88, 0.88, 0.88), 1)
        cv2.putText(
            hud,
            "Export USDZ",
            (button_x + max(12, button_w // 2 - 60), export_button_y + 23),
            self.HUD_FONT,
            0.6,
            (1.0, 1.0, 1.0),
            1,
            cv2.LINE_AA,
        )

    def _draw_bbox_slider(
        self,
        hud: np.ndarray,
        rect: tuple,
        bound_type: str,
        axis_idx: int,
        value: float,
        center_value: float,
        is_active: bool,
        state: dict,
    ):
        """Draw a single horizontal drag-slider for a BBox axis bound.

        The track fills the full ``rect`` width.  A filled thumb rectangle
        marks the current position.  The numeric value is rendered on top.

        When ``state["slider_editing_field"]`` matches this slider the widget
        switches to text-input appearance: a grey input background, the edit
        buffer text with a blinking ``|`` cursor, and an optional selection
        highlight (full-width blue overlay) while ``slider_edit_selected`` is
        ``True``.

        Args:
            hud: Target RGB canvas in OpenCV format.
            rect: Slider bounding box ``(x1, y1, x2, y2)``.
            bound_type: ``"min"`` or ``"max"``.
            axis_idx: Axis index 0/1/2 corresponding to x/y/z.
            value: Current slider value.
            center_value: Per-item center baseline used for display.
            is_active: ``True`` when this slider is being dragged.
            state: Mutable panel state; slider rect is registered here.

        Returns:
            None: Drawing occurs directly on ``hud`` and ``state`` is updated.
        """
        import time

        key = f"bbox_slider_{bound_type}_{axis_idx}"
        state["rects"][key] = rect

        x1, y1, x2, y2 = rect
        track_w = max(x2 - x1, 1)
        track_h = y2 - y1

        editing_field = state.get("slider_editing_field", None)
        is_editing = editing_field == (bound_type, axis_idx)

        if is_editing:
            # ---- Text-edit appearance ----
            edit_text = state.get("slider_edit_text", "")
            is_selected = state.get("slider_edit_selected", False)

            # Background: dark input field
            cv2.rectangle(hud, (x1, y1), (x2, y2), (0.12, 0.12, 0.18), -1)

            # Selection highlight (full-width blue overlay when text is selected)
            if is_selected:
                cv2.rectangle(hud, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0.18, 0.36, 0.72), -1)

            # Border: bright cyan to indicate active edit
            cv2.rectangle(hud, (x1, y1), (x2, y2), (0.25, 0.90, 0.95), 1)

            # Cursor blink at ~1 Hz
            cursor_visible = (int(time.monotonic() * 2) % 2) == 0
            display_text = edit_text + ("|" if cursor_visible else " ")

            label_col = (0.6, 0.75, 1.0) if is_selected else (0.95, 0.95, 0.95)
            (tw, th), _ = cv2.getTextSize(display_text, self.HUD_FONT, 0.42, 1)
            tx = x1 + (track_w - tw) // 2
            ty = y1 + (track_h + th) // 2 - 1
            cv2.putText(hud, display_text, (tx, ty), self.HUD_FONT, 0.42, (0.0, 0.0, 0.0), 2, cv2.LINE_AA)
            cv2.putText(hud, display_text, (tx, ty), self.HUD_FONT, 0.42, label_col, 1, cv2.LINE_AA)
            return

        # ---- Normal drag-value appearance (unbounded value) ----
        # Track background
        track_bg = (0.15, 0.15, 0.15)
        cv2.rectangle(hud, (x1, y1), (x2, y2), track_bg, -1)

        # Center guideline + relative visual displacement.
        center_x = x1 + track_w // 2
        cv2.line(hud, (center_x, y1 + 2), (center_x, y2 - 2), (0.35, 0.35, 0.35), 1)
        # Use tanh compression for display only so values remain unbounded.
        disp_ratio = float(np.tanh((value - center_value) * 0.12))
        disp_px = int(disp_ratio * max(track_w // 2 - 6, 1))
        fill_x = int(np.clip(center_x + disp_px, x1, x2))
        fill_color = (0.25, 0.55, 0.85) if not is_active else (0.35, 0.80, 1.0)
        if fill_x >= center_x:
            cv2.rectangle(hud, (center_x, y1 + 2), (fill_x, y2 - 2), fill_color, -1)
        else:
            cv2.rectangle(hud, (fill_x, y1 + 2), (center_x, y2 - 2), fill_color, -1)

        # Thumb marker
        thumb_half = 3
        cv2.rectangle(
            hud,
            (max(x1, fill_x - thumb_half), y1),
            (min(x2, fill_x + thumb_half), y2),
            (1.0, 1.0, 1.0) if is_active else (0.75, 0.75, 0.75),
            -1,
        )

        # Border
        border_col = (0.25, 0.85, 0.95) if is_active else (0.55, 0.55, 0.55)
        cv2.rectangle(hud, (x1, y1), (x2, y2), border_col, 1)

        # Value label centred on track
        label = f"{value:.2f}"
        (tw, th), _ = cv2.getTextSize(label, self.HUD_FONT, 0.42, 1)
        tx = x1 + (track_w - tw) // 2
        ty = y1 + (track_h + th) // 2 - 1
        cv2.putText(hud, label, (tx, ty), self.HUD_FONT, 0.42, (0.0, 0.0, 0.0), 2, cv2.LINE_AA)
        cv2.putText(hud, label, (tx, ty), self.HUD_FONT, 0.42, (0.95, 0.95, 0.95), 1, cv2.LINE_AA)

    def _compute_hud_box_geometry(self, hud, lines):
        """Compute the bounding box for the text HUD block.

        Args:
            hud: Target RGB canvas in OpenCV format.
            lines: HUD text lines to be rendered.

        Returns:
            tuple: ``(box_x, box_y, box_w, box_h)`` in pixels.
        """
        text_sizes = [
            cv2.getTextSize(line, self.HUD_FONT, self.HUD_FONT_SCALE, self.HUD_TEXT_THICKNESS)[0]
            for line in lines
        ]
        max_text_width = max((w for w, _ in text_sizes), default=0)
        box_x = self.HUD_BOX_X
        box_y = self.HUD_BOX_Y
        box_w = min(max_text_width + self.HUD_MARGIN * 2, hud.shape[1] - box_x - 2)
        box_h = min(len(lines) * self.HUD_LINE_HEIGHT + self.HUD_MARGIN * 2, hud.shape[0] - box_y - 2)
        return box_x, box_y, box_w, box_h

    def _apply_hud_background(self, hud, box_x, box_y, box_w, box_h):
        """Darken the background under the text HUD block.

        Args:
            hud: Target RGB canvas in OpenCV format.
            box_x: Left edge of the HUD block.
            box_y: Top edge of the HUD block.
            box_w: Width of the HUD block.
            box_h: Height of the HUD block.

        Returns:
            None: Pixel values are modified in place.
        """
        roi = hud[box_y:box_y + box_h, box_x:box_x + box_w]
        roi *= self.HUD_BACKGROUND_DARKEN

    def _draw_hud_lines(self, hud, lines, box_x, box_y, box_h):
        """Render the formatted HUD text with stroke and fill colors.

        Args:
            hud: Target RGB canvas in OpenCV format.
            lines: HUD text lines to render.
            box_x: Left edge of the HUD block.
            box_y: Top edge of the HUD block.
            box_h: Height of the HUD block.

        Returns:
            None: Drawing occurs directly on ``hud``.
        """
        text_x = box_x + self.HUD_MARGIN
        for idx, line in enumerate(lines):
            text_y = (
                box_y
                + self.HUD_MARGIN
                + (idx + 1) * self.HUD_LINE_HEIGHT
                - self.HUD_TEXT_BASELINE_OFFSET
            )
            if text_y >= box_y + box_h - 2:
                break
            cv2.putText(
                hud,
                line,
                (text_x, text_y),
                self.HUD_FONT,
                self.HUD_FONT_SCALE,
                self.HUD_STROKE_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                line,
                (text_x, text_y),
                self.HUD_FONT,
                self.HUD_FONT_SCALE,
                self.HUD_TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

    def _draw_axis_indicator(self, hud: np.ndarray, view_matrix: np.ndarray):
        """Draw a world-axis orientation indicator in the bottom-left corner.

        Projects the three world-space unit axes (X/Y/Z) through the current
        camera view rotation and renders them as red/green/blue arrows with
        letter labels at a fixed 2-D position.  A semi-transparent circular
        background is composited first so the indicator remains legible over
        any scene content.

        Args:
            hud: Target RGB canvas in OpenCV format (float32, values in [0,1]).
            view_matrix: 4×4 world-to-view matrix from the active camera.

        Returns:
            None: Drawing occurs directly on ``hud``.
        """
        h, w = hud.shape[:2]
        size = self.AXIS_INDICATOR_ARROW_LEN
        margin = self.AXIS_INDICATOR_MARGIN
        # Centre of the indicator widget, anchored to the bottom-left corner.
        cx = margin + size + 4
        cy = h - margin - size - 4

        # Extract the 3×3 rotation sub-matrix (upper-left of the 4×4 view).
        rot = view_matrix[:3, :3]  # shape (3, 3), maps world→camera

        # World-space axis directions.
        axes = {
            "X": (np.array([1.0, 0.0, 0.0]), (0.25, 0.25, 1.0)),   # red   (BGR→RGB in CV)
            "Y": (np.array([0.0, 1.0, 0.0]), (0.25, 1.0, 0.25)),   # green
            "Z": (np.array([0.0, 0.0, 1.0]), (1.0, 0.45, 0.25)),   # blue
        }

        # Draw a semi-transparent circular background.
        bg_radius = size + 18
        x0 = max(0, cx - bg_radius - 2)
        y0 = max(0, cy - bg_radius - 2)
        x1 = min(w, cx + bg_radius + 3)
        y1 = min(h, cy + bg_radius + 3)
        if x1 > x0 and y1 > y0:
            roi = hud[y0:y1, x0:x1]
            roi_overlay = roi.copy()
            cv2.circle(roi_overlay, (cx - x0, cy - y0), bg_radius, (0.08, 0.08, 0.08), -1, cv2.LINE_AA)
            cv2.addWeighted(roi_overlay, 0.55, roi, 0.45, 0.0, dst=roi)
        cv2.circle(hud, (cx, cy), bg_radius, (0.4, 0.4, 0.4), 1, cv2.LINE_AA)

        # Sort axes by camera-space depth (draw back-to-front for correct overlap).
        def _axis_depth(item):
            direction, _ = item[1]
            cam_dir = rot @ direction
            return -cam_dir[2]  # negative Z = further in front

        sorted_axes = sorted(axes.items(), key=_axis_depth, reverse=True)

        tip_radius = self.AXIS_INDICATOR_TIP_RADIUS
        for label, (world_dir, color) in sorted_axes:
            # Project world direction into camera space; use only XY for 2-D display.
            cam_dir = rot @ world_dir
            # Flip Y so +Y points upward on screen.
            screen_dir = np.array([cam_dir[0], -cam_dir[1]], dtype=np.float64)
            norm = np.linalg.norm(screen_dir)
            if norm < 1e-6:
                continue
            screen_dir /= norm

            tip_x = int(cx + screen_dir[0] * size)
            tip_y = int(cy + screen_dir[1] * size)

            cv2.line(hud, (cx, cy), (tip_x, tip_y), color,
                     self.AXIS_INDICATOR_LINE_THICKNESS, cv2.LINE_AA)
            cv2.circle(hud, (tip_x, tip_y), tip_radius, color, -1, cv2.LINE_AA)

            # Label positioned slightly beyond the arrow tip.
            label_offset = tip_radius + 7
            lx = int(cx + screen_dir[0] * (size + label_offset))
            ly = int(cy + screen_dir[1] * (size + label_offset))
            lx = max(4, min(w - 12, lx))
            ly = max(12, min(h - 4, ly))

            cv2.putText(hud, label, (lx - 5, ly + 5),
                        self.HUD_FONT, self.AXIS_INDICATOR_FONT_SCALE,
                        (0.0, 0.0, 0.0), 2, cv2.LINE_AA)
            cv2.putText(hud, label, (lx - 5, ly + 5),
                        self.HUD_FONT, self.AXIS_INDICATOR_FONT_SCALE,
                        color, self.AXIS_INDICATOR_FONT_THICKNESS, cv2.LINE_AA)

