"""BBox UI state and interactions for the GLFW viewer.

This module owns the text-editing state, panel hit testing, and the bridge
from UI actions to gaussian filtering and export operations.
"""

from __future__ import annotations

import numpy as np
import glfw


class BBoxUIController:
    """Manage BBox text fields, button state, and related actions.

    Args:
        viewer: High-level viewer instance that provides camera and export access.
    """

    def __init__(self, viewer):
        """Initialize default BBox UI state.

        Args:
            viewer: High-level viewer instance that provides camera and export access.
        """
        self.viewer = viewer
        self.filter_enabled = False
        self.last_kept_count = 0
        self.status_message = "BBox disabled"
        self.last_applied_min = None
        self.last_applied_max = None
        self.ui_rects = {}
        self.ui_canvas_size = (viewer.window_width, viewer.window_height)
        self.ui_drag_lock = False
        # Drag-value state: two rows (min/max), three axes each, no hard range
        # Initial values and center baselines are aligned with top text rows.
        self.slider_center_values = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
        self.slider_values = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
        self.slider_drag_sensitivity = 0.02  # value delta per dragged pixel
        self.active_slider = None        # (bound_type, axis_idx) or None
        self.slider_drag_start_x = None  # canvas-space X where drag started
        self.slider_drag_start_val = None  # slider value at drag start
        # Slider text-edit state (entered via double-click)
        self.slider_editing_field = None   # (bound_type, axis_idx) in text-edit mode
        self.slider_edit_text = ""         # current edit buffer
        self.slider_edit_selected = False  # True = whole text is selected (highlight)
        self._slider_last_click_time = 0.0
        self._slider_last_click_key = None

    def initialize_state(self, width: int, height: int):
        """Reset BBox UI state to its startup values.

        Args:
            width: Current canvas width in pixels.
            height: Current canvas height in pixels.

        Returns:
            None: Internal UI state is reset in place.
        """
        self.filter_enabled = False
        self.last_kept_count = 0
        self.status_message = "BBox disabled"
        self.last_applied_min = None
        self.last_applied_max = None
        self.ui_rects = {}
        self.ui_canvas_size = (width, height)
        self.ui_drag_lock = False
        self.slider_center_values = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
        self.slider_values = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}
        self.slider_drag_sensitivity = 0.02
        self.active_slider = None
        self.slider_drag_start_x = None
        self.slider_drag_start_val = None
        self.slider_editing_field = None
        self.slider_edit_text = ""
        self.slider_edit_selected = False
        self._slider_last_click_time = 0.0
        self._slider_last_click_key = None

    def parse_bbox_from_sliders(self):
        """Build BBox bounds from the bottom min/max drag rows.

        The slider rows are the sole source of truth for BBox filtering.  To
        keep interaction smooth during drag, each axis is auto-ordered so the
        returned ``min`` is always ``<= max``.

        Returns:
            tuple[np.ndarray, np.ndarray]: Ordered ``(min_vals, max_vals)``.
        """
        min_raw = np.array(self.slider_values["min"], dtype=np.float32)
        max_raw = np.array(self.slider_values["max"], dtype=np.float32)
        min_vals = np.minimum(min_raw, max_raw)
        max_vals = np.maximum(min_raw, max_raw)
        return min_vals, max_vals

    def refresh_bbox_filter_from_sliders(self):
        """Re-apply active BBox filter using current slider values.

        Returns:
            bool: ``True`` when no refresh was needed or apply succeeded.
        """
        if not self.filter_enabled:
            return True
        min_vals, max_vals = self.parse_bbox_from_sliders()
        return self.apply_bbox_filter(min_vals, max_vals)

    def apply_bbox_filter(self, min_vals, max_vals):
        """Apply BBox filtering to the gaussian model through the exporter.

        Args:
            min_vals: Minimum BBox corner in world coordinates.
            max_vals: Maximum BBox corner in world coordinates.

        Returns:
            bool: ``True`` when at least one gaussian remains after filtering.
        """
        ok, kept, total = self.viewer.exporter.apply_bbox_filter(min_vals, max_vals)
        if not ok:
            self.status_message = "No gaussians inside bbox"
            return False

        self.last_kept_count = kept
        self.last_applied_min = np.array(min_vals, dtype=np.float32)
        self.last_applied_max = np.array(max_vals, dtype=np.float32)
        self.status_message = f"BBox enabled: kept {kept}/{total}"
        self.filter_enabled = True
        return True

    def disable_bbox_filter(self):
        """Restore original gaussian density values and disable the BBox filter.

        Returns:
            None: Filter state is reset in place.
        """
        self.viewer.exporter.disable_bbox_filter()
        self.filter_enabled = False
        self.last_applied_min = None
        self.last_applied_max = None
        self.status_message = "BBox disabled"

    def toggle_bbox_filter_from_ui(self):
        """Toggle the filter using the bottom min/max drag rows.

        Returns:
            None: The filter state and status message are updated in place.
        """
        if self.filter_enabled:
            self.disable_bbox_filter()
            return

        min_vals, max_vals = self.parse_bbox_from_sliders()
        self.apply_bbox_filter(min_vals, max_vals)

    def export_current_gaussians_to_usdz(self):
        """Export the current gaussian selection to USDZ/NUREC.

        Returns:
            None: Export status is written back to ``status_message``.

        Note:
            When the BBox filter is active, the last successfully applied BBox
            is reused as the export crop region.
        """
        if self.filter_enabled and (self.last_applied_min is None or self.last_applied_max is None):
            self.status_message = "Export failed: no active bbox"
            return

        camera_state = {
            "theta": self.viewer.camera_controller.theta,
            "phi": self.viewer.camera_controller.phi,
            "distance": self.viewer.camera_controller.distance,
            "pan_x": self.viewer.camera_controller.pan_x,
            "pan_y": self.viewer.camera_controller.pan_y,
            "pan_z": self.viewer.camera_controller.pan_z,
        }
        ok, _, status = self.viewer.exporter.export_usdz(
            gs_object=self.viewer.gs_object,
            bbox_filter_enabled=self.filter_enabled,
            bbox_min=self.last_applied_min,
            bbox_max=self.last_applied_max,
            default_config=self.viewer.default_config,
            camera_state=camera_state,
        )
        self.status_message = status if ok else "USDZ export failed"

    @staticmethod
    def point_in_rect(x, y, rect):
        """Test whether a 2D point lies inside a UI rectangle.

        Args:
            x: Point x coordinate.
            y: Point y coordinate.
            rect: Rectangle stored as ``(x1, y1, x2, y2)``.

        Returns:
            bool: ``True`` when the point lies inside the rectangle bounds.
        """
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def handle_ui_click(self, x, y):
        """Dispatch a click to BBox buttons, fields, or the panel shell.

        Args:
            x: Click x coordinate in UI canvas space.
            y: Click y coordinate in UI canvas space.

        Returns:
            bool: ``True`` when the click is consumed by the BBox UI.
        """
        click_order = []
        if "bbox_export_button" in self.ui_rects:
            click_order.append("bbox_export_button")
        if "bbox_toggle_button" in self.ui_rects:
            click_order.append("bbox_toggle_button")
        click_order.extend(sorted(k for k in self.ui_rects.keys() if k.startswith("bbox_slider_")))
        if "bbox_panel" in self.ui_rects:
            click_order.append("bbox_panel")

        for key in click_order:
            rect = self.ui_rects[key]
            if not self.point_in_rect(x, y, rect):
                continue

            if key == "bbox_toggle_button":
                self.toggle_bbox_filter_from_ui()
                self.ui_drag_lock = True
                return True

            if key == "bbox_export_button":
                self.export_current_gaussians_to_usdz()
                self.ui_drag_lock = True
                return True

            if key.startswith("bbox_slider_"):
                # key format: bbox_slider_{min|max}_{0|1|2}
                import time
                parts = key.split("_")
                bound_type = parts[2]
                axis_idx = int(parts[3])
                now = time.monotonic()
                is_double_click = (
                    self._slider_last_click_key == key
                    and (now - self._slider_last_click_time) < 0.35
                )
                self._slider_last_click_time = now
                self._slider_last_click_key = key
                if is_double_click:
                    self._enter_slider_edit_mode(bound_type, axis_idx)
                    self.ui_drag_lock = True
                    return True
                # Single click → drag mode (cancel any ongoing text edit)
                self.slider_editing_field = None
                self.slider_edit_text = ""
                self.slider_edit_selected = False
                self.active_slider = (bound_type, axis_idx)
                self.slider_drag_start_x = x
                self.slider_drag_start_val = self.slider_values[bound_type][axis_idx]
                self.ui_drag_lock = True
                return True

            if key == "bbox_panel":
                self.ui_drag_lock = True
                return True

        return False

    def _enter_slider_edit_mode(self, bound_type: str, axis_idx: int):
        """Switch the specified slider into text-edit mode.

        The current numeric value is pre-filled into the edit buffer and the
        entire text is marked as selected so the next character typed will
        replace it immediately.

        Args:
            bound_type: ``"min"`` or ``"max"``.
            axis_idx: Axis index 0/1/2.

        Returns:
            None: Internal editing state is set in place.
        """
        self.slider_editing_field = (bound_type, axis_idx)
        self.slider_edit_text = f"{self.slider_values[bound_type][axis_idx]:.2f}"
        self.slider_edit_selected = True  # all text selected on entry
        self.active_slider = None

    def commit_slider_edit(self):
        """Parse the edit buffer and store the result, then exit edit mode.

        The parsed value is written directly without clamping.  When the
        buffer cannot be parsed the original value is preserved.

        Returns:
            None: ``slider_values`` and editing state are updated in place.
        """
        if self.slider_editing_field is None:
            return
        bound_type, axis_idx = self.slider_editing_field
        try:
            val = float(self.slider_edit_text)
            self.slider_values[bound_type][axis_idx] = val
        except (ValueError, TypeError):
            pass  # keep original value on bad input
        self.slider_editing_field = None
        self.slider_edit_text = ""
        self.slider_edit_selected = False
        self.refresh_bbox_filter_from_sliders()

    def cancel_slider_edit(self):
        """Discard the edit buffer and exit text-edit mode without saving.

        Returns:
            None: Editing state is cleared in place.
        """
        self.slider_editing_field = None
        self.slider_edit_text = ""
        self.slider_edit_selected = False

    def handle_slider_edit_key(self, key, action):
        """Handle key presses targeted at the active slider text-edit field.

        Args:
            key: GLFW key code.
            action: GLFW key action.

        Returns:
            bool: ``True`` when the event is consumed.
        """
        if self.slider_editing_field is None or action not in (glfw.PRESS, glfw.REPEAT):
            return False
        if key == glfw.KEY_BACKSPACE:
            if self.slider_edit_selected:
                self.slider_edit_text = ""
                self.slider_edit_selected = False
            else:
                self.slider_edit_text = self.slider_edit_text[:-1]
            return True
        if key == glfw.KEY_DELETE:
            self.slider_edit_text = ""
            self.slider_edit_selected = False
            return True
        if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self.commit_slider_edit()
            return True
        if key == glfw.KEY_ESCAPE:
            self.cancel_slider_edit()
            return True
        return False

    def handle_slider_edit_char(self, codepoint):
        """Append a character to the active slider edit buffer.

        When the text is in the initial *selected* state the buffer is cleared
        before appending so the first keystroke replaces the placeholder value.

        Args:
            codepoint: Unicode codepoint reported by GLFW.

        Returns:
            None: ``slider_edit_text`` is updated in place.
        """
        if self.slider_editing_field is None:
            return
        char = chr(codepoint)
        if char not in "0123456789.-+eE":
            return
        if self.slider_edit_selected:
            self.slider_edit_text = ""
            self.slider_edit_selected = False
        self.slider_edit_text += char

    def handle_slider_drag(self, canvas_x: float, slider_track_width: float = 200.0):
        """Update the active slider value based on horizontal mouse movement.

        Args:
            canvas_x: Current cursor x position in canvas space.
            slider_track_width: Retained for API compatibility; unused.

        Returns:
            None: ``slider_values`` is updated in place when a drag is active.
        """
        if self.active_slider is None or self.slider_drag_start_x is None:
            return
        bound_type, axis_idx = self.active_slider
        dx = canvas_x - self.slider_drag_start_x
        delta_val = dx * float(self.slider_drag_sensitivity)
        new_val = float(self.slider_drag_start_val + delta_val)
        self.slider_values[bound_type][axis_idx] = new_val
        self.refresh_bbox_filter_from_sliders()

    def release_slider(self):
        """Finish an active slider drag and clear drag state.

        Returns:
            None: Drag tracking fields are cleared.
        """
        self.active_slider = None
        self.slider_drag_start_x = None
        self.slider_drag_start_val = None

    def handle_active_field_key(self, key, action):
        """Handle key presses for active slider text-edit mode.

        Args:
            key: GLFW key code.
            action: GLFW key action.

        Returns:
            bool: ``True`` when the event is consumed by slider editing.
        """
        return self.handle_slider_edit_key(key, action)

    def handle_char_input(self, codepoint):
        """Append text input to active slider edit buffer only.

        Args:
            codepoint: Unicode codepoint reported by GLFW.

        Returns:
            None: Slider edit text is updated in place when valid.
        """
        self.handle_slider_edit_char(codepoint)
