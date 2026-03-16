"""Mouse and keyboard interaction helpers for the GLFW viewer.

This module isolates callback logic from the high-level viewer so input
behavior can stay close to the original monolithic script while the main
viewer class remains mostly orchestration-focused.
"""

from __future__ import annotations

import glfw
import numpy as np

from threedgrut.utils.logger import logger


class InteractionController:
    """Handle GLFW keyboard, mouse, and scroll callbacks.

    Args:
        viewer: High-level viewer instance whose state is mutated by input.
        bbox_ui: BBox UI controller that receives UI-specific interactions.
    """

    def __init__(self, viewer, bbox_ui):
        """Store references required by the interaction callbacks.

        Args:
            viewer: High-level viewer instance whose state is mutated by input.
            bbox_ui: BBox UI controller that receives UI-specific interactions.
        """
        self.viewer = viewer
        self.bbox_ui = bbox_ui

    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard shortcuts and modifier-state updates.

        Args:
            window: GLFW window handle that received the event.
            key: GLFW key code.
            scancode: Platform-dependent scan code.
            action: GLFW key action.
            mods: Bitmask of active modifier keys.

        Returns:
            None: Viewer state is updated in place.
        """
        self.viewer.shift_pressed = (mods & glfw.MOD_SHIFT) != 0
        self.viewer.ctrl_pressed = (mods & glfw.MOD_CONTROL) != 0

        if self.bbox_ui.handle_active_field_key(key, action):
            return

        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_S:
                self.viewer.render_style_index = (self.viewer.render_style_index + 1) % len(self.viewer.render_styles)
                logger.info(f"Switch to render style: {self.viewer.render_styles[self.viewer.render_style_index]}")
            elif key == glfw.KEY_H:
                self.viewer.show_hud_overlay = not self.viewer.show_hud_overlay
                state = "ON" if self.viewer.show_hud_overlay else "OFF"
                logger.info(f"🧩 HUD overlay: {state}")
            elif key == glfw.KEY_C and self.viewer.ctrl_pressed:
                self.viewer.camera_pose_clipboard = {
                    "theta": self.viewer.camera_controller.theta,
                    "phi": self.viewer.camera_controller.phi,
                    "roll": self.viewer.camera_controller.roll,
                    "distance": self.viewer.camera_controller.distance,
                    "pan_x": self.viewer.camera_controller.pan_x,
                    "pan_y": self.viewer.camera_controller.pan_y,
                    "pan_z": self.viewer.camera_controller.pan_z,
                }
                logger.info("📋 Camera pose copied to clipboard")
            elif key == glfw.KEY_V and self.viewer.ctrl_pressed:
                if self.viewer.camera_pose_clipboard is not None:
                    self.viewer.camera_controller.theta = self.viewer.camera_pose_clipboard["theta"]
                    self.viewer.camera_controller.phi = self.viewer.camera_pose_clipboard["phi"]
                    self.viewer.camera_controller.roll = self.viewer.camera_pose_clipboard.get("roll", 0.0)
                    self.viewer.camera_controller.distance = self.viewer.camera_pose_clipboard["distance"]
                    self.viewer.camera_controller.pan_x = self.viewer.camera_pose_clipboard["pan_x"]
                    self.viewer.camera_controller.pan_y = self.viewer.camera_pose_clipboard["pan_y"]
                    self.viewer.camera_controller.pan_z = self.viewer.camera_pose_clipboard["pan_z"]
                    self.viewer.camera_controller._update_camera_matrix()
                    logger.info("📋 Camera pose pasted from clipboard")
                else:
                    logger.info("⚠️  No camera pose in clipboard")
            elif key == glfw.KEY_R:
                self.viewer._initialize_camera_from_bounds()
                logger.info("🔄 Camera view reset to home position")
            elif key == glfw.KEY_G:
                self.viewer.show_scene_guides = not self.viewer.show_scene_guides
                state = "ON" if self.viewer.show_scene_guides else "OFF"
                logger.info(f"🔵 Scene guide planes: {state}")
            elif key == glfw.KEY_Z:
                self.viewer.camera_controller.roll_camera(np.deg2rad(-3.0))
            elif key == glfw.KEY_X:
                self.viewer.camera_controller.roll_camera(np.deg2rad(3.0))

    def mouse_button_callback(self, window, button, action, mods):
        """Handle mouse clicks, button-state updates, and UI hit testing.

        Args:
            window: GLFW window handle that received the event.
            button: GLFW mouse button identifier.
            action: GLFW button action.
            mods: Bitmask of active modifier keys.

        Returns:
            None: Viewer state is updated in place.
        """
        cursor_x, cursor_y = glfw.get_cursor_pos(window)
        win_w, win_h = glfw.get_window_size(window)
        canvas_w, canvas_h = self.bbox_ui.ui_canvas_size
        if win_w > 0 and win_h > 0 and canvas_w > 0 and canvas_h > 0:
            cursor_x = cursor_x * (float(canvas_w) / float(win_w))
            cursor_y = cursor_y * (float(canvas_h) / float(win_h))

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
            if self.bbox_ui.handle_ui_click(cursor_x, cursor_y):
                return
            cursor_y_flipped = max(0.0, float(canvas_h) - float(cursor_y))
            if self.bbox_ui.handle_ui_click(cursor_x, cursor_y_flipped):
                return

        if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE and self.bbox_ui.ui_drag_lock:
            self.bbox_ui.release_slider()
            # Keep ui_drag_lock alive while a slider text-edit is active so that
            # cursor motion doesn't bleed through to the camera.
            if self.bbox_ui.slider_editing_field is None:
                self.bbox_ui.ui_drag_lock = False
            return

        if self.bbox_ui.ui_drag_lock:
            # A click outside the panel while in slider text-edit cancels it.
            if (button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS
                    and self.bbox_ui.slider_editing_field is not None):
                if not self.bbox_ui.handle_ui_click(cursor_x, cursor_y):
                    cursor_y_flipped = max(0.0, float(canvas_h) - float(cursor_y))
                    if not self.bbox_ui.handle_ui_click(cursor_x, cursor_y_flipped):
                        self.bbox_ui.cancel_slider_edit()
                        self.bbox_ui.ui_drag_lock = False
            return

        if button == glfw.MOUSE_BUTTON_LEFT:
            self.viewer.mouse_left_pressed = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self.viewer.mouse_right_pressed = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.viewer.mouse_middle_pressed = action == glfw.PRESS

        self.viewer.shift_pressed = (mods & glfw.MOD_SHIFT) != 0
        self.viewer.ctrl_pressed = (mods & glfw.MOD_CONTROL) != 0
        if action == glfw.PRESS:
            self.viewer.prev_mouse_pos = self.viewer.mouse_pos.copy()

    def cursor_pos_callback(self, window, xpos, ypos):
        """Handle mouse motion and route it to rotate, pan, or zoom behavior.

        Args:
            window: GLFW window handle that received the event.
            xpos: Cursor x position in window coordinates.
            ypos: Cursor y position in window coordinates.

        Returns:
            None: Viewer and camera controller state are updated in place.
        """
        self.viewer.mouse_pos = np.array([xpos, ypos])

        if self.bbox_ui.ui_drag_lock:
            # Forward horizontal motion to an active slider drag.
            if self.bbox_ui.active_slider is not None:
                canvas_w, canvas_h = self.bbox_ui.ui_canvas_size
                win_w, win_h = glfw.get_window_size(window)
                scale_x = float(canvas_w) / float(win_w) if win_w > 0 else 1.0
                canvas_x = xpos * scale_x
                # Use the full slider track width from the stored rect if available
                bound_type, axis_idx = self.bbox_ui.active_slider
                slider_key = f"bbox_slider_{bound_type}_{axis_idx}"
                rect = self.bbox_ui.ui_rects.get(slider_key)
                track_w = float((rect[2] - rect[0])) if rect else 200.0
                self.bbox_ui.handle_slider_drag(canvas_x, slider_track_width=track_w)
            self.viewer.prev_mouse_pos = self.viewer.mouse_pos.copy()
            return

        if self.viewer.mouse_left_pressed or self.viewer.mouse_right_pressed or self.viewer.mouse_middle_pressed:
            delta = self.viewer.mouse_pos - self.viewer.prev_mouse_pos
            max_delta = 50.0
            delta = np.clip(delta, -max_delta, max_delta)
            is_pan = (self.viewer.shift_pressed and self.viewer.mouse_left_pressed) or self.viewer.mouse_right_pressed
            is_zoom = self.viewer.ctrl_pressed and self.viewer.shift_pressed and self.viewer.mouse_left_pressed
            is_rotate = self.viewer.mouse_left_pressed and not self.viewer.shift_pressed and not self.viewer.ctrl_pressed

            if is_zoom:
                zoom_factor = np.exp(-delta[1] * self.viewer.camera_zoom_speed)
                self.viewer.camera_controller.zoom(zoom_factor)
            elif is_rotate:
                self.viewer.camera_controller.orbit(
                    delta[0] * self.viewer.camera_rotation_speed,
                    delta[1] * self.viewer.camera_rotation_speed,
                )
            elif is_pan:
                self._pan_camera(delta)

            self.viewer.prev_mouse_pos = self.viewer.mouse_pos.copy()

    def _pan_camera(self, delta):
        """Translate the orbit target using screen-space mouse motion.

        Args:
            delta: Mouse delta in pixels.

        Returns:
            None: The camera target is updated in place.
        """
        cam_x = self.viewer.camera_controller.distance * np.sin(self.viewer.camera_controller.phi) * np.cos(self.viewer.camera_controller.theta)
        cam_y = self.viewer.camera_controller.distance * np.cos(self.viewer.camera_controller.phi)
        cam_z = self.viewer.camera_controller.distance * np.sin(self.viewer.camera_controller.phi) * np.sin(self.viewer.camera_controller.theta)
        eye = np.array([cam_x, cam_y, cam_z]) + np.array([
            self.viewer.camera_controller.pan_x,
            self.viewer.camera_controller.pan_y,
            self.viewer.camera_controller.pan_z,
        ])
        at = np.array([
            self.viewer.camera_controller.pan_x,
            self.viewer.camera_controller.pan_y,
            self.viewer.camera_controller.pan_z,
        ])
        up = np.array([0.0, 1.0, 0.0])

        forward = at - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm > 1e-6:
            forward = forward / forward_norm
        else:
            forward = np.array([0.0, 0.0, -1.0])

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
        else:
            right = np.array([1.0, 0.0, 0.0])

        up_normalized = np.cross(right, forward)
        pan_speed = self.viewer.camera_pan_speed * self.viewer.camera_controller.distance
        self.viewer.camera_controller.pan_x -= right[0] * delta[0] * pan_speed + up_normalized[0] * delta[1] * pan_speed
        self.viewer.camera_controller.pan_y -= right[1] * delta[0] * pan_speed + up_normalized[1] * delta[1] * pan_speed
        self.viewer.camera_controller.pan_z -= right[2] * delta[0] * pan_speed + up_normalized[2] * delta[1] * pan_speed
        self.viewer.camera_controller._update_camera_matrix()

    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse-wheel zooming.

        Args:
            window: GLFW window handle that received the event.
            xoffset: Horizontal scroll offset reported by GLFW.
            yoffset: Vertical scroll offset reported by GLFW.

        Returns:
            None: The camera zoom is updated in place.
        """
        zoom_factor = np.exp(-yoffset * self.viewer.camera_zoom_speed)
        self.viewer.camera_controller.zoom(zoom_factor)
