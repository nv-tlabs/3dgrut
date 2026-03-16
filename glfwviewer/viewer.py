"""High-level coordinator for the modular GLFW viewer.

This module wires together the engine, window, camera, HUD, exporter,
scene-state helpers, and input controllers that were split out from the
legacy ``glfwviwer.py`` monolith.
"""

import os
import glfw
import numpy as np
from threedgrut_playground.engine import Engine3DGRUT
from threedgrut.utils.logger import logger
from .window import GLWindow
from .camera import CameraController
from .render_loop import RenderLoop
from .hud import HUD
from .exporter import Exporter
from .bbox_panel import BBoxUIController
from .interaction import InteractionController
from .scene_state import SceneStateController

class InteractiveViewer:
    """Coordinate all subsystems required by the interactive viewer.

    Args:
        gs_object: Path to the gaussian scene asset to load.
        mesh_assets_folder: Optional mesh-assets directory for the engine.
        default_config: Default render/export config path.
        envmap_assets_folder: Optional environment-map assets directory.
        width: Initial window width in pixels.
        height: Initial window height in pixels.
        buffer_mode: CUDA/OpenGL interop mode passed through for compatibility.

    Note:
        The class intentionally preserves the public calling style and user
        interactions of the original standalone viewer script.
    """

    def __init__(self, 
                 gs_object: str,
                 mesh_assets_folder: str = None,
                 default_config: str = "apps/colmap_3dgrt.yaml",
                 envmap_assets_folder: str = None,
                 width: int = 1920, 
                 height: int = 1080,
                 buffer_mode: str = "device2device"):
        """Initialize the viewer and all of its collaborating controllers.

        Args:
            gs_object: Path to the gaussian scene asset to load.
            mesh_assets_folder: Optional mesh-assets directory for the engine.
            default_config: Default render/export config path.
            envmap_assets_folder: Optional environment-map assets directory.
            width: Initial window width in pixels.
            height: Initial window height in pixels.
            buffer_mode: CUDA/OpenGL interop mode passed through for compatibility.
        """
        self.window_width = width
        self.window_height = height
        self.buffer_mode = buffer_mode
        self.gs_object = gs_object
        self.default_config = default_config
        logger.info(f"Loading 3DGRUT model from {gs_object}...")
        if mesh_assets_folder is None:
            mesh_assets_folder = os.path.join(os.path.dirname(__file__), "..", "threedgrut_playground", "assets")
        if envmap_assets_folder is None:
            envmap_assets_folder = os.path.join(os.path.dirname(__file__), "..", "threedgrut_playground", "assets")
        mesh_assets_folder = os.path.abspath(mesh_assets_folder)
        envmap_assets_folder = os.path.abspath(envmap_assets_folder)
        self.engine = Engine3DGRUT(
            gs_object=gs_object,
            mesh_assets_folder=mesh_assets_folder,
            default_config=default_config,
            envmap_assets_folder=envmap_assets_folder
        )
        logger.info("3DGRUT engine initialized successfully!")
        self.engine.use_spp = False
        self.engine.use_depth_of_field = False
        self.engine.camera_type = "Pinhole"
        self.engine.camera_fov = 45.0

        self.scene_state = SceneStateController(self)
        self._compute_scene_bounds()

        # 组合各功能模块
        self.window = GLWindow(width, height)
        self.camera_controller = CameraController(
            fov=45.0,
            width=width,
            height=height,
            near=0.01,
            far=100.0,
            device=self.engine.device,
        )
        self.hud = HUD()
        self.hud.initialize_hud_state_cache(self.engine)
        self.exporter = Exporter(self.engine)
        self.bbox_ui = BBoxUIController(self)
        self.fps = 0.0
        self.loop = RenderLoop(self.window, self.engine, self.camera_controller, viewer=self)
        # 事件与交互相关状态
        self.mouse_pos = np.array([0.0, 0.0])
        self.prev_mouse_pos = np.array([0.0, 0.0])
        self.mouse_left_pressed = False
        self.mouse_right_pressed = False
        self.mouse_middle_pressed = False
        self.shift_pressed = False
        self.ctrl_pressed = False
        self.camera_pose_clipboard = None
        self.camera_rotation_speed = 0.0064
        self.camera_pan_speed = 0.001
        self.camera_zoom_speed = 0.1
        self.render_style_index = 0
        self.render_styles = ["color", "density"]
        self.show_hud_overlay = True
        self.show_scene_guides = True
        # BBox与UI相关状态
        self.bbox_filter_enabled = self.bbox_ui.filter_enabled
        self.bbox_last_kept_count = self.bbox_ui.last_kept_count
        self.bbox_status_message = self.bbox_ui.status_message
        self.bbox_last_applied_min = self.bbox_ui.last_applied_min
        self.bbox_last_applied_max = self.bbox_ui.last_applied_max
        self.bbox_ui_rects = self.bbox_ui.ui_rects
        self.bbox_ui_canvas_size = self.bbox_ui.ui_canvas_size
        self.bbox_ui_drag_lock = self.bbox_ui.ui_drag_lock
        self._initialize_bbox_ui_state()
        self.scene_state.initialize_camera_from_bounds()
        self.interaction = InteractionController(self, self.bbox_ui)
        # 注册GLFW事件
        glfw.set_framebuffer_size_callback(self.window.window, self._framebuffer_size_callback)
        glfw.set_key_callback(self.window.window, self._key_callback)
        glfw.set_mouse_button_callback(self.window.window, self._mouse_button_callback)
        glfw.set_cursor_pos_callback(self.window.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window.window, self._scroll_callback)
        glfw.set_char_callback(self.window.window, self._char_callback)
        logger.info("✅ Interactive Viewer initialized successfully!")
        logger.info("\n🎮 Controls (Turntable Mode - 与 Polyscope 一致):")
        logger.info("  Left Mouse Drag           → Rotate camera (旋转相机)")
        logger.info("  Shift + Left Mouse Drag   → Pan camera (平移相机)")
        logger.info("  Right Mouse Drag          → Pan camera (平移相机)")
        logger.info("  Ctrl + Shift + Left Drag  → Zoom in/out (缩放)")
        logger.info("  Scroll Wheel              → Zoom in/out (缩放)")
        logger.info("  S                         → Switch render style (切换渲染样式)")
        logger.info("  H                         → Toggle HUD overlay (切换HUD显示)")
        logger.info("  R                         → Reset view (重置视图)")
        logger.info("  G                         → Toggle guide planes (切换旋转平面显示)")
        logger.info("  Z / X                     → Roll camera -/+ (相机滚转)")
        logger.info("  Ctrl + C                  → Copy camera pose (复制相机姿态)")
        logger.info("  Ctrl + V                  → Paste camera pose (粘贴相机姿态)")
        logger.info("  Q / ESC                   → Quit (退出)")

    def _compute_scene_bounds(self):
        """Refresh scene-bound caches mirrored from ``SceneStateController``.

        Returns:
            None: Bound-related attributes on the viewer are updated in place.
        """
        self.scene_state.compute_scene_bounds()
        self.scene_bbox_min = self.scene_state.scene_bbox_min
        self.scene_bbox_max = self.scene_state.scene_bbox_max
        self.scene_diagonal = self.scene_state.scene_diagonal
        self.scene_center = self.scene_state.scene_center
        self.scene_bbox_min_fixed = self.scene_state.scene_bbox_min_fixed
        self.scene_bbox_max_fixed = self.scene_state.scene_bbox_max_fixed

    def _initialize_camera_from_bounds(self):
        """Reset the camera to the scene home pose.

        Returns:
            None: Camera controller state is updated in place.
        """
        self.scene_state.initialize_camera_from_bounds()

    def _build_hud_lines(self):
        """Build the current text lines for the runtime HUD overlay.

        Returns:
            list[str]: HUD lines describing runtime, input, and scene state.
        """
        self._sync_bbox_state_from_controller()
        runtime_state = {
            "fps": float(getattr(self, "fps", 0.0)),
            "render_style": self.render_styles[self.render_style_index],
        }
        interaction_state = {
            "mouse_pos": self.mouse_pos,
            "mouse_left_pressed": self.mouse_left_pressed,
            "mouse_right_pressed": self.mouse_right_pressed,
            "mouse_middle_pressed": self.mouse_middle_pressed,
            "shift_pressed": self.shift_pressed,
            "ctrl_pressed": self.ctrl_pressed,
        }
        camera_state = {
            "theta": self.camera_controller.theta,
            "phi": self.camera_controller.phi,
            "roll": self.camera_controller.roll,
            "distance": self.camera_controller.distance,
        }
        bbox_state = {
            "enabled": self.bbox_filter_enabled,
            "status_message": self.bbox_status_message,
        }
        return self.hud.build_hud_lines(
            self.engine,
            runtime_state,
            interaction_state,
            camera_state,
            bbox_state,
        )

    def _apply_hud_overlay(self, rgb):
        """Blend HUD, scene guides, and BBox UI onto a rendered RGB frame.

        Args:
            rgb: Float RGB frame produced by the engine.

        Returns:
            np.ndarray: RGB frame after overlay composition.
        """
        if not self.show_hud_overlay:
            return rgb
        self._sync_bbox_state_from_controller()
        lines = self._build_hud_lines()
        bbox_ui_state = {
            "filter_enabled": self.bbox_filter_enabled,
            "status_message": self.bbox_status_message,
            "rects": {},
            "canvas_size": self.bbox_ui_canvas_size,
            "slider_values": self.bbox_ui.slider_values,
            "slider_center_values": self.bbox_ui.slider_center_values,
            "active_slider": self.bbox_ui.active_slider,
            "slider_editing_field": self.bbox_ui.slider_editing_field,
            "slider_edit_text": self.bbox_ui.slider_edit_text,
            "slider_edit_selected": self.bbox_ui.slider_edit_selected,
        }
        scene_state = self.scene_state.build_hud_scene_state() if self.show_scene_guides else None
        view_matrix = self.camera_controller.get_view_matrix()
        rgb, bbox_ui_state = self.hud.draw_hud_overlay(
            rgb,
            lines,
            bbox_ui_state=bbox_ui_state,
            scene_state=scene_state,
            view_matrix=view_matrix,
        )
        self.bbox_ui.ui_rects = bbox_ui_state["rects"]
        self.bbox_ui.ui_canvas_size = bbox_ui_state["canvas_size"]
        self._sync_bbox_state_from_controller()
        return rgb

    def _sync_bbox_state_from_controller(self):
        """Mirror BBox UI controller state onto legacy viewer attributes.

        Returns:
            None: Compatibility attributes are updated in place.

        Note:
            These mirrored attributes help keep parity with logic that still
            expects the original monolithic viewer field layout.
        """
        self.bbox_filter_enabled = self.bbox_ui.filter_enabled
        self.bbox_last_kept_count = self.bbox_ui.last_kept_count
        self.bbox_status_message = self.bbox_ui.status_message
        self.bbox_last_applied_min = self.bbox_ui.last_applied_min
        self.bbox_last_applied_max = self.bbox_ui.last_applied_max
        self.bbox_ui_rects = self.bbox_ui.ui_rects
        self.bbox_ui_canvas_size = self.bbox_ui.ui_canvas_size
        self.bbox_ui_drag_lock = self.bbox_ui.ui_drag_lock

    def _framebuffer_size_callback(self, window, width, height):
        """Handle framebuffer resize events and recreate the display texture.

        Args:
            window: GLFW window handle that received the event.
            width: New framebuffer width in pixels.
            height: New framebuffer height in pixels.

        Returns:
            None: Window, texture, camera, and UI canvas state are updated in place.
        """
        if width > 0 and height > 0:
            self.window_width = width
            self.window_height = height
            from OpenGL.GL import glViewport, glDeleteTextures, glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, GL_TEXTURE_2D, GL_LINEAR, GL_CLAMP_TO_EDGE, GL_RGB, GL_FLOAT
            glViewport(0, 0, width, height)
            glDeleteTextures([self.window.texture])
            self.window.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.window.texture)
            glTexParameteri(GL_TEXTURE_2D, 0x2801, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, 0x2800, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, 0x2802, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, 0x2803, GL_CLAMP_TO_EDGE)
            import numpy as np
            test_texture = np.ones((height, width, 3), dtype=np.float32) * 0.2
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, test_texture)
            self.camera_controller.set_resolution(width, height)
            self.bbox_ui.ui_canvas_size = (width, height)
            self._sync_bbox_state_from_controller()

    def _key_callback(self, window, key, scancode, action, mods):
        """Forward keyboard input to the dedicated interaction controller.

        Args:
            window: GLFW window handle that received the event.
            key: GLFW key code.
            scancode: Platform-dependent scan code.
            action: GLFW key action.
            mods: Bitmask of active modifier keys.

        Returns:
            None: Input handling side effects are applied to viewer state.
        """
        self.interaction.key_callback(window, key, scancode, action, mods)

    def _mouse_button_callback(self, window, button, action, mods):
        """Forward mouse-button events and resync mirrored BBox state.

        Args:
            window: GLFW window handle that received the event.
            button: GLFW mouse button identifier.
            action: GLFW button action.
            mods: Bitmask of active modifier keys.

        Returns:
            None: Input handling side effects are applied to viewer state.
        """
        self.interaction.mouse_button_callback(window, button, action, mods)
        self._sync_bbox_state_from_controller()

    def _cursor_pos_callback(self, window, xpos, ypos):
        """Forward cursor-motion events to the interaction controller.

        Args:
            window: GLFW window handle that received the event.
            xpos: Cursor x position in window coordinates.
            ypos: Cursor y position in window coordinates.

        Returns:
            None: Viewer mouse and camera state are updated in place.
        """
        self.interaction.cursor_pos_callback(window, xpos, ypos)

    def _scroll_callback(self, window, xoffset, yoffset):
        """Forward scroll-wheel events to the interaction controller.

        Args:
            window: GLFW window handle that received the event.
            xoffset: Horizontal scroll offset reported by GLFW.
            yoffset: Vertical scroll offset reported by GLFW.

        Returns:
            None: Viewer zoom state is updated in place.
        """
        self.interaction.scroll_callback(window, xoffset, yoffset)

    def _char_callback(self, window, codepoint):
        """Forward character input to the active BBox text field.

        Args:
            window: GLFW window handle that received the event.
            codepoint: Unicode codepoint reported by GLFW.

        Returns:
            None: BBox text-field state is updated in place.
        """
        self.bbox_ui.handle_char_input(codepoint)
        self._sync_bbox_state_from_controller()

    def _initialize_bbox_ui_state(self):
        """Initialize the BBox UI controller using the current window size.

        Returns:
            None: UI state and mirrored attributes are refreshed in place.
        """
        self.bbox_ui.initialize_state(self.window_width, self.window_height)
        self._sync_bbox_state_from_controller()

    def _apply_bbox_filter(self, min_vals, max_vals):
        """Apply the requested BBox filter through the UI controller.

        Args:
            min_vals: Minimum BBox corner in world coordinates.
            max_vals: Maximum BBox corner in world coordinates.

        Returns:
            bool: ``True`` when the filter keeps at least one gaussian.
        """
        result = self.bbox_ui.apply_bbox_filter(min_vals, max_vals)
        self._sync_bbox_state_from_controller()
        return result

    def _disable_bbox_filter(self):
        """Disable the active BBox filter and resync compatibility state.

        Returns:
            None: Filter state and mirrored attributes are updated in place.
        """
        self.bbox_ui.disable_bbox_filter()
        self._sync_bbox_state_from_controller()

    def _toggle_bbox_filter_from_ui(self):
        """Toggle the BBox filter from current UI field contents.

        Returns:
            None: Filter state and mirrored attributes are updated in place.
        """
        self.bbox_ui.toggle_bbox_filter_from_ui()
        self._sync_bbox_state_from_controller()

    def _export_current_gaussians_to_usdz(self):
        """Export the currently visible gaussian selection to disk.

        Returns:
            None: Export status and mirrored attributes are updated in place.
        """
        self.bbox_ui.export_current_gaussians_to_usdz()
        self._sync_bbox_state_from_controller()

    def _point_in_rect(self, x, y, rect):
        """Delegate rectangle hit testing to the BBox UI controller.

        Args:
            x: Point x coordinate.
            y: Point y coordinate.
            rect: Rectangle stored as ``(x1, y1, x2, y2)``.

        Returns:
            bool: ``True`` when the point lies inside the rectangle.
        """
        return self.bbox_ui.point_in_rect(x, y, rect)

    def _handle_ui_click(self, x, y):
        """Delegate a click event to the BBox UI controller.

        Args:
            x: Click x coordinate in UI canvas space.
            y: Click y coordinate in UI canvas space.

        Returns:
            bool: ``True`` when the click is consumed by the UI.
        """
        result = self.bbox_ui.handle_ui_click(x, y)
        self._sync_bbox_state_from_controller()
        return result

    def run(self):
        """Enter the main rendering loop.

        Returns:
            None: Control returns only after the viewer is closed.
        """
        self.loop.run()
