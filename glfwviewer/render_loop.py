"""Rendering loop and final presentation helpers for the GLFW viewer.

This module connects the engine, camera controller, and display window.
It is responsible for running the render pass, optionally blending the
HUD overlay, and presenting the resulting image to the screen.
"""

import time
import numpy as np
import torch
from OpenGL.GL import *
import glfw

from threedgrut.utils.logger import logger

class RenderLoop:
    """Drive frame rendering, display, and event polling.

    Args:
        window: ``GLWindow`` instance that owns display-side GL objects.
        engine: Rendering backend used to generate RGB frames.
        camera_controller: Camera state provider for the engine.
        viewer: Optional high-level viewer used for HUD composition and FPS sync.
    """

    def __init__(self, window, engine, camera_controller, viewer=None):
        """Store rendering dependencies and initialize FPS counters.

        Args:
            window: ``GLWindow`` instance that owns display-side GL objects.
            engine: Rendering backend used to generate RGB frames.
            camera_controller: Camera state provider for the engine.
            viewer: Optional high-level viewer used for HUD composition and FPS sync.
        """
        self.window = window
        self.engine = engine
        self.camera_controller = camera_controller
        self.viewer = viewer
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.stage_time_render = 0.0
        self.stage_time_hud = 0.0
        self.stage_time_upload = 0.0
        self.stage_time_display = 0.0
        self.stage_time_swap_poll = 0.0

    @torch.no_grad()
    def render_frame(self):
        """Render a frame and upload it into the screen texture.

        Returns:
            None: The uploaded texture is updated in place on the OpenGL side.

        Note:
            The method mirrors the original viewer's frame post-processing
            order so rendering behavior stays aligned after refactoring.
        """
        t0 = time.perf_counter()
        camera = self.camera_controller.get_camera()
        is_first_pass = self.engine.is_dirty(camera)
        outputs = self.engine.render_pass(camera, is_first_pass=is_first_pass)
        rgb = outputs["rgb"]
        t1 = time.perf_counter()

        if self.frame_count < 5:
            try:
                rgb_min = float(rgb.min().item()) if isinstance(rgb, torch.Tensor) else float(np.min(rgb))
                rgb_max = float(rgb.max().item()) if isinstance(rgb, torch.Tensor) else float(np.max(rgb))
            except Exception:
                rgb_min, rgb_max = 0.0, 0.0
            logger.info(
                f"Frame {self.frame_count}: is_first_pass={is_first_pass}, "
                f"RGB shape: {getattr(rgb, 'shape', None)}, min: {rgb_min:.6f}, max: {rgb_max:.6f}"
            )

        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()

        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)

        rgb = np.clip(rgb, 0.0, 1.0)

        if rgb.ndim == 4:
            rgb = rgb[0]

        if rgb.ndim == 3:
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            elif rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)

        if self.viewer is not None:
            rgb = self.viewer._apply_hud_overlay(rgb)
        t2 = time.perf_counter()

        rgb = np.ascontiguousarray(rgb)
        glBindTexture(GL_TEXTURE_2D, self.window.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, rgb.shape[1], rgb.shape[0], GL_RGB, GL_FLOAT, rgb)
        t3 = time.perf_counter()
        self.stage_time_render += (t1 - t0)
        self.stage_time_hud += (t2 - t1)
        self.stage_time_upload += (t3 - t2)

    def display(self):
        """Draw the fullscreen quad textured with the latest rendered frame.

        Returns:
            None: The current OpenGL back buffer is updated in place.
        """
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.window.shader_program)
        glBindVertexArray(self.window.quad_vao)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.window.texture)
        texture_loc = glGetUniformLocation(self.window.shader_program, "texture1")
        glUniform1i(texture_loc, 0)
        glDrawElements(GL_TRIANGLES, self.window.quad_index_count, GL_UNSIGNED_INT, None)

    def run(self):
        """Execute the main rendering loop until the window is closed.

        Returns:
            None: Control returns only after the GLFW window is closed.
        """
        while not glfw.window_should_close(self.window.window):
            t_display0 = time.perf_counter()
            self.render_frame()
            self.display()
            t_display1 = time.perf_counter()
            glfw.swap_buffers(self.window.window)
            glfw.poll_events()
            t_display2 = time.perf_counter()
            self.stage_time_display += (t_display1 - t_display0)
            self.stage_time_swap_poll += (t_display2 - t_display1)
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_time)
                if self.viewer is not None:
                    self.viewer.fps = self.fps
                frame_denom = max(self.frame_count, 1)
                logger.info(
                    f"FPS: {self.fps:.1f} | ms/frame: "
                    f"render={1000.0 * self.stage_time_render / frame_denom:.2f}, "
                    f"hud={1000.0 * self.stage_time_hud / frame_denom:.2f}, "
                    f"upload={1000.0 * self.stage_time_upload / frame_denom:.2f}, "
                    f"display={1000.0 * self.stage_time_display / frame_denom:.2f}, "
                    f"swap+poll={1000.0 * self.stage_time_swap_poll / frame_denom:.2f}"
                )
                self.frame_count = 0
                self.last_time = current_time
                self.stage_time_render = 0.0
                self.stage_time_hud = 0.0
                self.stage_time_upload = 0.0
                self.stage_time_display = 0.0
                self.stage_time_swap_poll = 0.0
        self.window.cleanup()
