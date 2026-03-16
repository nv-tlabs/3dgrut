"""OpenGL window bootstrap utilities for the GLFW viewer.

This module owns GLFW context creation, shader compilation, the screen
quad used for final image presentation, and the OpenGL texture that is
updated every frame by the render loop.
"""

import ctypes

import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

from .shaders import VERTEX_SHADER, FRAGMENT_SHADER

class GLWindow:
    """Create and manage the GLFW window plus display-side GL objects.

    Args:
        width: Initial framebuffer width in pixels.
        height: Initial framebuffer height in pixels.
        title: Window title displayed by the operating system.

    Note:
        The class intentionally keeps a small surface area and only owns
        the resources required to show the renderer output texture.
    """

    def __init__(self, width, height, title="3DGRUT Interactive Viewer"):
        """Initialize GLFW, compile shaders, and allocate display resources.

        Args:
            width: Initial framebuffer width in pixels.
            height: Initial framebuffer height in pixels.
            title: Window title displayed by the operating system.

        Raises:
            RuntimeError: Raised when GLFW initialization or window creation fails.
        """
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)
        # 编译shader
        self.shader_program = compileProgram(
            compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
            compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        )
        # 创建四边形网格（用于显示纹理）
        self._setup_quad_vao()
        # 创建纹理
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        test_texture = np.ones((height, width, 3), dtype=np.float32) * 0.5
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_FLOAT, test_texture)

    def _setup_quad_vao(self):
        """Build the fullscreen quad VAO/VBO/EBO used for texture display.

        Returns:
            None: The created OpenGL object handles are stored on the instance.
        """
        quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,
             1.0, -1.0, 1.0, 0.0,
             1.0,  1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 1.0
        ], dtype=np.float32)
        quad_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self.quad_vao = glGenVertexArrays(1)
        self.quad_vbo = glGenBuffers(1)
        self.quad_ebo = glGenBuffers(1)
        glBindVertexArray(self.quad_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.quad_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        self.quad_index_count = len(quad_indices)

    def cleanup(self):
        """Release OpenGL resources and terminate the GLFW session.

        Returns:
            None: All GL objects owned by the window are deleted in place.
        """
        glDeleteBuffers(1, [self.quad_vbo])
        glDeleteBuffers(1, [self.quad_ebo])
        glDeleteVertexArrays(1, [self.quad_vao])
        glDeleteTextures([self.texture])
        glDeleteProgram(self.shader_program)
        glfw.destroy_window(self.window)
        glfw.terminate()
