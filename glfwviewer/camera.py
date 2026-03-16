"""Turntable-style camera controller for the GLFW viewer.

The controller maintains orbit, pan, zoom, and projection parameters,
and exposes a Kaolin ``Camera`` object that stays synchronized with the
interactive state used by mouse and keyboard callbacks.
"""

import numpy as np
import torch
from kaolin.render.camera import Camera

class CameraController:
    """Manage interactive camera state and derived matrices.

    Args:
        fov: Vertical field of view in degrees.
        width: Render target width in pixels.
        height: Render target height in pixels.
        near: Near clipping plane distance.
        far: Far clipping plane distance.
        device: Torch device on which the Kaolin camera should live.
    """

    def __init__(self, fov=45.0, width=1920, height=1080, near=0.01, far=100.0, device="cpu"):
        """Initialize the camera controller with default turntable settings.

        Args:
            fov: Vertical field of view in degrees.
            width: Render target width in pixels.
            height: Render target height in pixels.
            near: Near clipping plane distance.
            far: Far clipping plane distance.
            device: Torch device used for the Kaolin camera tensor state.
        """
        self.fov = fov
        self.width = width
        self.height = height
        self.near = near
        self.far = far
        self.device = device
        # Turntable模式参数
        self.theta = 0.0
        self.phi = np.pi / 4.0
        self.roll = 0.0
        self.distance = 3.625
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.pan_z = 0.0
        self._update_camera_matrix()

    def _update_camera_matrix(self):
        """Recompute the view matrix and rebuild the Kaolin camera object.

        Returns:
            None: Updated matrix data is stored in ``view_matrix_np`` and
            ``camera``.

        Note:
            The view matrix is kept both as a NumPy array for HUD projection
            helpers and as a torch tensor for the rendering backend.
        """
        cam_x = self.distance * np.sin(self.phi) * np.cos(self.theta)
        cam_y = self.distance * np.cos(self.phi)
        cam_z = self.distance * np.sin(self.phi) * np.sin(self.theta)
        eye = np.array([cam_x + self.pan_x, cam_y + self.pan_y, cam_z + self.pan_z], dtype=np.float64)
        at = np.array([self.pan_x, self.pan_y, self.pan_z], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        forward = at - eye
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0])
        else:
            forward = forward / forward_norm
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            up = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        up_normalized = np.cross(right, forward)
        if abs(self.roll) > 1e-10:
            cos_roll = np.cos(self.roll)
            sin_roll = np.sin(self.roll)
            right_rot = right * cos_roll + up_normalized * sin_roll
            up_rot = -right * sin_roll + up_normalized * cos_roll
            right = right_rot / max(np.linalg.norm(right_rot), 1e-8)
            up_normalized = up_rot / max(np.linalg.norm(up_rot), 1e-8)
        view_matrix = np.eye(4, dtype=np.float64)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up_normalized
        view_matrix[2, :3] = -forward
        view_matrix[0, 3] = -np.dot(right, eye)
        view_matrix[1, 3] = -np.dot(up_normalized, eye)
        view_matrix[2, 3] = np.dot(forward, eye)
        self.view_matrix_np = view_matrix
        fov_rad = np.deg2rad(self.fov)
        self.camera = Camera.from_args(
            view_matrix=torch.from_numpy(view_matrix).unsqueeze(0).to(self.device),
            fov=fov_rad,
            width=self.width,
            height=self.height,
            near=self.near,
            far=self.far,
            dtype=torch.float64,
            device=self.device
        )

    def set_resolution(self, width, height):
        """Update render resolution and rebuild the internal camera object.

        Args:
            width: New render width in pixels.
            height: New render height in pixels.

        Returns:
            None: The resolution change is applied immediately.
        """
        self.width = width
        self.height = height
        self._update_camera_matrix()

    def set_fov(self, fov):
        """Update the vertical field of view.

        Args:
            fov: New field of view in degrees.

        Returns:
            None: The projection settings are rebuilt immediately.
        """
        self.fov = fov
        self._update_camera_matrix()

    def orbit(self, dtheta, dphi):
        """Orbit the camera around the current target point.

        Args:
            dtheta: Horizontal angular delta in radians.
            dphi: Vertical angular delta in radians.

        Returns:
            None: The updated orbit state is applied to the camera.
        """
        self.theta += dtheta
        self.phi = np.clip(self.phi + dphi, 0.01, np.pi - 0.01)
        self._update_camera_matrix()

    def roll_camera(self, droll):
        """Roll the camera around its forward axis.

        Args:
            droll: Roll angular delta in radians.

        Returns:
            None: The updated roll state is applied to the camera.
        """
        self.roll += droll
        self._update_camera_matrix()

    def pan(self, dx, dy, dz=0.0):
        """Translate the camera target in world space.

        Args:
            dx: Translation along the world-space x direction.
            dy: Translation along the world-space y direction.
            dz: Translation along the world-space z direction.

        Returns:
            None: The camera target and view matrix are updated in place.
        """
        self.pan_x += dx
        self.pan_y += dy
        self.pan_z += dz
        self._update_camera_matrix()

    def zoom(self, factor):
        """Scale the orbit radius while clamping it to a safe range.

        Args:
            factor: Multiplicative zoom factor.

        Returns:
            None: The new camera distance is applied immediately.
        """
        self.distance = np.clip(self.distance * factor, 0.1, 100.0)
        self._update_camera_matrix()

    def get_camera(self):
        """Return the current Kaolin camera used by the renderer.

        Returns:
            Camera: The latest camera object synchronized with controller state.
        """
        return self.camera

    def get_view_matrix(self):
        """Return the current view matrix as a NumPy array.

        Returns:
            np.ndarray: A ``4 x 4`` world-to-view transform matrix.
        """
        return self.view_matrix_np
