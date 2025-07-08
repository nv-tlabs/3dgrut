# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from kaolin.render.camera import Camera
from threedgrut_playground.utils.kaolin_future.interpolated_cameras import camera_path_generator


class VideoRecorder:
    """ A module for saving video footage of camera trajectories within the Playground"""

    MODES = ['path_smooth', 'path_spline', 'cyclic', 'depth_of_field']

    def __init__(self,
        renderer,
        trajectory_output_path="output.mp4",
        cameras_save_path="cameras.npy",
        mode='path_smooth',
        frames_between_cameras=60,
        video_fps=30,
        min_dof=2.5,
        max_dof=24
    ):
        """
        Creates a video recorder for saving camera animation along trajectories.
        Args:
            renderer: Reference to the Playground rendering engine, interfaced through a 'render()' function.
            trajectory_output_path (str): Output path for saved video
            cameras_save_path (str): Save path for storing and loading camera trajectories.
            mode (str): Determines how keyframes are interpolated:
                'path_smooth' - shifts the camera in a path along the trajectory, using smoothstep interpolation
                'path_spline' - shifts the camera in a path along the trajectory, using catmull-rom splines
                'cyclic' - shifts the camera in a cyclic trajectory, determined by fitting a bspline
                'depth_of_field' - interpolates the depth of field over a static frame (first frame in trajectory)
            frames_between_cameras (int): How many cameras are inserted between keyframe cameras in the trajectory
            video_fps (int): FPS of exported video
            min_dof (float): For 'depth_of_field' mode only, determines the minimum dof for interpolation start / end
            max_dof (float): For 'depth_of_field' mode only, determines the maximum dof for interpolation start / end
        """
        self.renderer = renderer

        # List of kaolin.render.camera.Camera objects forming the trajectory
        self.trajectory = []

        # Selected interpolation modes
        self.mode = mode

        # Output path for generated video file
        self.trajectory_output_path = trajectory_output_path

        # For saving and loading trajectory paths
        self.cameras_save_path = cameras_save_path

        # Camera FPS -- how many cameras are interpolated between key views
        self.frames_between_cameras = frames_between_cameras

        # Saved Video FPS
        self.video_fps = video_fps

        # For depth-of-field interpolation mode only - determines the boundary
        self.min_dof = min_dof
        self.max_dof = max_dof

    def add_camera(self, camera: Camera):
        self.trajectory.append(camera)

    def reset_trajectory(self):
        self.trajectory = []

    def save_trajectory(self):
        torch.save(self.trajectory, self.cameras_save_path)

    def load_trajectory(self):
        self.trajectory = torch.load(self.cameras_save_path)

    def render_dof_trajectory(self):
        out_video = None
        old_use_dof = self.renderer.use_depth_of_field
        old_focus_z = self.renderer.depth_of_field.focus_z
        try:
            camera = self.trajectory[0]
            dofs = np.linspace(self.min_dof, self.max_dof, self.frames_between_cameras)
            for dof in tqdm(dofs):
                self.renderer.use_depth_of_field = True
                self.renderer.depth_of_field.focus_z = dof
                rgb = self.renderer.render(camera)['rgb']

                if out_video is None:
                    out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
                data = rgb[0].clip(0, 1).detach().cpu().numpy()
                data = (data * 255).astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                out_video.write(data)
        finally:
            self.renderer.use_depth_of_field = old_use_dof
            self.renderer.depth_of_field.focus_z = old_focus_z

    def render_linear_trajectory(self, interpolation_mode='polynomial'):
        if len(self.trajectory) < 2:
            raise ValueError('Rendering a path trajectory requires at least 2 cameras.')
        elif interpolation_mode == 'catmull_rom' and len(self.trajectory) < 4:
            raise ValueError('Rendering a path with a spline interpolated trajectory requires at least 4 cameras.')

        out_video = None
        interpolated_path = camera_path_generator(
            trajectory=self.trajectory,
            frames_between_cameras=self.frames_between_cameras,
            interpolation=interpolation_mode
        )

        for camera in tqdm(interpolated_path):
            rgb = self.renderer.render(camera)['rgb']

            if out_video is None:
                out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)
        out_video.release()

    def render_continuous_trajectory(self):
        if len(self.trajectory) < 4:
            raise ValueError('Rendering a continuous trajectory requires at least 4 cameras.')

        def _fit_bspline(data_points, frames_between_cameras):
            stacked_data_points = torch.stack([dp for dp in data_points], dim=1).cpu().numpy()
            tck, u = splprep(stacked_data_points, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), frames_between_cameras * len(data_points))
            interpolated_cam_param = np.stack(splev(u_new, tck, der=0)).T
            interpolated_cam_param = torch.tensor(interpolated_cam_param)
            return interpolated_cam_param

        cam_poses = _fit_bspline(
            data_points=[c.cam_pos().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ats = _fit_bspline(
            data_points=[c.cam_pos().squeeze() - c.cam_forward().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ups = _fit_bspline(
            data_points=[c.cam_up().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )

        first_cam = self.trajectory[0]
        interpolated_path = []
        for eye, at, up in tqdm(zip(cam_poses, cam_ats, cam_ups)):
            interpolated_path.append(
                Camera.from_args(
                    eye=eye,
                    at=at,
                    up=up,
                    fov=first_cam.fov(in_degrees=False),
                    width=first_cam.width, height=first_cam.height,
                    device=first_cam.device
                )
            )

        out_video = None
        for camera in tqdm(interpolated_path):
            rgb = self.renderer.render(camera)['rgb']

            if out_video is None:
                out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)
        out_video.release()

    def render_video(self):
        """
        Renders the trajectory to a video file, according to the set mode (see init()).
        """
        if self.mode == 'depth_of_field':
            self.render_dof_trajectory()
        elif self.mode == 'cyclic':
            self.render_continuous_trajectory()
        elif self.mode == 'path_smooth':
            self.render_linear_trajectory('polynomial')
        elif self.mode == 'path_spline':
            self.render_linear_trajectory('catmull_rom')
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
