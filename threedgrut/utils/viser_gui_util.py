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

import time
import numpy as np
import torch
import viser
import viser.transforms as tf
from typing import Optional, Tuple, Dict, Any

from threedgrut.datasets.protocols import Batch, DatasetVisualization
from threedgrut.datasets.utils import fov2focal, DEFAULT_DEVICE
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.misc import to_np



class ViserGUI:
    def __init__(self, conf, model, train_dataset, val_dataset, scene_bbox):
        self.conf = conf
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.scene_bbox = scene_bbox
        
        # Initialize Viser server
        self.server = viser.ViserServer(port=8080)
        
        # GUI state
        self.viz_do_train = False
        self.viz_final = True
        self.training_done = False
        self.viz_bbox = False
        self.live_update = False
        self.viz_render_styles = ["color", "density", "distance", "hits", "normals"]
        self.viz_render_style = "color"
        self.viz_render_style_scale = 1.0
        self.viz_render_enabled = True
        self.viz_render_subsample = 1
        self.viz_render_train_view = True
        self.viz_render_show_details = False
        self.render_timer = CudaTimer()
        self.render_width = 1920
        self.render_height = 1080
        self.terminate_gui = False
        self.show_point_cloud = False

        # Initialize UI components
        self._init_ui()
        
        # Initialize scene visualization
        self.point_cloud = None
        self.init_point_cloud()

        @self.do_train_checkbox.on_update
        def _(_):
            self.viz_do_train = self.do_train_checkbox.value

        @self.live_update_checkbox.on_update
        def _(_):
            self.live_update = self.live_update_checkbox.value

        @self.show_render_checkbox.on_update
        def _(_):
            self.viz_render_enabled = self.show_render_checkbox.value

        @self.render_style_dropdown.on_update
        def _(_):
            self.viz_render_style = self.render_style_dropdown.value

        @self.terminate_gui_checkbox.on_update
        def _(_):
            self.terminate_gui = self.terminate_gui_checkbox.value

        @self.show_point_cloud_checkbox.on_update
        def _(_):
            self.show_point_cloud = self.show_point_cloud_checkbox.value
                

    def _init_ui(self):
        """Initialize UI components"""
        # Main control panel
        with self.server.gui.add_folder("Controls"):
            # Render controls
            self.render_style_dropdown = self.server.gui.add_dropdown(
                "Render Style",
                options=self.viz_render_styles,
                initial_value=self.viz_render_styles[0]
            )
            
            self.show_render_checkbox = self.server.gui.add_checkbox(
                "Show Render",
                initial_value=True
            )
            
            self.adjust_resolution_checkbox = self.server.gui.add_checkbox(
                "Adjust Browser Size",
                initial_value=False
            )

            self.resolution_slider = self.server.gui.add_slider(
                "Resolution", min=384, max=4096, 
                step=2, initial_value=1024
            )

            self.subsample_slider = self.server.gui.add_slider(
                "Subsample",
                min=1,
                max=8,
                step=1,
                initial_value=1
            )
            
            # Training controls
            self.do_train_checkbox = self.server.gui.add_checkbox(
                "Do Training",
                initial_value=False
            )
            
            self.live_update_checkbox = self.server.gui.add_checkbox(
                "Live Update",
                initial_value=False
            )

            self.terminate_gui_checkbox = self.server.gui.add_checkbox(
                "Terminate GUI",
                initial_value=False
            )

            self.show_point_cloud_checkbox = self.server.gui.add_checkbox(
                "Show Point Cloud",
                initial_value=False
            )

            # Camera controls
            self.camera_type_dropdown = self.server.gui.add_dropdown(
                "Camera Type",
                options=["Perspective", "Fisheye"],
                initial_value="Perspective"
            )
            
            # Export controls
            self.export_button = self.server.gui.add_button("Export Model")
            
    def init_point_cloud(self):
        # Add point cloud for gaussian centers
        self.point_cloud = self.server.scene.add_point_cloud(
            "3dgs object points",
            points=to_np(self.model.positions),
            colors = to_np(self.model.features_albedo),
            point_size=0.001    
        )

    def update_point_cloud(self):
        if self.show_point_cloud:
            if self.point_cloud is not None:
                self.point_cloud.points = to_np(self.model.positions)
                self.point_cloud.colors = to_np(self.model.features_albedo)
            else:
                self.init_point_cloud()
        else:
            self.remove_point_cloud()

    def remove_point_cloud(self):
        if self.point_cloud is not None:
            self.point_cloud.remove()
            self.point_cloud = None

    def get_c2w(self, camera):
        from threedgrut.utils.misc import quaternion_to_so3
        import numpy as np
        c2w = np.eye(4, dtype=np.float32)
        # camera.wxyz: (4,) numpy, quaternion (w, x, y, z)
        # quaternion_to_so3 expects (N,4) torch, so convert
        q = np.asarray(camera.wxyz)[None, :]
        q_torch = torch.from_numpy(q).float()
        R = quaternion_to_so3(q_torch)[0].cpu().numpy()
        c2w[:3, :3] = R
        c2w[:3, 3] = camera.position
        return c2w

    def get_w2c(self, camera):
        c2w = self.get_c2w(camera)
        w2c = np.linalg.inv(c2w)
        return w2c

    def render_from_current_view(self, client) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render from current camera view - rewritten to match polyscope version"""
        # Get current camera parameters from viser
        # for client in self.server.get_clients().values():
        camera = client.camera
        
        # Get window size and apply subsample
        
        if self.adjust_resolution_checkbox.value:
            window_w = self.render_width // self.viz_render_subsample
            window_h = self.render_height // self.viz_render_subsample
        else:
            window_w = self.resolution_slider.value
            window_h = int(self.resolution_slider.value / camera.aspect)


        
        # Get camera parameters from viser
        view_matrix = self.get_c2w(camera)  # This is W2C (world to camera)
        
        # Convert view matrix to camera-to-world (C2W)
        C2W = np.linalg.inv(view_matrix)
        C2W[:, 1:3] *= -1  # [right up back] to [right down front] - same as polyscope
        
        # Get FOV and calculate focal length
        fov_vertical_deg = camera.fov / np.pi * 180.0
        FOCAL = fov2focal(np.deg2rad(fov_vertical_deg), window_h)
        
        # Generate ray directions similar to polyscope version
        interp_x, interp_y = torch.meshgrid(
            torch.linspace(0.0, window_w - 1, window_w, device=DEFAULT_DEVICE, dtype=torch.float32),
            torch.linspace(0.0, window_h - 1, window_h, device=DEFAULT_DEVICE, dtype=torch.float32),
            indexing="xy",
        )
        u = interp_x
        v = interp_y
        
        xs = ((u + 0.5) - 0.5 * window_w) / FOCAL
        ys = ((v + 0.5) - 0.5 * window_h) / FOCAL
        rays_dir = torch.nn.functional.normalize(torch.stack((xs, ys, torch.ones_like(xs)), axis=-1), dim=-1).unsqueeze(0)
        
        # Create Batch object similar to polyscope version
        inputs = Batch(
            intrinsics=[FOCAL, FOCAL, window_w / 2, window_h / 2],
            T_to_world=torch.FloatTensor(C2W).unsqueeze(0),
            rays_ori=torch.zeros((1, window_h, window_w, 3), device=DEFAULT_DEVICE, dtype=torch.float32),
            rays_dir=rays_dir.reshape(1, window_h, window_w, 3),
        )
        
        # Render using model(inputs) instead of model.render()
        with torch.no_grad():
            self.render_timer.start()
            outputs = self.model(inputs, train=self.viz_render_train_view)
            self.render_timer.end()
            self.render_width = window_w
            self.render_height = window_h
        


        points = to_np(self.model.positions) 
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N,4)
        points_cam = (view_matrix @ points_h.T).T  # (N,4)

        X, Y, Z, _ = points_cam.T

        mask = Z > 0 
        X = X[mask]; Y = Y[mask]; Z = Z[mask]

        fx, fy = FOCAL, FOCAL
        cx, cy = window_w / 2, window_h / 2

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        points_plane = np.stack([u, v], axis=1)

        # points_plane = self.model.positions[u, v]
        # Return the same outputs as polyscope version
        return (
            outputs["pred_rgb"],
            outputs["pred_opacity"], 
            outputs["pred_dist"],
            outputs["pred_normals"],
            outputs["hits_count"] / self.conf.writer.max_num_hits,
            points_plane
        )

    def update_render_view(self, client, force: bool = False):            
        # Get current render style
        style = self.viz_render_style
        
        # Render current view
        sple_orad, sple_odns, sple_odist, sple_onrm, sple_ohit, points_plane = self.render_from_current_view(client)
        

        """Update rendered view - rewritten to match polyscope version"""
        if not self.viz_render_enabled and force:
            # Create a pure white background image
            # Get the shape from the model's features_specular to maintain dimensions
            rgb_np = to_np(sple_orad[0])
            img = np.ones(rgb_np.shape, dtype=np.float32)  # Pure white image
            client.scene.set_background_image(img)
            return

        # Update viser background image based on style
        if style == "color":
            # Convert RGB to numpy and set as background
            rgb_np = to_np(sple_orad[0])  # Remove batch dimension
            # rgb_np[points_plane[:, 0], points_plane[:, 1]] = [0, 0, 255]
            client.scene.set_background_image(rgb_np)
        elif style == "density":
            # Convert density to grayscale image
            density_np = to_np(sple_odns[0])  # Remove batch dimension
            # Normalize to 0-1 range for visualization
            density_np = np.clip(density_np, 0, 1)
            # Convert to RGB by repeating the channel
            rgb_np = np.stack([density_np, density_np, density_np], axis=-1)
            client.scene.set_background_image(rgb_np)
        elif style == "distance":
            # Convert distance to grayscale image
            distance_np = to_np(sple_odist[0])  # Remove batch dimension
            # Apply scale factor like polyscope version
            distance_np = (distance_np * self.viz_render_style_scale) / np.clip(to_np(sple_odns[0]), 1e-06, None)
            # Normalize to 0-1 range
            distance_np = np.clip(distance_np, 0, 1)
            # Convert to RGB by repeating the channel
            rgb_np = np.stack([distance_np, distance_np, distance_np], axis=-1)
            client.scene.set_background_image(rgb_np)
        elif style == "hits":
            # Convert hits count to grayscale image
            hits_np = to_np(sple_ohit[0])  # Remove batch dimension
            # Normalize to 0-1 range
            hits_np = np.clip(hits_np, 0, 1)
            # Convert to RGB by repeating the channel
            rgb_np = np.stack([hits_np, hits_np, hits_np], axis=-1)
            client.scene.set_background_image(rgb_np)
        elif style == "normals":
            # Convert normals to RGB image
            normals_np = to_np(sple_onrm[0])  # Remove batch dimension
            # Scale from [-1,1] to [0,1] like polyscope version
            normals_np = 0.5 * (normals_np + 1)
            client.scene.set_background_image(normals_np)

    def block_in_rendering_loop(self, fps: int = 60):
        """Block in rendering loop"""
        while not self.terminate_gui and self.training_done:
            for client in self.server.get_clients().values():
                self.update_render_view(client, force=True)
            time.sleep(1.0 / fps)