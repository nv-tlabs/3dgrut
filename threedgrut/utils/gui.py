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

import cv2
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from threedgrut.datasets.protocols import Batch, DatasetVisualization
from threedgrut.datasets.utils import fov2focal, DEFAULT_DEVICE
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import CudaTimer
from threedgrut.utils.misc import to_np


trajectory = []


class GUI:
    def __init__(self, conf, model, train_dataset, val_dataset, scene_bbox):
        self.conf = conf

        self.update_from_device = self.conf.gui_update_from_device
        if not self.update_from_device:
            logger.info("polyscope set to host2device mode.")
        else:  # device2device
            from threedgrt_tracer.gui.ps_extension import initialize_cugl_interop
            initialize_cugl_interop()
            logger.info("polyscope set to device2device mode.")

        ps.set_use_prefs_file(False)

        if self.conf.dataset.type == "nerf":  # NeRF synthetic uses the blender coordinate-system
            ps.set_up_dir("z_up")
            ps.set_front_dir("neg_y_front")
            ps.set_navigation_style("turntable")
        elif self.conf.dataset.type == "colmap":  # Colmap scenes use a cartesian coordinate-system
            ps.set_up_dir("neg_y_up")
            ps.set_front_dir("neg_z_front")
            ps.set_navigation_style("free")
        else:  # AV use cartesian coordinate-system with z-up
            ps.set_up_dir("neg_y_up")
            ps.set_front_dir("neg_z_front")
            ps.set_navigation_style("free")

        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0.0, 0.0, 0.0))
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(1920, 1080)
        ps.set_give_focus_on_show(True)

        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(to_np(scene_bbox[0]), to_np(scene_bbox[1]))

        # viz stateful parameters & options
        self.viz_do_train = False
        self.viz_final = True
        self.training_done = False
        self.viz_bbox = False
        self.live_update = True  # if disabled , will skip rendering updates to accelerate background training loop
        self.viz_render_styles = ["color", "density", "distance", "hits", "normals"]
        self.viz_render_style_ind = 0
        self.viz_render_style_scale = 1.0
        self.viz_curr_render_size = None
        self.viz_curr_render_style_ind = None
        self.viz_render_color_buffer = None
        self.viz_render_scalar_buffer = None
        self.viz_render_name = "render"
        self.viz_render_enabled = True
        self.viz_render_subsample = 1
        self.viz_render_train_view = False
        self.viz_render_show_details = False
        self.render_timer = CudaTimer()
        self.render_width = 1920
        self.render_width = 1080

        if self.conf.render.method == "torch":
            self.viz_render_subsample = 4

        self.train_dataset = train_dataset
        self.model = model
        ps.init()
        self.ps_point_cloud = ps.register_point_cloud(
            "centers", to_np(model.positions), radius=1e-3, point_render_mode="quad"
        )
        self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

        # Only implemented for NeRF and Colmap dataset
        if isinstance(train_dataset, DatasetVisualization):
            train_dataset.create_dataset_camera_visualization()
        if isinstance(val_dataset, DatasetVisualization):
            val_dataset.create_dataset_camera_visualization()

        bbox_min, bbox_max = scene_bbox
        nodes = np.array(
            [
                [bbox_min[0], bbox_min[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_min[2]],
                [bbox_min[0], bbox_max[1], bbox_min[2]],
                [bbox_min[0], bbox_min[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_min[2]],
                [bbox_max[0], bbox_min[1], bbox_max[2]],
                [bbox_min[0], bbox_max[1], bbox_max[2]],
                [bbox_max[0], bbox_max[1], bbox_max[2]],
            ]
        )
        edges = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [2, 4], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]]
        )
        ps.register_curve_network("bbox", nodes, edges)

        ps.set_user_callback(self.ps_ui_callback)

        # Update once to popualte lazily-created structures
        self.update_render_view_viz(force=True)

    def update_cloud_viz(self):
        # re-initialize the viz
        if self.ps_point_cloud is None or self.ps_point_cloud.n_points() != self.model.positions.shape[0]:
            self.ps_point_cloud = ps.register_point_cloud("centers", to_np(self.model.positions))
            self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

            # always do a CPU-roundtrip update
            # TODO: add missing device update func to enable a device update here
            self.ps_point_cloud_buffer.update_data(to_np(self.model.positions))

    def render_from_current_ps_view(self):
        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.viz_render_subsample
        window_h = window_h // self.viz_render_subsample
        view_params = ps.get_view_camera_parameters()
        cam_center = view_params.get_position()
        corner_rays = view_params.generate_camera_ray_corners()
        c_ul, c_ur, c_ll, c_lr = [torch.tensor(a, device=DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays]

        # generate view camera ray origins and directions
        interp_x, interp_y = torch.meshgrid(
            torch.linspace(0.0, window_w - 1, window_w, device=DEFAULT_DEVICE, dtype=torch.float32),
            torch.linspace(0.0, window_h - 1, window_h, device=DEFAULT_DEVICE, dtype=torch.float32),
            indexing="xy",
        )
        u = interp_x
        v = interp_y

        FOCAL = fov2focal(np.deg2rad(view_params.get_fov_vertical_deg()), window_h)
        xs = ((u + 0.5) - 0.5 * window_w) / FOCAL
        ys = ((v + 0.5) - 0.5 * window_h) / FOCAL
        rays_dir = torch.nn.functional.normalize(torch.stack((xs, ys, torch.ones_like(xs)), axis=-1), dim=-1).unsqueeze(
            0
        )

        # Render a frame
        with torch.no_grad():
            W2C = view_params.get_view_mat()
            C2W = np.linalg.inv(W2C)
            C2W[:, 1:3] *= -1  # [right up back] to [right down front]
            inputs = Batch(
                intrinsics=[FOCAL, FOCAL, window_w / 2, window_h / 2],
                T_to_world=torch.FloatTensor(C2W).unsqueeze(0),
                rays_ori=torch.zeros((1, window_h, window_w, 3), device=DEFAULT_DEVICE, dtype=torch.float32),
                rays_dir=rays_dir.reshape(1, window_h, window_w, 3),
            )

            self.render_timer.start()
            outputs = self.model(inputs, train=self.viz_render_train_view)
            self.render_timer.end()
            self.render_width = window_w
            self.render_height = window_h

        return (
            outputs["pred_rgb"],
            outputs["pred_opacity"],
            outputs["pred_dist"],
            outputs["pred_normals"],
            outputs["hits_count"] / self.conf.writer.max_num_hits,
        )

    def update_render_view_viz(self, force=False):
        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.viz_render_subsample
        window_h = window_h // self.viz_render_subsample

        # re-initialize if needed
        style = self.viz_render_styles[self.viz_render_style_ind]
        if (
            force
            or self.viz_curr_render_style_ind != self.viz_render_style_ind
            or self.viz_curr_render_size != (window_w, window_h)
        ):
            self.viz_curr_render_style_ind = self.viz_render_style_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style == "color" or style == "normals":

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                    self.viz_render_name,
                    dummy_image,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )

                self.viz_render_color_buffer = ps.get_quantity_buffer(self.viz_render_name, "colors")
                self.viz_render_scalar_buffer = None

            elif style == "density":

                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                self.viz_main_image = ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 1),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

            elif style == "distance":

                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 3.0  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="jet",
                    vminmax=(0, 3),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

            elif style == "hits":

                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                self.viz_main_image = ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="jet",
                    vminmax=(0, 1),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

        # do the actual rendering
        sple_orad, sple_odns, sple_odist, sple_onrm, sple_ohit = self.render_from_current_ps_view()

        # update the data
        if style == "color":
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:, :, :, 0:1])), dim=-1)
            if self.update_from_device:
                self.viz_render_color_buffer.update_data_from_device(sple_orad.detach())
            else:
                self.viz_render_color_buffer.update_data(to_np(sple_orad))

        elif style == "density":
            if self.update_from_device:
                self.viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())
            else:
                self.viz_render_scalar_buffer.update_data(to_np(sple_odns))

        elif style == "distance":
            if self.update_from_device:
                self.viz_render_scalar_buffer.update_data_from_device(
                    (sple_odist.detach() * self.viz_render_style_scale) / torch.clamp(sple_odns, min=1e-06)
                )
            else:
                self.viz_render_scalar_buffer.update_data(
                    to_np((sple_odist * self.viz_render_style_scale) / torch.clamp(sple_odns, min=1e-06))
                )

        elif style == "hits":
            if self.update_from_device:
                self.viz_render_scalar_buffer.update_data_from_device(sple_ohit.detach())
            else:
                self.viz_render_scalar_buffer.update_data(to_np(sple_ohit))

        elif style == "normals":
            # scale in rendering space
            sple_onrm = 0.5 * (sple_onrm + 1)
            # append 1s for alpha
            sple_onrm = torch.cat((sple_onrm, torch.ones_like(sple_onrm[:, :, :, 0:1])), dim=-1)
            if self.update_from_device:
                self.viz_render_color_buffer.update_data_from_device(sple_onrm.detach())
            else:
                self.viz_render_color_buffer.update_data(to_np(sple_onrm))

    @torch.no_grad()
    def render_trajectory(self):
        global trajectory

        splprep_k = 3  # default value for splprep
        if len(trajectory) <= splprep_k:
            logger.warning("Trajectory too short to interpolate. Need at least {} points.".format(splprep_k + 1))
            return

        out_video = None
        eyes = np.stack([eye for eye, target in trajectory])
        targets = np.stack([target for eye, target in trajectory])

        def interpolate_points(points):
            from scipy.interpolate import splev, splprep

            tck, u = splprep(points.T, u=None, s=0.0, per=1, k=splprep_k)
            u_new = np.linspace(u.min(), u.max(), 60 * len(trajectory))
            return np.stack(splev(u_new, tck, der=0)).T

        eyes_new = interpolate_points(eyes)
        targets_new = interpolate_points(targets)

        output_video_filename = "output.mp4"

        for eye, target in logger.track(
            list(zip(eyes_new, targets_new)), description="Write Video", color="misty_rose1", transient=True
        ):
            ps.look_at(eye, target)
            rgb, _, _, _, _ = self.render_from_current_ps_view()

            if out_video is None:
                out_video = cv2.VideoWriter(
                    output_video_filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (rgb.shape[2], rgb.shape[1]), True
                )

            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)

        out_video.release()

        logger.info(f"Video saved to {output_video_filename}")

    def ps_ui_callback(self):
        global trajectory

        # FPS counter
        io = psim.GetIO()
        psim.TextUnformatted(f"Rolling: {1000.0 / io.Framerate:.1f} ms/frame ({io.Framerate:.1f} fps)")
        psim.TextUnformatted(f"Last: {io.DeltaTime * 1000:.1f} ms/frame ({1./io.DeltaTime:.1f} fps)")

        # Create a little ImGUI UI
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if self.training_done:
            psim.SameLine()
            psim.Text("Training Complete.")
        else:
            if psim.TreeNode("Training"):
                _, self.viz_do_train = psim.Checkbox("Train", self.viz_do_train)
                psim.SameLine()
                _, self.live_update = psim.Checkbox("Update View", self.live_update)
                psim.TreePop()

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Render"):
            psim.PushItemWidth(100)

            if psim.Button("Show"):
                self.viz_render_enabled = True
                self.update_render_view_viz(force=True)
            psim.SameLine()
            if psim.Button("Hide"):
                self.viz_render_enabled = False
                self.update_render_view_viz(force=True)

            if psim.Button("Add Cam to Trajectory"):
                view_params = ps.get_view_camera_parameters()
                cam_center = view_params.get_position()
                corner_rays = view_params.generate_camera_ray_corners()
                c_ul, c_ur, c_ll, c_lr = [
                    torch.tensor(a, device=DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays
                ]
                c_mid = (c_ul + c_ur + c_ll + c_lr) / 4
                trajectory.append((cam_center, cam_center + c_mid.cpu().numpy()))
            psim.Text(f"There are {len(trajectory)} cameras in the trajectory.")
            if psim.Button("Reset Trajectory"):
                trajectory = []
            if psim.Button("Render Trajectory"):
                self.render_trajectory()

            _, self.viz_render_show_details = psim.Checkbox("Infos", self.viz_render_show_details)
            if self.viz_render_show_details:
                psim.SameLine()
                psim.Text(f"({self.render_width}x{self.render_height}) @ {self.render_timer.timing()} ms.")

            _, self.viz_render_style_ind = psim.Combo("Style", self.viz_render_style_ind, self.viz_render_styles)
            if self.viz_render_styles[self.viz_render_style_ind] == "distance":
                psim.SameLine()
                _, self.viz_render_style_scale = psim.InputFloat("scale", self.viz_render_style_scale, 0.01)

            changed, self.viz_render_subsample = psim.InputInt("Subsample Factor", self.viz_render_subsample, 1)
            if changed:
                self.viz_render_subsample = max(self.viz_render_subsample, 1)

            _, self.viz_render_train_view = psim.Checkbox("render w/ train=True", self.viz_render_train_view)

        if self.live_update:
            self.update_render_view_viz()

    def block_in_rendering_loop(self, fps=60):
        self.live_update = True
        render_every_sec = 1.0 / float(fps)
        while not ps.window_requests_close():
            ps.frame_tick()
            last_render_time = time.time()
            while time.time() - last_render_time <= render_every_sec:
                time.sleep(1 / fps)
