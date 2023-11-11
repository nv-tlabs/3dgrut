import torch 

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from utils import to_np
from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset


DEFAULT_DEVICE = torch.device('cuda')

class GUI:
    def __init__(self, conf, model, train_dataset, val_dataset):
        ps.set_use_prefs_file(False)

        if conf.dataset.type == 'nerf': # NeRF synthetic uses the blender coordinate-system
            ps.set_up_dir("z_up")
            ps.set_front_dir("neg_y_front")
            ps.set_navigation_style("turntable")
        elif conf.dataset.type == 'colmap': # Colmap scenes use a cartesian coordinate-system
            ps.set_up_dir("neg_y_up")
            ps.set_front_dir("neg_z_front")
            ps.set_navigation_style("free")
        else:                              # AV use cartesian coordinate-system with z-up
            ps.set_up_dir("z_up")
            ps.set_front_dir("x_front")
            ps.set_navigation_style("free")

        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0., 0., 0.))
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(1920, 1080)
        ps.set_give_focus_on_show(True)

        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(to_np(train_dataset.scene_bbox[0]), to_np(train_dataset.scene_bbox[1]))

        # viz stateful parameters & options
        self.viz_do_train = False
        self.viz_bbox = False
        self.viz_skip_update = False  # if enabled, will skip rendering updates to accelerate background training loop
        self.viz_render_styles = ['color', 'density']
        self.viz_render_style_ind = 0
        self.viz_curr_render_size = None
        self.viz_curr_render_style_ind = None
        self.viz_render_color_buffer = None
        self.viz_render_scalar_buffer = None
        self.viz_render_name = 'render'

        self.train_dataset = train_dataset
        self.model = model
        ps.init()
        self.ps_point_cloud = ps.register_point_cloud("centers", to_np(model.get_positions()), 
                                radius=1e-3, point_render_mode='quad')
        self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

        # Only implemented for NeRF and Colmap dataset
        if isinstance(train_dataset, (NeRFDataset, ColmapDataset)):
            train_dataset.create_dataset_camera_visualization()
            val_dataset.create_dataset_camera_visualization()

        ps.set_user_callback(self.ps_ui_callback)

    def update_cloud_viz(self):

        # re-initialize the viz
        if self.ps_point_cloud is None or self.ps_point_cloud.n_points() != self.model.get_positions().shape[0]:
            self.ps_point_cloud = ps.register_point_cloud("centers", to_np(self.model.get_positions()))
            self.ps_point_cloud_buffer = self.ps_point_cloud.get_buffer("points")

        # direct on-GPU update, must not have changed size
        self.ps_point_cloud_buffer.update_data_from_device(self.model.get_positions().detach())

    def render_from_current_ps_view(self):

        window_w, window_h = ps.get_window_size()
        view_params = ps.get_view_camera_parameters()
        cam_center = view_params.get_position()
        corner_rays = view_params.generate_camera_ray_corners()
        c_ul, c_ur, c_ll, c_lr = [torch.tensor(a, device=DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays]

        # generate view camera ray origins and directions
        rays_ori = torch.tensor(cam_center, device=DEFAULT_DEVICE, dtype=torch.float32).reshape(1,1,1,3).expand(1,window_h,window_w,3)
        interp_x, interp_y= torch.meshgrid(
                        torch.linspace(0., 1., window_w, device=DEFAULT_DEVICE, dtype=torch.float32),
                        torch.linspace(0., 1., window_h, device=DEFAULT_DEVICE, dtype=torch.float32),
                        indexing='xy')
        interp_x = interp_x.unsqueeze(-1)
        interp_y = interp_y.unsqueeze(-1)
        rays_dir = c_ul + interp_x * (c_ur - c_ul) + interp_y * (c_ll - c_ul)
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
        rays_dir = rays_dir.unsqueeze(0)

        # Render a frame
        with torch.no_grad():
            outputs = self.model(rays_ori, rays_dir)

        return outputs['pred_rgb'], outputs['pred_opacity'], outputs['pred_ohit']

    def update_render_view_viz(self):

        window_w, window_h = ps.get_window_size()

        # re-initialize if needed
        style = self.viz_render_styles[self.viz_render_style_ind]
        if  self.viz_curr_render_style_ind !=  self.viz_render_style_ind or  self.viz_curr_render_size != (window_w, window_h):
            self.viz_curr_render_style_ind = self.viz_render_style_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style == "color":

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                     self.viz_render_name,
                    dummy_image,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )

                self.viz_render_color_buffer = ps.get_quantity_buffer(self.viz_render_name, "colors")
                self.viz_render_scalar_buffer = None
            
            elif style == "density":
            
                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=True,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 1),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")


        if self.viz_bbox: 
            bbox_min, bbox_max = self.train_dataset.scene_bbox
            nodes = np.array([[bbox_min[0], bbox_min[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_min[2]], [bbox_min[0], bbox_max[1], bbox_min[2]], 
                                [bbox_min[0], bbox_min[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_min[2]], [bbox_max[0], bbox_min[1], bbox_max[2]], 
                                [bbox_min[0], bbox_max[1], bbox_max[2]], [bbox_max[0], bbox_max[1], bbox_max[2]]])
            edges = np.array(
                [[0, 1], [0, 2], [0, 3], [1, 4], [1, 5], [2, 6], [2, 4], [3, 5], [3, 6], [4, 7], [5, 7], [6, 7]])
            ps.register_curve_network("bbox", nodes, edges)

        # do the actual rendering
        sple_orad, sple_odns, sple_ohit = self.render_from_current_ps_view()

        # update the data
        if style == "color":
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:,:,:,0:1])), dim=-1)
            self.viz_render_color_buffer.update_data_from_device(sple_orad.detach())

        elif style == "density":
            self.viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())


    def ps_ui_callback(self):
        # Create a little ImGUI UI
        _, self.viz_do_train = psim.Checkbox("Train", self.viz_do_train)
        _, self.viz_skip_update = psim.Checkbox("Skip Render Update", self.viz_skip_update)
        if isinstance(self.train_dataset, (NeRFDataset, ColmapDataset)):
            _, self.viz_bbox = psim.Checkbox("Visualize the bbox", self.viz_bbox)
        _, self.viz_render_style_ind = psim.Combo("Render Display", self.viz_render_style_ind, self.viz_render_styles)

        if not self.viz_skip_update:
            self.update_render_view_viz()


