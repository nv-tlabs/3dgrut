import numpy as np
import torch
import polyscope as ps
import polyscope.imgui as psim
from utils import to_np
from model import MixtureOfGaussians

import hydra
from omegaconf import DictConfig, OmegaConf

class Playground:

    DEFAULT_DEVICE = torch.device('cuda')

    def __init__(self, conf):
        self.conf = conf
        self.object_paths = conf.playground.objects
        objects = self.load_objects(self.object_paths)
        self.scene_mog = self.mash_into_scene(objects)
        self.init_polyscope()
        self.is_running = True

    def load_objects(self, object_paths):
        objects =  []
        for single_object_path in object_paths:
            checkpoint = torch.load(single_object_path)
            obj_conf = checkpoint["config"]
            model = MixtureOfGaussians(obj_conf)
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            objects.append(model)
        return objects

    def mash_into_scene(self, objects):

        scene_mog = MixtureOfGaussians(self.conf).to(self.DEFAULT_DEVICE)

        # TODO(operel): Validate that all objects use the same render / activations conf
        for idx, o in enumerate(objects):
            # TODO(operel): Scatter objects with a more interesting pattern
            jitter = 0.5 * torch.rand(3, device=self.DEFAULT_DEVICE)
            jitter[0] += 0.5 if idx % 2 == 0 else -0.5
            jitter[1] += -0.5 if idx % 3 == 0 else 0.5
            jitter[2] = 0.0  # Put them at the same level
            jittered_positions = jitter[None].expand_as(o.positions)
            scene_mog.positions = torch.nn.Parameter(torch.cat((scene_mog.positions, o.positions + jittered_positions), dim=0))
            scene_mog.scale = torch.nn.Parameter(torch.cat((scene_mog.scale, o.scale), dim=0))
            scene_mog.rotation = torch.nn.Parameter(torch.cat((scene_mog.rotation, o.rotation), dim=0))
            scene_mog.density = torch.nn.Parameter(torch.cat((scene_mog.density, o.density), dim=0))
            scene_mog.features_albedo = torch.nn.Parameter(torch.cat((scene_mog.features_albedo, o.features_albedo), dim=0))

            if scene_mog.features_specular.shape[1] < o.features_specular.shape[1]:
                missing_sh_dims = o.features_specular.shape[1] - scene_mog.features_specular.shape[1]
                num_gaussians = scene_mog.features_specular.shape[0]
                padding = torch.zeros(num_gaussians, missing_sh_dims, device=self.DEFAULT_DEVICE)
                scene_mog.features_specular = torch.nn.Parameter(torch.cat((scene_mog.features_specular, padding), dim=1))
            scene_mog.features_specular = torch.nn.Parameter(torch.cat((scene_mog.features_specular, o.features_specular), dim=0))
        scene_mog.set_optix_context()
        scene_mog.build_bvh()
        return scene_mog

    def run(self):
        while self.is_running:
            ps.frame_tick()

    def init_polyscope(self):
        ps.set_use_prefs_file(False)

        ps.set_up_dir("z_up")
        ps.set_front_dir("neg_y_front")
        ps.set_navigation_style("free")

        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0., 0., 0.))
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        if self.conf.render.method == 'torch':
            ps.set_window_size(1280, 720)
        else:
            ps.set_window_size(1920, 1080)
        ps.set_give_focus_on_show(True)

        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]))

        self.viz_do_train = False
        self.viz_bbox = False
        self.live_update = True  # if disabled , will skip rendering updates to accelerate background training loop
        self.viz_render_styles = ['color', 'density']
        self.viz_render_style_ind = 0
        self.viz_curr_render_size = None
        self.viz_curr_render_style_ind = None
        self.viz_render_color_buffer = None
        self.viz_render_scalar_buffer = None
        self.viz_render_name = 'render'
        self.viz_render_enabled = True
        self.viz_render_subsample = 1

        ps.init()
        ps.set_user_callback(self.ps_ui_callback)

        # Update once to popualte lazily-created structures
        self.update_render_view_viz(force=True)

    @torch.no_grad()
    def render(self, rays_ori, rays_dir):
        return self.scene_mog(rays_ori, rays_dir)

    @torch.no_grad()
    def render_from_current_ps_view(self):
        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.viz_render_subsample
        window_h = window_h // self.viz_render_subsample
        view_params = ps.get_view_camera_parameters()
        cam_center = view_params.get_position()
        corner_rays = view_params.generate_camera_ray_corners()
        c_ul, c_ur, c_ll, c_lr = [torch.tensor(a, device=self.DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays]

        # generate view camera ray origins and directions
        rays_ori = torch.tensor(cam_center, device=self.DEFAULT_DEVICE, dtype=torch.float32).reshape(1,1,1,3).expand(1,window_h,window_w,3)
        interp_x, interp_y= torch.meshgrid(
                        torch.linspace(0., 1., window_w, device=self.DEFAULT_DEVICE, dtype=torch.float32),
                        torch.linspace(0., 1., window_h, device=self.DEFAULT_DEVICE, dtype=torch.float32),
                        indexing='xy')
        interp_x = interp_x.unsqueeze(-1)
        interp_y = interp_y.unsqueeze(-1)
        rays_dir = c_ul + interp_x * (c_ur - c_ul) + interp_y * (c_ll - c_ul)
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
        rays_dir = rays_dir.unsqueeze(0)

        # Render a frame
        outputs = self.render(rays_ori, rays_dir)

        return outputs['pred_rgb'], outputs['pred_opacity'], outputs['pred_dist']

    @torch.no_grad()
    def update_render_view_viz(self, force=False):

        window_w, window_h = ps.get_window_size()
        window_w = window_w // self.viz_render_subsample
        window_h = window_h // self.viz_render_subsample

        # re-initialize if needed
        style = self.viz_render_styles[self.viz_render_style_ind]
        if force or self.viz_curr_render_style_ind != self.viz_render_style_ind or self.viz_curr_render_size != (
        window_w, window_h):
            self.viz_curr_render_style_ind = self.viz_render_style_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style == "color":

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

        # do the actual rendering
        sple_orad, sple_odns, sple_odist = self.render_from_current_ps_view()

        # update the data
        if style == "color":
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:, :, :, 0:1])), dim=-1)
            self.viz_render_color_buffer.update_data_from_device(sple_orad.detach())

        elif style == "density":
            self.viz_render_scalar_buffer.update_data_from_device(sple_odns.detach())

    def ps_ui_callback(self):
        # Create a little ImGUI UI
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Render"):
            psim.PushItemWidth(100)

            if (psim.Button("Show")):
                self.viz_render_enabled = True
                self.update_render_view_viz(force=True)
            psim.SameLine()
            if (psim.Button("Hide")):
                self.viz_render_enabled = False
                self.update_render_view_viz(force=True)

            _, self.viz_render_style_ind = psim.Combo("Style", self.viz_render_style_ind, self.viz_render_styles)
            
            changed, self.viz_render_subsample = psim.InputInt("Subsample Factor", self.viz_render_subsample, 1)
            if changed:
                self.viz_render_subsample = max(self.viz_render_subsample, 1)
            
            psim.PopItemWidth()
            psim.TreePop()

        if self.live_update:
            self.update_render_view_viz()

@hydra.main(config_path="configs", config_name='base', version_base=None)
def main(conf: DictConfig) -> None:
    playground = Playground(conf)
    playground.run()

if __name__ == "__main__":
    main()
