import os
import sys
import time
import numpy as np
import torch
try:
    import viser
    import viser.transforms as tf
except ImportError:
    print('viser not installed, please run "pip install viser"')
    sys.exit(1)
import cv2
import argparse
import kaolin
from collections import deque
from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes
from kaolin.render.camera import Camera
from threedgrut.utils.misc import quaternion_to_so3

# Below code referenced from viser https://github.com/nerfstudio-project/viser
# and viser 3dgs example: https://github.com/WangFeng18/3d-gaussian-splatting/blob/main/visergui.py

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    # camera.wxyz: (4,) numpy, quaternion (w, x, y, z)
    # quaternion_to_so3 expects (N,4) torch, so convert
    q = np.asarray(camera.wxyz)[None, :]
    q_torch = torch.from_numpy(q).float()
    R = quaternion_to_so3(q_torch)[0].cpu().numpy()
    c2w[:3, :3] = R
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


class ViserViewer:
    def __init__(self, viewer_ip_port, engine, target_fps=20.0):
        self.engine = engine
        self.set_initial_mesh()
        self.port = viewer_ip_port
        self.target_fps = target_fps
        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.gui.add_button("Reset View")
        self.need_update = False
        self.resolution_slider = self.server.gui.add_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.gui.add_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.gui.add_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )
        self.fps = self.server.gui.add_text("FPS", initial_value="-1", disabled=True)
        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True
        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True
        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True
        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])
        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

    def set_initial_mesh(self):
        """ Current implementation sets an empty scene with no meshes """
        # The 3dgrut playground engine is loaded with a sample mesh primitive (glass sphere),
        # remove the default mesh primitives from the engine here
        for mesh_name in list(self.engine.primitives.objects.keys()):
            self.engine.primitives.remove_primitive(mesh_name)

    def fast_render(self, in_cam, is_first_pass=False):
        # Called during interactions, disables effects for quick rendering
        framebuffer = self.engine.render_pass(in_cam, is_first_pass=is_first_pass)
        rgba_buffer = torch.cat([framebuffer['rgb'], framebuffer['opacity']], dim=-1)
        rgba_buffer = torch.clamp(rgba_buffer, 0.0, 1.0)
        img = (rgba_buffer[0, :, :, :3] * 255).to(torch.uint8)  # [H, W, 3], RGB
        img_np = img.cpu().numpy()
        return img_np

    # refrenced https://github.com/nv-tlabs/3dgrut/blob/main/threedgrut_playground/ps_gui.py#L132
    @torch.no_grad()
    def update(self):
        if self.need_update:
            interval = 0
            for client in self.server.get_clients().values():
                camera = client.camera
                try:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value/camera.aspect)
                    view_matrix = get_c2w(client.camera)
                    fov_y = client.camera.fov
                    width, height = W, H
                    near, far = self.near_plane_slider.value, self.far_plane_slider.value
                    kaolin_camera = Camera.from_args(
                            view_matrix=view_matrix,
                            fov=fov_y,
                            width=width, height=height,
                            near=near, far=far,
                            dtype=torch.float32,
                            device=self.engine.device
                        )
                    
                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()
                    out = self.fast_render(in_cam=kaolin_camera, is_first_pass=True)
                    
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.
                    
                except RuntimeError as e:
                    print(e)
                    interval = 1
                    continue
                client.scene.set_background_image(out, format="jpeg")
            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            self.need_update = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gs_object',
        type=str,
        required=True,
        help="Path of pretrained 3dgrt checkpoint, as .pt / .ingp / .ply file."
    )
    parser.add_argument(
        '--mesh_assets',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'assets'),
        help="Path to folder containing mesh assets of .obj or .glb format."
    )
    parser.add_argument(
        '--default_gs_config',
        type=str,
        default='apps/colmap_3dgrt.yaml',
        help="Name of default config to use for .ingp, .ply files, or .pt files not trained with 3dgrt."
    )
    parser.add_argument(
        '--envmap_assets',
        type=str,
        default=os.path.join(os.path.dirname(__file__), 'assets'),
        help="Optional path to folder containing .hdr environment maps to use for lighting mesh assets."
    )
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--target_fps', type=float, default=20.0)
    args = parser.parse_args()

    engine = Engine3DGRUT(
        gs_object=args.gs_object,
        mesh_assets_folder=args.mesh_assets,
        envmap_assets_folder=args.envmap_assets,
        default_config=args.default_gs_config,
    )
    viewer = ViserViewer(viewer_ip_port=args.port, engine=engine, target_fps=args.target_fps)
    last_time = time.time()
    while True:
        start = time.time()
        viewer.update()
        # Dynamic FPS Control
        elapsed = time.time() - start
        sleep_time = max(0, (1.0 / args.target_fps) - elapsed)
        time.sleep(sleep_time)
