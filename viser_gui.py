from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
import cv2
from collections import deque

import numpy as np
import torch
import torchvision.transforms.functional as F
import kaolin

#Below referenced threedgrut_playground code
from threedgrut.utils.logger import logger
from threedgrut.gui.ps_extension import initialize_cugl_interop
from threedgrut_playground.utils.video_out import VideoRecorder
from threedgrut_playground.engine import Engine3DGRUT, OptixPrimitiveTypes

gs_object = "/workspace/runs/lego-3004_092246/ckpt_last.pt"
# gs_object = "/workspace/runs/flowers-0504_030839/ours_7000/ckpt_7000.pt"
mesh_assets_folder = "./threedgrut_playground/assets"
# default_config = "apps/colmap_3dgut.yaml"
default_config = "apps/colmap_3dgrt.yaml"

engine = Engine3DGRUT(
    gs_object=gs_object,
    mesh_assets_folder=mesh_assets_folder,
    default_config=default_config
)

# Below code referenced viser https://github.com/nerfstudio-project/viser
# and viser 3dgs example: https://github.com/WangFeng18/3d-gaussian-splatting/blob/main/visergui.py

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class ViserViewer:
    def __init__(self, viewer_ip_port):
        self.engine = engine
        self.port = viewer_ip_port

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
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def fast_render(self, in_cam, is_first_pass=False):
        # Called during interactions, disables effects for quick rendering
        framebuffer = self.engine.render_pass(in_cam, is_first_pass=is_first_pass)
        rgba_buffer = torch.cat([framebuffer['rgb'], framebuffer['opacity']], dim=-1)
        rgba_buffer = torch.clamp(rgba_buffer, 0.0, 1.0)
        # return (rgba_buffer[0] * 255).to(torch.uint8)
        img = (rgba_buffer[0, :, :, :3] * 255).to(torch.uint8)  # [H, W, 3], RGB
        img_np = img.cpu().numpy()
        # img_np = img_np[..., ::-1]  # RGB to BGR
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
                    from kaolin.render.camera import Camera
                    view_matrix = get_c2w(client.camera)
                    # view_matrix = get_w2c(client.camera)
                    fov_y = client.camera.fov
                    # fov_y = client.camera.fov * np.pi / 180
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
                    
                    # kaolin_camera.change_coordinate_system(
                    #     torch.tensor([[1, 0, 0],
                    #                 [0, 0, 1],
                    #                 [0, -1, 0]]
                    # ))
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
                self.debug_idx += 1
            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"

if __name__ == "__main__":
    viewer = ViserViewer(viewer_ip_port=8080)
    while True:
        viewer.update()
        time.sleep(0.05)