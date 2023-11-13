import os

import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset

from datasets.colmap_utils import read_extrinsics_binary, read_intrinsics_binary, read_extrinsics_text, read_intrinsics_text,qvec2rotmat
from datasets.nerf_utils import create_camera_visualization
from datasets.utils import pinhole_camera_rays, camera_to_world_rays, get_center_and_diag
from utils import to_torch, to_np

class ColmapDataset(Dataset):

    def __init__(self, path, split='train', sample_full_image=False, batch_size=8192, downsample_factor=1):
        self.path = path
        self.split = split
        self.llff_test_split = 8
        self.sample_full_image = sample_full_image
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor

        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
        self.cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        self.cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

        # # Get the scene data
        self.get_scene_info()
        self.load_camera_data()

        temp_idx = np.arange(self.n_frames)
        if split == 'train':
            temp_idx = np.mod(temp_idx, self.llff_test_split) != 0
        else:
            temp_idx = np.mod(temp_idx, self.llff_test_split) == 0

        self.poses = self.poses[temp_idx]
        self.rgb = self.rgb[temp_idx]
        self.intrinsic = self.intrinsic[temp_idx]

        self.length_scale, self.center, self.scene_bbox = self.compute_spatial_extents()

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f'_{self.downsample_factor}'
        return f"images{downsample_suffix}"

    def get_scene_info(self):
        self.image_h = 0
        self.image_w = 0
        self.n_frames = len(self.cam_extrinsics)

        image_path = os.path.join(self.path, self.get_images_folder(), os.path.basename(self.cam_extrinsics[0].name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = np.asarray(Image.open(image_path))
        self.image_h = image.shape[0]
        self.image_w = image.shape[1]
        self.scaling_factor =  int(round(self.cam_intrinsics[self.cam_extrinsics[0].camera_id -1].height / self.image_h))

    def load_camera_data(self):
        self.poses = []
        self.intrinsic = []
        self.rgb = []
        self.indices_matrix = []

        cam_centers = []
        for idx, extr in enumerate(self.cam_extrinsics):
            intr = self.cam_intrinsics[extr.camera_id - 1]
            
            height = intr.height
            width = intr.width

            assert abs(height / self.scaling_factor - self.image_h) <= 1
            assert abs(width / self.scaling_factor - self.image_w) <= 1

            uid = intr.id
            R = qvec2rotmat(extr.qvec)
            T = np.array(extr.tvec)

            W2C = np.zeros((4,4),dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3:4])

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                self.intrinsic.append([focal_length_x / self.scaling_factor, focal_length_x / self.scaling_factor])
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                self.intrinsic.append([focal_length_x / self.scaling_factor, focal_length_y / self.scaling_factor])
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(self.path, self.get_images_folder(), os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            self.rgb.append(np.asarray(Image.open(image_path)).astype(np.float32) / 255.0 )

        # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/__init__.py#L69
        _, diagonal = get_center_and_diag(cam_centers)
        self.cameras_extent = diagonal * 1.1

        self.camera_centers = np.array(cam_centers)[:,:,0]
        self.rgb = torch.FloatTensor(np.stack(self.rgb))
        self.poses = np.stack(self.poses)
        self.intrinsic = np.stack(self.intrinsic)
    
    def compute_spatial_extents(self):

        with torch.no_grad():

            # mean distance between of cameras from center
            camera_origins = self.poses[:,:,3]
            center = np.mean(camera_origins,axis=0)
            dists = np.linalg.norm(camera_origins - center[None,:], axis=-1)
            mean_dist = np.mean(dists)
            bbox_min = np.min(camera_origins, axis=0)
            bbox_max = np.max(camera_origins, axis=0)

            center = torch.FloatTensor(center)
            mean_dist = torch.FloatTensor([mean_dist])
            bbox_min = torch.FloatTensor(bbox_min)
            bbox_max = torch.FloatTensor(bbox_max)

            return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale
    
    def get_center(self):
        return self.center
    
    def get_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tuple of vec3 (min,max)"""
        return self.scene_bbox

    def get_observer_points(self):
        return self.camera_centers

    def __len__(self) -> int:
        if self.split == 'train': 
            return self.n_frames if self.sample_full_image else 1000
        else:
            return self.n_frames
            
    def __getitem__(self, idx):
        if self.split == 'train':
            if self.sample_full_image:
                # Sample specific frame
                frame_idx = np.array((idx,))
                u = np.tile(np.arange(self.image_w), self.image_h)
                v = np.arange(self.image_h).repeat(self.image_w)
                out_shape = (self.image_h,self.image_w,3)

            else:
                # Sample random frame
                frame_idx = np.random.choice(self.n_frames, self.batch_size, replace=True)
                u = np.random.choice(self.image_w, self.batch_size, replace=True)
                v = np.random.choice(self.image_h, self.batch_size, replace=True)
                out_shape = (1,self.batch_size,3)

            rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, self.intrinsic[frame_idx,0], self.intrinsic[frame_idx,1], self.image_w, self.image_h)

            rays_o, rays_d = camera_to_world_rays(rays_o_cam, rays_d_cam, self.poses[frame_idx])

            rgb = self.rgb[frame_idx, v, u]

            rays_o = torch.FloatTensor(rays_o)
            rays_d = torch.FloatTensor(rays_d)

            sample = {
                "rays_ori":rays_o.reshape(out_shape), 
                "rays_dir":rays_d.reshape(out_shape), 
                "rgb_gt":rgb.reshape(out_shape)
            }
            return sample
        
        elif self.split == 'val':
            frame_idx = np.array((idx,))
            u = np.tile(np.arange(self.image_w), self.image_h)
            v = np.arange(self.image_h).repeat(self.image_w)

            rays_o_cam, rays_d_cam = pinhole_camera_rays(u, v, self.intrinsic[frame_idx,0], self.intrinsic[frame_idx,1], self.image_w, self.image_h)

            rays_o, rays_d = camera_to_world_rays(rays_o_cam, rays_d_cam, self.poses[frame_idx])
            rgb = self.rgb[frame_idx, v, u]

            rays_o = torch.FloatTensor(rays_o)
            rays_d = torch.FloatTensor(rays_d)
            
            assert self.sample_full_image, 'val mode assumes sampling full images'

            sample = {
                "rays_ori": rays_o.reshape(self.image_h,self.image_w,3),  
                "rays_dir": rays_d.reshape(self.image_h,self.image_w,3),
                "rgb_gt": rgb.reshape(self.image_h,self.image_w,3)
            }
            return sample
    

    def create_dataset_camera_visualization(self):

        # just one global intrinsic mat for now

        cam_list = []

        for i_cam, pose in enumerate(self.poses):
        
            trans_mat = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # these cameras follow the opposite convention from polyscope
            camera_convention_rot = np.array([[1., 0., 0., 0.,],
                                              [0., -1., 0., 0.,],
                                              [0., 0.,-1., 0.,],
                                              [0., 0., 0., 1.,]])
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            w = self.image_w
            h = self.image_h
            f_w = self.intrinsic[i_cam,0]
            f_h = self.intrinsic[i_cam,1]

            fov_w = 2. * np.arctan(0.5 * w / f_w)
            fov_h = 2. * np.arctan(0.5 * h / f_h)
            
            rgb = to_np(self.rgb[i_cam,:]).reshape(h,w,3)

            cam_list.append({
                'ext_mat' : trans_mat_world_to_camera,
                'w' : w,
                'h' : h,
                'fov_w' : fov_w,
                'fov_h' : fov_h,
                'rgb_img' : rgb,
                'split' : self.split,
            })

        create_camera_visualization(cam_list) 
