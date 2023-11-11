import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

from datasets.nerf_utils import get_ray_directions, read_image, get_rays, create_camera_visualization
from utils import to_np


class NeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', sample_full_image=False, batch_size=8192, return_alphas=False, **kwargs):
        self.root_dir = root_dir 
        self.split = split
        
        self.sample_full_image = sample_full_image
        self.batch_size = batch_size

        self.return_alphas = return_alphas

        self.read_intrinsics()
        self.read_meta(split)
    
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
            meta = json.load(f)

        # !! Assumptions !!
        # 1. All images have the same intrinsics
        # 2. Principal point is at canvas center
        # 3. Camera has no distortion params
        first_frame_path = meta['frames'][0]['file_path']
        frame = Image.open(os.path.join(self.root_dir,first_frame_path + '.png'))
        w = frame.width
        h = frame.height
        fx = fy = 0.5*w/np.tan(0.5*meta['camera_angle_x'])

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rgbs = []
        self.poses = []
        if self.return_alphas:
            self.alphas = []

        if split == 'trainval':
            with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
                frames = json.load(f)["frames"]
            with open(os.path.join(self.root_dir, "transforms_val.json"), 'r') as f:
                frames+= json.load(f)["frames"]
        else:
            with open(os.path.join(self.root_dir, f"transforms_{split}.json"), 'r') as f:
                frames = json.load(f)["frames"]

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]
            c2w[:, 1:3] *= -1 # [right up back] to [right down front]
            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                if self.return_alphas:
                    img, alpha = read_image(img_path, self.img_wh, return_alpha=True)
                    self.alphas.append(alpha)
                else:
                    img = read_image(img_path, self.img_wh, return_alpha=False)


                self.rgbs += [img]
            except: pass

        if len(self.rgbs)>0:
            self.rgbs = torch.FloatTensor(np.stack(self.rgbs)) # (N_images, hw, ?)
        if self.return_alphas and len(self.alphas)>0:
            self.alphas = torch.FloatTensor(np.stack(self.alphas))
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

    def compute_spatial_extents(self):

        with torch.no_grad():

            # mean distance between of cameras from center
            camera_origins = self.poses[:,:,3]
            center = camera_origins.mean(dim=0)
            dists = torch.linalg.norm(camera_origins - center[None,:], dim=-1)
            mean_dist = torch.mean(dists)
            bbox_min = torch.min(camera_origins, dim=0).values
            bbox_max = torch.max(camera_origins, dim=0).values

            return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale
    
    def get_center(self):
        return self.center
    
    def get_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Tuple of vec3 (min,max)"""
        return self.scene_bbox

    def __len__(self):
        if self.split == 'train': 
            return len(self.poses) if self.sample_full_image else 1000
        else:
            return len(self.poses)

    def __getitem__(self, idx):
        if self.split == 'train':          
            # randomly select pixels
            if self.sample_full_image:
                img_idxs = idx
                pix_idxs = np.arange(self.img_wh[0]*self.img_wh[1])
                out_shape = (self.image_h,self.image_w,3)
            else:
                img_idxs = np.random.choice(len(self.poses), self.batch_size, replace=True)
                pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                out_shape = (1,self.batch_size,3)
            rgb = self.rgbs[img_idxs, pix_idxs]
            poses = self.poses[img_idxs]
            directions = self.directions[pix_idxs]
            rays_o, rays_d = get_rays(directions, poses)

            sample = {
                "rays_ori":rays_o.reshape(out_shape), 
                "rays_dir":rays_d.reshape(out_shape), 
                "rgb_gt":rgb.reshape(out_shape)
            }

            if self.return_alphas:
                sample["alpha"] = self.alphas[img_idxs,pix_idxs].reshape(out_shape[0],out_shape[1],1)
            return sample
        
        elif self.split == 'val' or self.split == "test":
            assert self.sample_full_image, 'val mode assumes sampling full images'
            if len(self.rgbs)>0: # if ground truth available
                rgb = self.rgbs[idx]
                pose = self.poses[idx]
                directions = self.directions[np.arange(self.img_wh[0]*self.img_wh[1])]
                rays_o, rays_d = get_rays(directions, pose)


                sample = {
                    "rays_ori": rays_o.reshape(self.image_h,self.image_w,3),  
                    "rays_dir": rays_d.reshape(self.image_h,self.image_w,3),
                    "rgb_gt": rgb.reshape(self.image_h,self.image_w,3)
                }


                if self.return_alphas:
                    sample["alpha"] = self.alphas[idx].reshape(self.image_h,self.image_w,1)
                return sample
        
    @property
    def image_h(self):
        return self.img_wh[1]

    @property
    def image_w(self):
        return self.img_wh[0]

    def create_dataset_camera_visualization(self):

        # just one global intrinsic mat for now
        intrinsics = to_np(self.K)

        cam_list = []

        for i_cam, pose in enumerate(self.poses):

            trans_mat = np.eye(4)
            trans_mat[:3,:4] = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # these cameras follow the opposite convention from polyscope
            camera_convention_rot = np.array([[1., 0., 0., 0.,],
                                              [0., -1., 0., 0.,],
                                              [0., 0.,-1., 0.,],
                                              [0., 0., 0., 1.,]])
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            w = self.image_w
            h = self.image_h
            f_w = intrinsics[0,0]
            f_h = intrinsics[1,1]

            fov_w = 2. * np.arctan(0.5 * w / f_w)
            fov_h = 2. * np.arctan(0.5 * h / f_h)
            
            rgb = to_np(self.rgbs[i_cam,:]).reshape(h,w,3)

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
