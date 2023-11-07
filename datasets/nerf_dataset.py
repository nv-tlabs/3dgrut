import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from tqdm import tqdm
from datasets.nerf_utils import get_ray_directions, read_image, get_rays
from PIL import Image


class NeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', sample_full_image=False, batch_size=8192,
                 use_white_background=False, **kwargs):
        print(root_dir)
        self.root_dir = root_dir 
        self.split = split
        
        self.sample_full_image = sample_full_image
        self.batch_size = batch_size

        # True: Transparent pixels with non-saturated alpha values will be blended with white
        # False: Transparent pixels with non-saturated alpha values will be blended with black
        self.use_white_background = use_white_background

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)
    
        self.length_scale, self.center, self.scene_bbox = self.compute_spatial_extents()

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

            # determine scale
            if 'Jrender_Dataset' in self.root_dir:
                c2w[:, :2] *= -1 # [left up front] to [right down front]
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene=='Easyship':
                    pose_radius_scale = 1.2
                elif scene=='Scar':
                    pose_radius_scale = 1.8
                elif scene=='Coffee':
                    pose_radius_scale = 2.5
                elif scene=='Car':
                    pose_radius_scale = 0.8
                else:
                    pose_radius_scale = 1.5
            else:
                c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                pose_radius_scale = 1.5
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale

            # add shift
            if 'Jrender_Dataset' in self.root_dir:
                if scene=='Coffee':
                    c2w[1, 3] -= 0.4465
                elif scene=='Car':
                    c2w[0, 3] -= 0.7
            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = read_image(img_path, self.img_wh, blend_a=self.use_white_background)
                self.rgbs += [img]
            except: pass

        if len(self.rgbs)>0:
            self.rgbs = torch.FloatTensor(np.stack(self.rgbs)) # (N_images, hw, ?)
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
    
    def get_bbox(self):
        """Tuple of vec3 (min,max)"""
        return self.scene_bbox

    def __len__(self):
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

            return rays_o.reshape(out_shape),rays_d.reshape(out_shape),rgb.reshape(out_shape)

        elif self.split == 'val':
            assert self.sample_full_image, 'val mode assumes sampling full images'
            if len(self.rgbs)>0: # if ground truth available
                rgb = self.rgbs[idx]
                pose =self.poses[idx]
                directions = self.directions[np.arange(self.img_wh[0]*self.img_wh[1])]
                rays_o, rays_d = get_rays(directions, pose)

                return rays_o.reshape(self.image_h,self.image_w,3), rays_d.reshape(self.image_h,self.image_w,3), rgb.reshape(self.image_h,self.image_w,3)

    @property
    def image_h(self):
        return self.img_wh[1]

    @property
    def image_w(self):
        return self.img_wh[0]
