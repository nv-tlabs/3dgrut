from typing import Optional, Union, List, Tuple

import glob
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image as PILImage
import torch
from torch.utils.data import Dataset

# C++ utils
import ray_utils  # type: ignore

from datasets.utils import fov2focal
from datasets.ngp_utils import load_pc_dat, nerf_ray_to_colmap, nerf_matrix_to_colmap
from datasets.utils import PointCloud
from utils import to_torch


class NGPDataset(Dataset):
    def __init__(self, root_dir, split="train", batch_size=8192, use_lidar=False, sample_full_image=False, val_downsample=1, max_dist_m=150, use_dynamic_masks=False, val_frame_subsample=5, use_aux=False):
        super().__init__()

        # store relevant parameters from config
        self.path: str = root_dir
        self.split: str = split
        self.batch_size: int = batch_size
        self.use_lidar: bool = use_lidar
        self.sample_full_image: bool = sample_full_image
        self.val_downsample: int = val_downsample
        self.val_frame_subsample: int = val_frame_subsample
        self.aux_data: bool = use_aux

        self.cameras: List[CameraDataset] = []      
        if self.use_lidar:
            self.lidars: List[LidarDataset] = []

        self.aabb_scale = None
        # If aabb_scale is None, the scene will be kept true to scale
        self.max_dist_m: float = max_dist_m
        self.use_dynamic_mask: bool = use_dynamic_masks

        assert self.use_dynamic_mask == False if self.sample_full_image else True, "Can't sample full images consistently if dynamic masks are enabled"

        self.img_wh: Optional[Tuple[int, int]] = None
        self.xform_matrices = np.empty((0, 4), dtype=np.float32)
        self.indices_matrix = np.empty((0, 4), dtype=np.uint16)
        self.rgb = np.empty((0, 3), dtype=np.uint8)

        if self.aux_data:
            self.semantic_masks = np.empty((0, 1), dtype=np.uint8)

        self.n_cameras = len(self.cameras)
        self.load_transforms()
        self.set_camera_matrices()
        self.set_indices_matrix()


    def load_transforms(self) -> None:
        ls_dir = glob.glob(os.path.join(self.path, "*.json"))
        split_in_file = np.any([self.split in f.split("/")[-1] for f in ls_dir])

        # load NGP compatible transforms_*train.json
        if os.path.isdir(self.path) and os.path.exists(ls_dir[0]):
            transform_paths = ls_dir
            transform_paths.sort()
            for transform_path in transform_paths:
                if split_in_file and self.split in transform_path.split("/")[-1]:
                    self.add_transform(transform_path)
                elif not split_in_file:
                    print(f"Not found any transforms containing {self.split} in {self.path}")
                    self.add_transform(transform_path)
        elif self.path.endswith(".json"):
            transform_path = self.path
            self.path = "/".join(transform_path.split("/")[:-1])
            self.add_transform(transform_path)
        else:
            raise NotImplementedError(f"NGPDataset: Cannot find *train.json under {self.path}")

        # All cameras have the same world_to_ngp_scale
        self.world_to_ngp_scale = self.cameras[0].world_to_ngp_scale
        self.n_cameras = len(self.cameras)


    def add_transform(self, transform_path):
        with open(transform_path, "r") as f:
            transform_json = json.load(f)  # variable called json in instant-ngp

            if self.aabb_scale and "aabb_scale" in transform_json:
                assert (
                    transform_json["aabb_scale"] == self.aabb_scale
                ), "aabb scale in json file doesn't match the one in the config file"

            # Add CameraDataset
            if "frames" in transform_json:
                self.cameras.append(
                    CameraDataset(
                        self.path,
                        transform_path,
                        transform_json,
                        len(self.cameras),
                        split=self.split,
                        aabb_scale=self.aabb_scale,
                    )
                )

                if self.img_wh is None:
                    self.img_wh = self.cameras[-1].img_wh
                    self.image_w = self.img_wh[0]
                    self.image_h = self.img_wh[1]
            
            # Add LidarDataset
            if self.use_lidar and "lidar" in transform_json:
                self.lidars.append(
                    LidarDataset(
                        self.path,
                        transform_path,
                        transform_json,
                        len(self.lidars),
                        split=self.split,
                        max_dist_m=self.max_dist_m,
                        aabb_scale=self.aabb_scale,
                    )
                )
    def set_camera_matrices(self) -> None:
        def _convert_cam_distortion_mode(mode: str) -> int:
            if mode == "FTheta":
                return 0
            elif mode == "":
                return 1
            elif mode == "Iterative":
                return 2
            else:
                raise NotImplementedError(f"[NGPDataset] camera distortion mode {mode} not known")

        self.intrinsics = np.zeros((self.n_cameras, 6), dtype=np.float32)
        self.camera_distortion_params = np.zeros((self.n_cameras, 7), dtype=np.float32)
        self.camera_distortion_mode = np.zeros((self.n_cameras, 1), dtype=np.int32)
        self.rolling_shutter = np.zeros((self.n_cameras, 4), dtype=np.float32)

        self.n_frames = 0
        for ind_camera, camera in enumerate(self.cameras):
            self.intrinsics[ind_camera, :4] = camera.intrinsics[0]
            self.intrinsics[ind_camera, 4] = camera.w
            self.intrinsics[ind_camera, 5] = camera.h
            self.camera_distortion_params[ind_camera] = camera.camera_distortion_params[0]
            self.camera_distortion_mode[ind_camera] = _convert_cam_distortion_mode(camera.camera_distortion_mode)

            self.rolling_shutter[ind_camera] = camera.rolling_shutter[0]

            self.n_frames += camera.n_frames
        
        if self.split == 'val':
            self.n_frames = int(np.ceil(self.n_frames / self.val_frame_subsample))

        self.max_w = int(np.max(self.intrinsics[:, 4]))
        self.max_h = int(np.max(self.intrinsics[:, 5]))

    def check_valid_json(self, transform_json: dict) -> bool:
        assert "frames" in transform_json, "NGPAVDataset: 'frames' not found in json"
        
        if self.aux_data:
            assert "semantic" in transform_json, "NGPAVDataset: aux_data set but 'semantic' not found in json"
            assert "semantic_meta" in transform_json, "NGPAVDataset: aux_data set but 'semantic_meta' not found in json"
            self.semantic_meta = transform_json["semantic_meta"]
            self.sky_class_id = self.semantic_meta["stuff_classes"].index("sky")
            
        return True
        
    def set_indices_matrix(self):
        self.rgb = np.zeros((self.n_frames * self.max_h * self.max_w, 3), dtype=np.uint8)
        self.indices_matrix = np.zeros((self.n_frames * self.max_h * self.max_w, 4), dtype=np.uint16)

        if self.aux_data:
            self.semantic_masks = np.zeros((self.n_frames * self.max_h * self.max_w, 1), dtype=np.uint8)

        rolling_shutter_flag = np.any(self.rolling_shutter)  # test if all 0s
        if not rolling_shutter_flag:
            self.xform_matrices = np.zeros((self.n_frames * 3, 4), dtype=np.float32)
        else:
            self.xform_matrices = np.zeros((self.n_frames * 2 * 3, 4), dtype=np.float32)

        run_frames = 0
        run_valid_rays = 0
        for cam_ind, cam in enumerate(self.cameras):
            # load json
            with open(cam.transform_path, "r") as f:
                transform_json = json.load(f)

            assert self.check_valid_json(transform_json)
            frames = transform_json["frames"]

            for f_ind, frame in enumerate(
                tqdm(frames, desc=f"Preparing indices matrix for {self.split} frames from camera {cam_ind}")
            ):
                if self.split == 'train' or (self.split == 'val' and f_ind % self.val_frame_subsample == 0):
                    f_path = os.path.join(cam.path, frame["file_path"])

                    # dynamic mask
                    assert cam.w is not None and cam.h is not None, "camera resolution needs to be set"
                    dynamic_mask = np.zeros((cam.h, cam.w), dtype=np.bool_)
                    if self.use_dynamic_mask and self.split == "train":
                        dirname, basename = os.path.split(f_path)
                        f_path_dynamic_mask = os.path.join(
                            dirname, "dynamic_mask_" + os.path.splitext(basename)[0] + ".png"
                        )
                        assert os.path.exists(
                            f_path_dynamic_mask
                        ), f"use_dynamic_mask flag is set but mask not found at {f_path_dynamic_mask}"
                        dynamic_mask_image = PILImage.open(f_path_dynamic_mask)
                        if dynamic_mask_image.size[1] != cam.h or dynamic_mask_image.size[0] != cam.w:
                            dynamic_mask_image = dynamic_mask_image.resize((cam.w, cam.h), PILImage.Resampling.NEAREST)

                        dynamic_mask = np.asarray(dynamic_mask_image).astype(np.bool_)

                    # valid coords
                    frame_valid_y, frame_valid_x = np.where(dynamic_mask == 0)
                    n_valid_rays = len(frame_valid_x)

                    # indices matrix
                    self.indices_matrix[run_valid_rays : run_valid_rays + n_valid_rays, 0] = cam_ind
                    self.indices_matrix[run_valid_rays : run_valid_rays + n_valid_rays, 1] = run_frames
                    self.indices_matrix[run_valid_rays : run_valid_rays + n_valid_rays, 2] = frame_valid_x
                    self.indices_matrix[run_valid_rays : run_valid_rays + n_valid_rays, 3] = frame_valid_y

                    # xform matrices
                    self.set_xform_matrix(frame, rolling_shutter_flag, run_frames, cam.offset)

                    # rgb
                    image = PILImage.open(f_path)
                    if image.size[1] != cam.h or image.size[0] != cam.w:
                        image = image.resize((cam.w, cam.h), PILImage.Resampling.LANCZOS)
                    image = np.asarray(image)  # [H, W, 3]
                    self.rgb[run_valid_rays : run_valid_rays + n_valid_rays] = image.reshape(-1, image.shape[-1])[
                        dynamic_mask.flatten() == 0
                    ]

                    # semantic mask
                    if self.aux_data:
                        f_path_semantic = os.path.join(cam.path, transform_json["semantic"][f_ind]["file_path"])
                        assert os.path.exists(
                            f_path_semantic
                        ), f"NGPAVDataset: aux_data flag is set but semantic mask not found at {f_path_semantic}"
                        semantic_mask = PILImage.open(f_path_semantic)
                        if semantic_mask.size[1] != cam.h or semantic_mask.size[0] != cam.w:
                            semantic_mask = semantic_mask.resize((cam.w, cam.h), PILImage.Resampling.NEAREST)
                        semantic_mask = np.asarray(semantic_mask)
                        self.semantic_masks[run_valid_rays : run_valid_rays + n_valid_rays] = semantic_mask.reshape(
                            -1, 1
                        )[dynamic_mask.flatten() == 0]

                    run_frames += 1
                    run_valid_rays += n_valid_rays

        self.n_valid_rays = run_valid_rays
        self.n_frames = run_frames
        assert len(self.rgb) == len(self.indices_matrix)

        if self.aux_data:
            assert len(self.semantic_masks) == len(self.indices_matrix)

    def set_xform_matrix(
        self, frame: dict, rolling_shutter_flag: Union[bool, np.bool_], run_frames: int, offset: np.ndarray
    ) -> None:
        assert self.xform_matrices is not None, "NGPDataset: tried to set x_form_matrices but not defined"
        if rolling_shutter_flag:
            if not ("transform_matrix_start" in frame and "transform_matrix_end" in frame):
                raise KeyError(
                    "NGPAVDataset: transform_matrix_start or transform_matrix_end not found in frame but rolling_shutter_flag is set"
                )
            jsonmatrix_start = np.array(frame["transform_matrix_start"], dtype=np.float32)  # [4, 4]
            jsonmatrix_start = nerf_matrix_to_colmap(jsonmatrix_start, scale=self.world_to_ngp_scale, offset=offset)

            jsonmatrix_end = np.array(frame["transform_matrix_end"], dtype=np.float32)  # [4, 4]
            jsonmatrix_end = nerf_matrix_to_colmap(jsonmatrix_end, scale=self.world_to_ngp_scale, offset=offset)

            self.xform_matrices[2 * 3 * run_frames : 2 * 3 * run_frames + 3] = jsonmatrix_start[:-1, :]
            self.xform_matrices[2 * 3 * run_frames + 3 : 2 * 3 * run_frames + 6] = jsonmatrix_end[:-1, :]
        else:
            if not ("transform_matrix" in frame):
                if not ("transform_matrix_end" in frame):
                    raise KeyError(
                        "NGPAVDataset: neither transform_matrix nor transform_matrix_end not found in frame but rolling_shutter_flag is not set"
                    )
                else:
                    jsonmatrix = np.array(frame["transform_matrix_end"], dtype=np.float32)  # [4, 4]
            else:
                jsonmatrix = np.array(frame["transform_matrix"], dtype=np.float32)  # [4, 4]
            jsonmatrix = nerf_matrix_to_colmap(jsonmatrix, scale=self.world_to_ngp_scale, offset=offset)

            self.xform_matrices[3 * run_frames : 3 * run_frames + 3] = jsonmatrix[:-1, :]

    def get_observer_points(self):
        rolling_shutter_flag = np.any(self.rolling_shutter)
        if rolling_shutter_flag:
            return self.xform_matrices[:,3].reshape(self.n_frames*2,3)[::2] # returning t_start observer points
        else:
            return self.xform_matrices[:,3].reshape(self.n_frames,3)

    def get_point_cloud(
        self,
        non_dynamic_points_only: bool = True,
        step_frame: int = 1,
        device: str = 'cpu'
    ) -> PointCloud:
        """Returns a point-cloud for all available for point-cloud sensor (lidar / camera), transformed into NGP frame.

        Can be parameterized to only return non-dynamic points (default).
        """

        if not non_dynamic_points_only:
            print(
                "NGPAVDataset: dynamic points requested, but NGPAVDataset only loads non-dynamic ones by default"
            )

        xyz_starts = []
        xyz_ends = []
        for lidar_dataset in self.lidars:
            all_frame_idxs = np.unique(lidar_dataset.lidar_frame_indices)

            for frame_idx in all_frame_idxs[::step_frame]:
                valid_rays = np.where(lidar_dataset.lidar_frame_indices == frame_idx)[0]
                xyz_starts.append(to_torch(lidar_dataset.rays[valid_rays, :3], device=device))
                xyz_ends.append(to_torch(lidar_dataset.rays[valid_rays, :3] + lidar_dataset.rays[valid_rays, 3:6], device=device))
        return PointCloud(xyz_start=torch.cat(xyz_starts), xyz_end=torch.cat(xyz_ends), device=device)
    
    def get_sky_rays(
        self,
        camera_idx: Optional[int] = None,
        step_frame: int = 1,
        step_pixel: int = 1,
    ) -> torch.Tensor:
        """Returns all camera rays (Nx6) belonging to a given semantic class.

        Camera sensor are specified by by either logical or unique sensor IDs.

        Each generated sample represents a different frame."""

        assert len(self.cameras), "[NGPAVDataset]: no camera sensors loaded"
        assert self.aux_data, "[NGPAVDataset]: Auxiliary data was not loaded"
        assert self.indices_matrix is not None, "[NGPAVDataset] indices_matrix needs to be set"

        # map sensor IDs to indices
        camera_idx = camera_idx if camera_idx is not None else 0 

        assert camera_idx < len(
            self.cameras
        ), f"Only {len(self.cameras)} camera sensors are available, but {camera_idx} was selected."

        camera_id_mask = self.indices_matrix[:, 0] == camera_idx
        camera_id_indices_matrix = self.indices_matrix[camera_id_mask]
        camera_id_semantics = self.semantic_masks[camera_id_mask]
        all_frame_idxs = np.unique(camera_id_indices_matrix[:, 1])  # returned indices are sorted

        rays_out = []
        for frame_idx in all_frame_idxs[::step_frame]:

            valid_pixels = np.where(camera_id_indices_matrix[:, 1] == frame_idx)[0]
            semantic_mask = camera_id_semantics[valid_pixels, 0] == self.sky_class_id
            rays = ray_utils._batchPix2WorldRays(
                camera_id_indices_matrix[valid_pixels[semantic_mask]].astype(np.int32),
                self.intrinsics,
                self.camera_distortion_params,
                self.rolling_shutter,
                self.xform_matrices,
                self.camera_distortion_mode,
            )

            rays_out.append(rays[::step_pixel])
        return torch.cat(rays_out)

    def __len__(self):
        if self.split.startswith("train"):
            return self.n_frames if self.sample_full_image else 1000

        return self.n_frames

    def __getitem__(self, idx):
        if self.split == "train":
            if self.sample_full_image:
                # Sample specific frame
                idxs = np.where(self.indices_matrix[:, 1] == idx)[0]
                out_shape = (self.image_h,self.image_w,3)
            else:
                # only sample from valid RGB rays
                idxs = np.random.choice(self.n_valid_rays, self.batch_size)
                out_shape = (1,self.batch_size,3)

            rays = ray_utils._batchPix2WorldRays(
                self.indices_matrix[idxs].astype(np.int32),
                self.intrinsics,
                self.camera_distortion_params,
                self.rolling_shutter,
                self.xform_matrices,
                self.camera_distortion_mode,
            )


            rays_o = rays[..., :3]
            rays_d = rays[..., 3:]

            rgbs = self.rgb[idxs].astype(np.float32) / 255

            sample = {
                "rays_ori":rays_o.reshape(out_shape), 
                "rays_dir":rays_d.reshape(out_shape), 
                "rgb_gt":rgbs.reshape(out_shape),
            }
            if self.aux_data:
                sample["sky_mask"]= (self.semantic_masks[idxs] == self.sky_class_id).squeeze().reshape(out_shape[0],out_shape[1],1)

            return sample
        elif self.split == 'val':
            # choose indices for specific test image

            assert (self.image_w % self.val_downsample==0) and (self.image_h % self.val_downsample == 0), 'need to set appropriate val_downsample factor'
            idxs = np.where(self.indices_matrix[:, 1] == idx)[0]
            
            rays = ray_utils._batchPix2WorldRays(
                self.indices_matrix[idxs].astype(np.int32),
                self.intrinsics,
                self.camera_distortion_params,
                self.rolling_shutter,
                self.xform_matrices,
                self.camera_distortion_mode,
            )

            # Subsample to speed up validation
            rays = rays.reshape(self.image_h, self.image_w, 6)
            rays = rays[:: self.val_downsample, :: self.val_downsample]

            rays_o = rays[..., :3]
            rays_d = rays[..., 3:]

            rgbs = self.rgb[idxs].astype(np.float32) / 255
            rgbs = rgbs.reshape(self.image_h, self.image_w, 3)
            rgbs = rgbs[:: self.val_downsample, :: self.val_downsample]


            sample = {
                "rays_ori":rays_o, 
                "rays_dir":rays_d, 
                "rgb_gt":rgbs
            }
            return sample


class CameraDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform_path: str,
        transform_json: dict,
        camera_index: int = 0,
        split: str = "train",
        aabb_scale: Optional[float] = None,
        downsample: float = 1.0,
    ) -> None:

        super().__init__()

        self.split = split  # TODO implement train/val/test split capabilities
        self.path = path
        self.transform_path = transform_path
        self.camera_index = camera_index
        self.h: Optional[int] = None
        self.w: Optional[int] = None
        self.img_wh: Optional[tuple[int, int]] = None
        self.aabb_scale = aabb_scale
        self.downsample = downsample

        # load image size
        if "h" in transform_json and "w" in transform_json:
            self.h = int(transform_json["h"])
            self.w = int(transform_json["w"])
            self.img_wh = (self.w, self.h)

        # read images
        frames = transform_json["frames"]

        self.world_to_ngp_scale = 1.0
        self.offset = np.zeros((3,))
        if "offset" in transform_json:
            self.offset.put(
                [0, 1, 2],
                transform_json["offset"]
                if isinstance(transform_json["offset"], list)
                else [transform_json["offset"], transform_json["offset"], transform_json["offset"]],
            )

        """ If null is given to aabb_scale in the config, we ignore the value form the .json file and keep the scene true to scale
            To this end we also need to reverse the scaling of the offset """
        if self.aabb_scale:
            self.world_to_ngp_scale = transform_json.get("scale", 0.33)
            self.offset -= np.ones((3,)) * 0.5
        else:
            self.offset = (self.offset - np.ones((3,)) * 0.5) / transform_json.get("scale", 0.33)

        self.read_camera_distortion(transform_json)

        # load intrinsics
        if not self.read_focal_length(transform_json):
            raise RuntimeError("Failed to load focal length, please check the transforms.json!")

        if self.downsample != 1.0:
            self.downsample_intrinsics()

        self.n_frames = len(frames)

    def read_camera_distortion(self, transform_json):
        self.camera_distortion_mode = ""
        self.camera_distortion_params = np.zeros((1, 7), dtype=np.float32)
        self.rolling_shutter = np.zeros((1, 4), dtype=np.float32)
        self.intrinsics = np.zeros((1, 4), dtype=np.float32)

        if "k1" in transform_json:
            self.camera_distortion_params[0, 0] = transform_json["k1"]
            self.camera_distortion_mode = "Iterative"

        if "k2" in transform_json:
            self.camera_distortion_params[0, 1] = transform_json["k2"]

        if "p1" in transform_json:
            self.camera_distortion_params[0, 2] = transform_json["p1"]

        if "p2" in transform_json:
            self.camera_distortion_params[0, 3] = transform_json["p2"]

        if "cx" in transform_json:
            self.intrinsics[0, 0] = transform_json["cx"]

        if "cy" in transform_json:
            self.intrinsics[0, 1] = transform_json["cy"]

        if "rolling_shutter" in transform_json:
            # the rolling shutter is a list of [A,B,C] where the time
            # for each pixel is t= A + B * u + C * v
            # where u and v are the pixel coordinates (0-1),
            # and the resulting t is used to interpolate between the start
            # and end transforms for each training xform

            if len(transform_json["rolling_shutter"]) >= 4:
                self.rolling_shutter[0, 3] = transform_json["rolling_shutter"][3]
            self.rolling_shutter[0, :3] = transform_json["rolling_shutter"][:3]
            assert (self.rolling_shutter[0, 1] == 0) ^ (self.rolling_shutter[0, 2] == 0)

        if "ftheta_p0" in transform_json:
            self.camera_distortion_params[0, 0] = transform_json["ftheta_p0"]
            self.camera_distortion_params[0, 1] = transform_json["ftheta_p1"]
            self.camera_distortion_params[0, 2] = transform_json["ftheta_p2"]
            self.camera_distortion_params[0, 3] = transform_json["ftheta_p3"]
            self.camera_distortion_params[0, 4] = transform_json["ftheta_p4"]
            self.camera_distortion_params[0, 5] = transform_json["w"]
            self.camera_distortion_params[0, 6] = transform_json["h"]
            self.camera_distortion_mode = "FTheta"
            if self.downsample != 1:  # downsampling for the ftheta camera model not supported
                raise NotImplementedError(
                    f"Downsampling factor is : {self.downsample} and camera model is Ftheta, check that you pointed to the correct json"
                )

    def read_focal_length(self, transform_json: dict) -> bool:
        def _read_focal_length(resolution, axis):
            if axis + "_fov" in transform_json:
                return fov2focal(transform_json[axis + "_fov"], resolution)
            elif "fl_" + axis in transform_json:
                return transform_json["fl_" + axis]
            elif "camera_angle_" + axis in transform_json:
                return fov2focal(transform_json["camera_angle_" + axis], resolution)
            else:
                return 0.0

        x_fl = _read_focal_length(self.w, "x")
        y_fl = _read_focal_length(self.h, "y")

        if x_fl != 0:
            self.intrinsics[0, 2] = x_fl
            self.intrinsics[0, 3] = x_fl
            if y_fl != 0:
                self.intrinsics[0, 3] = y_fl
            return True
        elif y_fl != 0:
            self.intrinsics[0, 2] = y_fl
            self.intrinsics[0, 3] = y_fl
            return True
        else:
            return False

    def downsample_intrinsics(self) -> None:
        assert (
            self.camera_distortion_mode != "FTheta"
        ), "[NGPDatset]: downsampling is not implemented for FTheta cameras."

        # Scale the principal point and the focal length
        self.intrinsics[0, 0] *= self.downsample
        self.intrinsics[0, 1] *= self.downsample
        self.intrinsics[0, 2] *= self.downsample
        self.intrinsics[0, 3] *= self.downsample

        assert self.w is not None and self.h is not None, "width and height need to be set"

        self.w = int(self.w * self.downsample)
        self.h = int(self.h * self.downsample)
        self.img_wh = (self.w, self.h)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int) -> None:
        pass


class LidarDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform_path: str,
        transform_json: dict,
        lidar_number: int = 0,
        split: str = "train",
        max_dist_m: float = 150.0,
        aabb_scale: Optional[float] = None,
    ) -> None:

        super().__init__()
        self.path = path
        self.transform_path = transform_path
        self.lidar_number = lidar_number
        self.split = split
        self.max_dist_m = max_dist_m
        self.aabb_scale = aabb_scale

        # Initialize the world to ngp scale and offset
        self.world_to_ngp_scale = 1.0
        self.offset = np.zeros((3,))
        if "offset" in transform_json:
            self.offset.put(
                [0, 1, 2],
                transform_json["offset"]
                if isinstance(transform_json["offset"], list)
                else [transform_json["offset"], transform_json["offset"], transform_json["offset"]],
            )

        """ If null is given to aabb_scale in the config, we ignore the value form the .json file and keep the scene true to scale
            To this end we also need to reverse the scaling of the offset """
        if self.aabb_scale:
            self.world_to_ngp_scale = transform_json.get("scale", 0.33)
            self.offset -= np.ones((3,)) * 0.5
        else:
            self.offset = (self.offset - np.ones((3,)) * 0.5) / transform_json.get("scale", 0.33)

        rays_list = []
        lidar_frame_indices_list = []
        frames = transform_json["lidar"]
        self.n_frames = len(frames)
        for i, f in enumerate(tqdm(frames, desc=f"Loading {self.split} frames from lidar {self.lidar_number}")):
            f_path = os.path.join(self.path, f["file_path"])
            raw_rays = load_pc_dat(f_path)
            raw_rays = raw_rays[raw_rays[:, -1] != 1]  # last column is dynamic flag

            rays_o = raw_rays[:, :3]
            rays_d = (
                raw_rays[:, 3:6] - rays_o
            )  # instant-ngp stores these un-normalized, so that the distance is implicit in the 'd' field

            rays_o, rays_d = nerf_ray_to_colmap(
                rays_o, rays_d, scale=self.world_to_ngp_scale, offset=self.offset, scale_direction=True
            )
            rays_dist = np.linalg.norm(rays_d, axis=1)

            mask_dist = rays_dist < (self.max_dist_m * self.world_to_ngp_scale)

            rays_o, rays_d = rays_o[mask_dist], rays_d[mask_dist]

            rays_list.append(np.hstack([rays_o, rays_d]))
            lidar_frame_indices_list.append(np.full(len(rays_o), i, np.int32))

        self.rays = np.vstack(rays_list).astype(np.float32)
        self.n_valid_rays = len(self.rays)
        self.lidar_frame_indices = np.hstack(lidar_frame_indices_list)

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx):
        pass
