import logging, os
from typing import Optional

import numpy as np
import torch
import tqdm
from plyfile import PlyData

from libs import optixtracer
from utils import to_torch, get_activation_function, inverse_sigmoid, get_scheduler, quaternion_to_so3, \
    sh_degree_to_num_features, sh_degree_to_specular_dim
from datasets.colmap_utils import read_next_bytes
from datasets.utils import PointCloud
from geometry import nearest_neighbor_dist_cpuKD
from utils import to_np
import background
from color import RGB2SH
from render_utils import evaluate_rays

class MixtureOfGaussians(torch.nn.Module):
    def __init__(self, conf, scene_extent=None):
        super().__init__()

        self.conf = conf
        self.scene_extent = scene_extent
        self.positions = torch.nn.Parameter(torch.empty([0, 3]))  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation = torch.nn.Parameter(torch.empty([0, 4]))  # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale = torch.nn.Parameter(torch.empty([0, 3]))  # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density = torch.nn.Parameter(torch.empty([0, 1]))  # Density of each Gaussian [n_gaussians, 1]
        self.features_albedo = torch.nn.Parameter(torch.empty([0, 3]))  # Feature vector of the 0th order SH coefficients [n_gaussians, 3] (We split it into two due to different learning rates)
        self.features_specular = torch.nn.Parameter(torch.empty([0, 1]))  # Features of the higher order SH coefficients [n_gaussians, 3]
        
        if self.conf.model.log_rolling_buffers:
            self.rolling_error = torch.empty([0,1]) 
            self.rolling_weight_contrib = torch.empty([0,1]) 

        match self.conf.model.densify.method:
            case 'gradient-buffer':
                # Accumulation of the norms of the positions gradients
                self.positional_grad_norm_accum = torch.empty([0,1]) 
                self.positional_grad_norm_denom = torch.empty([0,1])
            case 'error':
                assert self.conf.model.log_rolling_buffers, "must log weights and errors to densify based on them"
            case 'adam':
                pass 
            case _:
                raise ValueError(f"densify.method {self.conf.model.densify.method} not supported")            
        
        self.device = 'cuda'
        self.optimizer = None
        self.optix_ctx = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.density_activation_inv = get_activation_function(self.conf.model.density_activation, inverse=True)
        self.scale_activation =  get_activation_function(self.conf.model.scale_activation)
        self.scale_activation_inv =  get_activation_function(self.conf.model.scale_activation, inverse=True)
        self.rotation_activation =   get_activation_function("normalize") # The default value of the dim parameter is 1

        # Rendering parameters
        self.render_method = conf.render.method

        self.background = background.make(self.conf.model.background.name, self.conf.model.background)

        # Parameters related to densification, pruning and reset
        self.split_n_gaussians = self.conf.model.densify.split.n_gaussians
        self.relative_size_threshold = self.conf.model.densify.relative_size_threshold
        self.prune_density_threshold = self.conf.model.prune.density_threshold
        self.clone_grad_threshold = self.conf.model.densify.clone_grad_threshold
        self.split_grad_threshold = self.conf.model.densify.split_grad_threshold
        self.new_max_density = self.conf.model.reset_density.new_max_density

        # Check if we would like to do progressive training
        self.feature_type = self.conf.model.progressive_training.feature_type
        self.n_active_features = self.conf.model.progressive_training.init_n_features
        self.max_n_features = self.conf.model.progressive_training.max_n_features   # For SH, this is the SH degree
        self.progressive_training = False
        if self.n_active_features < self.max_n_features:
            self.feature_dim_increase_interval = self.conf.model.progressive_training.increase_frequency
            self.feature_dim_increase_step = self.conf.model.progressive_training.increase_step
            self.progressive_training = True

    def validate_fields(self):
        num_gsplat = self.positions.shape[0]
        assert self.positions.shape == (num_gsplat, 3)
        assert self.density.shape == (num_gsplat, 1)
        assert self.rotation.shape == (num_gsplat, 4)
        assert self.scale.shape == (num_gsplat, 3)

        if self.conf.model.log_rolling_buffers:
            assert self.rolling_error.shape == (num_gsplat, 1)
            assert self.rolling_weight_contrib.shape == (num_gsplat, 1)

        match self.conf.model.densify.method:
            case 'gradient-buffer':
                assert self.positional_grad_norm_accum.shape == (num_gsplat, 1)
                assert self.positional_grad_norm_denom.shape == (num_gsplat, 1)
            case 'error':
                assert self.conf.model.log_rolling_buffers
            case 'adam': 
                pass
            case _:
                raise ValueError(f"densify.method {self.conf.model.densify.method} not supported")            
            
        if self.feature_type == 'sh':
            assert self.features_albedo.shape == (num_gsplat, 3)
            specular_sh_dims = sh_degree_to_specular_dim(self.max_n_features)
            assert self.features_specular.shape == (num_gsplat, specular_sh_dims)
        else:
            raise ValueError('Neural features not yet supported.')

    def set_optix_context(self):
        torch.zeros(1, device=self.device) # Create a dummy tensor to force cuda context init
        self.optix_ctx = optixtracer.OptiXContext(
            params = optixtracer.OptixMogTracingParams(
                hit_mode = self.conf.render.hit_mode,
                max_hit_per_slab = self.conf.render.max_hit_per_slab,
                max_num_slabs = self.conf.render.max_num_slabs,
                topk_hits = self.conf.render.topk_hits,
                patch_size = self.conf.render.patch_size,
                sph_degree = 0, # Dummy, dynamically controlled
                gaussian_sigma_threshold = self.conf.render.gaussian_sigma_threshold,
                min_transmittance = self.conf.render.min_transmittance,
                max_hits_returned=128,
            )
        )

    def init_from_colmap(self, root_path: str, observer_pts):
        # TODO this reads from the binary format, also implement the nearly-identical plaintext version?
        points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
        if not os.path.isfile(points_file):
            raise ValueError(f"colmap points file {points_file} not found")

        with open(points_file, "rb") as file:

            n_pts = read_next_bytes(file, 8, "Q")[0]
            logging.info(f"Found {n_pts} colmap points")

            file_pts = np.zeros((n_pts, 3), dtype=np.float32)
            file_rgb = np.zeros((n_pts, 3), dtype=np.float32)

            for i_pt in range(n_pts):
                # read the points
                pt_data = read_next_bytes(file, 43, "QdddBBBd")
                file_pts[i_pt, :] = np.array(pt_data[1:4])
                file_rgb[i_pt, :] = np.array(pt_data[4:7])
                # NOTE: error stored in last element of file, currently not used

                # skip the track data
                t_len = read_next_bytes(file, num_bytes=8, format_char_sequence="Q")[0]
                read_next_bytes(file, num_bytes=8 * t_len, format_char_sequence="ii" * t_len)

        file_rgb = file_rgb / 255.

        file_pts = torch.tensor(file_pts, dtype=torch.float32, device=self.device)
        file_rgb = torch.tensor(file_rgb, dtype=torch.float32, device=self.device)

        self.default_initialize_from_points(file_pts, observer_pts, file_rgb)

    def init_from_pretrained_point_cloud(self, pc_path: str, set_optimizable_parameters: bool = True):
        data = PlyData.read(pc_path)
        num_gsplat = len(data['vertex'])
        self.positions = torch.nn.Parameter(to_torch(
            np.transpose(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), dtype=np.float32)),
            device=self.device))  # type: ignore
        self.rotation = torch.nn.Parameter(to_torch(np.transpose(np.stack(
            (data['vertex']['rot_0'], data['vertex']['rot_1'], data['vertex']['rot_2'], data['vertex']['rot_3']),
            dtype=np.float32)), device=self.device))  # type: ignore
        self.scale = torch.nn.Parameter(to_torch(np.transpose(
            np.stack((data['vertex']['scale_0'], data['vertex']['scale_1'], data['vertex']['scale_2']),
                     dtype=np.float32)), device=self.device))  # type: ignore
        self.density = torch.nn.Parameter(
            to_torch(data['vertex']['opacity'].astype(np.float32).reshape(num_gsplat, 1), device=self.device))
        self.features_albedo = torch.nn.Parameter(to_torch(np.transpose(np.stack((
            data['vertex']['f_dc_0'], data['vertex']['f_dc_1'], data['vertex']['f_dc_2']),
            dtype=np.float32)), device=self.device))  # type: ignore

        feats_sph = to_torch(np.transpose(np.stack((
            data['vertex']['f_rest_0'], data['vertex']['f_rest_1'], data['vertex']['f_rest_2'],
            data['vertex']['f_rest_3'], data['vertex']['f_rest_4'],
            data['vertex']['f_rest_5'], data['vertex']['f_rest_6'], data['vertex']['f_rest_7'],
            data['vertex']['f_rest_8'], data['vertex']['f_rest_9'],
            data['vertex']['f_rest_10'], data['vertex']['f_rest_11'], data['vertex']['f_rest_12'],
            data['vertex']['f_rest_13'], data['vertex']['f_rest_14'],
            data['vertex']['f_rest_15'], data['vertex']['f_rest_16'], data['vertex']['f_rest_17'],
            data['vertex']['f_rest_18'], data['vertex']['f_rest_19'],
            data['vertex']['f_rest_20'], data['vertex']['f_rest_21'], data['vertex']['f_rest_22'],
            data['vertex']['f_rest_23'], data['vertex']['f_rest_24'],
            data['vertex']['f_rest_25'], data['vertex']['f_rest_26'], data['vertex']['f_rest_27'],
            data['vertex']['f_rest_28'], data['vertex']['f_rest_29'],
            data['vertex']['f_rest_30'], data['vertex']['f_rest_31'], data['vertex']['f_rest_32'],
            data['vertex']['f_rest_33'], data['vertex']['f_rest_34'],
            data['vertex']['f_rest_35'], data['vertex']['f_rest_36'], data['vertex']['f_rest_37'],
            data['vertex']['f_rest_38'], data['vertex']['f_rest_39'],
            data['vertex']['f_rest_40'], data['vertex']['f_rest_41'], data['vertex']['f_rest_42'],
            data['vertex']['f_rest_43'], data['vertex']['f_rest_44']),
        dtype=np.float32)), device=self.device)

        # reinterpret from C-style to F-style layout
        feats_sph = feats_sph.reshape(num_gsplat,3,-1).transpose(-1,-2).reshape(num_gsplat,-1)
        
        self.features_specular = torch.nn.Parameter(feats_sph)
        
        self.init_densification_buffer()

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    @torch.no_grad()
    def init_from_random_point_cloud(self,
                                     num_gsplat: int = 100_000,
                                     dtype=torch.float32,
                                     set_optimizable_parameters: bool = True,
                                     xyz_max = 1.5,
                                     xyz_min = -1.5):

        logging.info(f"Generating random point cloud ({num_gsplat})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz in [-1.5, 1.5] -> standard NeRF convention, people often scale with 0.33 to get it to [-0.5, 0.5]
        fused_point_cloud = torch.rand((num_gsplat, 3), dtype=dtype, device=self.device) * (xyz_max - xyz_min) + xyz_min
        # sh albedo in [0, 0.0039]
        fused_color = torch.rand((num_gsplat, 3), dtype=dtype, device=self.device) / 255.0

        features_albedo = features_specular = None
        if self.feature_type == 'sh':
            features_albedo = fused_color.contiguous()
            max_sh_degree = self.max_n_features
            num_specular_features = sh_degree_to_specular_dim(max_sh_degree)
            features_specular = torch.zeros((num_gsplat, num_specular_features),
                                            dtype=dtype, device=self.device).contiguous()

        dist = torch.clamp_min(nearest_neighbor_dist_cpuKD(fused_point_cloud), 1e-3)
        scales = torch.log(dist)[..., None].repeat(1, 3)
        rots = torch.zeros((num_gsplat, 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((num_gsplat, 1), dtype=dtype, device=self.device))

        self.positions = torch.nn.Parameter(fused_point_cloud)  # type: ignore
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.init_densification_buffer()

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    def init_from_checkpoint(self, checkpoint: dict, setup_optimizer=True):
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features_albedo = checkpoint["features_albedo"]
        self.features_specular = checkpoint["features_specular"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.scene_extent = checkpoint["scene_extent"]

        if self.progressive_training:
            self.feature_dim_increase_interval = checkpoint["feature_dim_increase_interval"]
            self.feature_dim_increase_step = checkpoint["feature_dim_increase_step"]

        self.init_densification_buffer(checkpoint)

        self.background.load_state_dict(checkpoint["background"])
        if setup_optimizer:
            self.set_optimizable_parameters()
            self.setup_optimizer(state_dict=checkpoint['optimizer'])
        self.validate_fields()

    def default_initialize_from_points(self, pts, observer_pts, colors=None):
        """
        Given an Nx3 array of points (and optionally Nx3 rgb colors), 
        initialize default values for the other parameters of the model
        """
                              
        dtype = torch.float32

        N = pts.shape[0]
        positions = pts
       
        # identity rotations
        rots = torch.zeros((N,4), dtype=dtype, device=self.device)
        rots[:,0] = 1. # they're quaternions

        # estimate scales based on distances to observers
        dist_to_observers = torch.clamp_min(nearest_neighbor_dist_cpuKD(pts, observer_pts), 1e-7)
        observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        scales = self.scale_activation_inv(observation_scale)[:,None].repeat(1,3)

        # set density as a constant
        opacities = self.density_activation_inv(torch.full((N,1), fill_value=0.1, dtype=dtype, device=self.device))

        # set colors, constant if they weren't given
        if colors is None:
            features_albedo = torch.rand((N, 3), dtype=dtype, device=self.device) / 255.0
        else:
            features_albedo = to_torch(RGB2SH(to_np(colors)), device=self.device)
                                       
        num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims))

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.init_densification_buffer()
        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()
            
    def init_from_lidar(self, 
                        point_cloud : PointCloud, 
                        observer_pts):
        """
        Observer points can be any set locations that observation came from. Camera centers, ray source points, etc. They are used to esimate initial scales.
        """
        
        logging.info(f"Initializing based on lidar point cloud ...")
       
        # only initialize by default from points for now
        self.default_initialize_from_points(point_cloud.xyz_end.to(device=self.device), observer_pts, None)

    def delete_hit_gaussians(self, mask):
        # take output of the forward pass and delete gaussians that where hit
        optimizable_tensors = self.prune_optimizer_tensors(mask=mask)
        self.update_optimizable_parameters(optimizable_tensors)

    def init_from_auxiliary_data(self, dataset, scene_bbox, spacing, hit_count_threshold, sky_step_frame, sky_step_pixel, pc_step_frame, pc_step_points):
        # Initialize Gaussian grid in the scene
        dx = torch.arange(scene_bbox[0][0], scene_bbox[1][0], spacing, device='cpu')
        dy = torch.arange(scene_bbox[0][1], scene_bbox[1][1], spacing, device='cpu')
        dz = torch.arange(scene_bbox[0][2], scene_bbox[1][2], spacing, device='cpu')
        grid_x, grid_y, grid_z = torch.meshgrid(dx, dy, dz)
        random_point_cloud = torch.cat([grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)], dim=1)

        observer_points = torch.tensor(dataset.get_observer_points(), dtype=torch.float32, device='cpu')

        print(f"Initializing {len(grid_x.flatten())} evenly-spaced Gaussiasn...")
        self.default_initialize_from_points(random_point_cloud, observer_points, None)

        self.set_optix_context()
        self.build_bvh()
        
        # Iterate over the point clouds and prune away gaussians along lidar rays
        point_clouds = []
        num_deleted_lidar = 0
        for lidar_pc in tqdm.tqdm(
            dataset.get_point_clouds(step_frame=pc_step_frame),
            desc="Pruning Gaussians with lidar rays...",
        ):
            ray_o = lidar_pc.xyz_start.to(self.device)
            ray_d = (lidar_pc.xyz_end - lidar_pc.xyz_start).to(self.device)
            ray_d /= torch.linalg.norm(ray_d, dim=1, keepdim=True)
                       
            point_clouds.append(lidar_pc)

            # Prune Gaussians along lidar ray
            hit_counts = self.get_hit_counts(rays_o=ray_o[None,None,::pc_step_points,:],rays_d=ray_d[None,None,::pc_step_points,:])
            mask = (hit_counts <= hit_count_threshold).squeeze()
            if (~mask).any():
                self.delete_hit_gaussians(mask)
                self.build_bvh()
            num_deleted_lidar += (~mask).sum()
        print(f"Culled {num_deleted_lidar} Gaussians using lidar rays...")


        # Iterate over the sky rays and prune away gaussians along them
        num_deleted_sky = 0
        for semantic_rays in tqdm.tqdm(
            dataset.get_sky_rays(step_frame=sky_step_frame, step_pixel=sky_step_pixel),
            desc="Pruning Gaussians with sky rays...",
        ):
            ray_o = semantic_rays[:, :3].to(self.device)
            ray_d = semantic_rays[:, 3:6].to(self.device)
            ray_d /= torch.linalg.norm(ray_d, dim=1, keepdim=True)


            # Prune Gaussians along ray
            hit_counts = self.get_hit_counts(rays_o=ray_o[None,None,...],rays_d=ray_d[None,None,...])
            mask = (hit_counts <= hit_count_threshold).squeeze()
            if (~mask).any():
                self.delete_hit_gaussians(mask)
                self.build_bvh()
            num_deleted_sky += (~mask).sum()
        print(f"Culled {num_deleted_sky} Gaussians using sky rays...")

        # Get the aggregated point cloud and add the gaussians back
        aggregated_point_cloud = PointCloud.from_sequence(point_clouds, device=self.device).xyz_end
        N = aggregated_point_cloud.size()[0]
        print(f"Adding {N} Gaussians using lidar rays...")

        dtype = torch.float

        # identity rotations
        rots = torch.zeros((N,4), dtype=dtype, device=self.device)
        rots[:,0] = 1. # they're quaternions

        # estimate scales based on distances to observers
        dist_to_observers = torch.clamp_min(nearest_neighbor_dist_cpuKD(aggregated_point_cloud, observer_points), 1e-7)
        observation_scale = dist_to_observers * self.conf.initialization.observation_scale_factor
        scales = self.scale_activation_inv(observation_scale)[:,None].repeat(1,3)

        # set density as a constant
        opacities = self.density_activation_inv(torch.full((N,1), fill_value=0.1, dtype=dtype, device=self.device))

        # set color as a constant
        features_albedo = torch.rand((N, 3), dtype=dtype, device=self.device) / 255.0
                                       
        num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims))

        add_gaussians = {
                "positions": torch.nn.Parameter(aggregated_point_cloud.to(dtype=dtype, device=self.device)),
                "density":  torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device)),
                "scale":  torch.nn.Parameter(scales.to(dtype=dtype, device=self.device)),
                "rotation": torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
                }
        if self.feature_type == 'sh':
            add_gaussians["features_albedo"] = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
            add_gaussians["features_specular"] = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.densify_postfix(add_gaussians)
        print(f"Initialized a total of {self.positions.shape[0]} Gaussians")

    def setup_optimizer(self, state_dict=None):
        params = []
        for name, args in self.conf.optimizer.params.items():
            module =  getattr(self, name)
            if isinstance(module, torch.nn.Module):
                module_parameters = filter(lambda p: p.requires_grad and len(p)>0, module.parameters())
                n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                if n_params > 0:
                    params.append({"params": module.parameters(), "name": name, **args})

            elif isinstance(module, torch.nn.Parameter):
                if module.requires_grad:
                    params.append({"params": [module], "name": name, **args})

        self.optimizer = torch.optim.Adam(params, lr=self.conf.optimizer.lr, eps=self.conf.optimizer.eps)

        self.setup_scheduler()

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)
    
    def setup_scheduler(self):
        self.schedulers = {}
        for name, args in self.conf.scheduler.items():
            if args.type is not None and getattr(self, name).requires_grad:
                if name == "positions":
                    self.schedulers[name] = get_scheduler(args.type)(
                        lr_init=args.lr_init * self.scene_extent,
                        lr_final=args.lr_final * self.scene_extent,
                        max_steps=args.max_steps
                    )
                else:
                    self.schedulers[name] = (get_scheduler(args.type)(**args))

    def scheduler_step(self, step):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in self.schedulers:
                lr = self.schedulers[param_group["name"]](step)
                if lr is not None:
                    param_group['lr'] = lr

    def set_optimizable_parameters(self):
        if not self.conf.model.optimize_density:
            self.density.requires_grad = False
        if not self.conf.model.optimize_features_albedo:
            self.features_albedo.requires_grad = False
        if not self.conf.model.optimize_features_specular:
            self.features_specular.requires_grad = False
        if not self.conf.model.optimize_rotation:
            self.rotation.requires_grad = False
        if not self.conf.model.optimize_scale:
            self.scale.requires_grad = False
        if not self.conf.model.optimize_position:
            self.positions.requires_grad = False

    def update_optimizable_parameters(self, optimizable_tensors: dict[str, torch.Tensor]):
        for name, value in optimizable_tensors.items():
            setattr(self, name, value)

    def build_bvh(self):
        optixtracer.build_mog_bvh(self.optix_ctx, self.positions, self.rotation_activation(self.rotation), self.scale_activation(self.scale), True)

    def increase_num_active_features(self) -> None:
        self.n_active_features = min(self.max_n_features, self.n_active_features + self.feature_dim_increase_step)

    def get_active_feature_mask(self) -> torch.Tensor:
        if self.feature_type == 'sh':
            current_sh_degree = self.n_active_features
            max_sh_degree = self.max_n_features
            active_features = sh_degree_to_num_features(current_sh_degree)
            num_features = sh_degree_to_num_features(max_sh_degree)
        else:
            active_features = self.n_active_features
            num_features = self.max_n_features
        mask = torch.zeros((1, num_features), device=self.device, dtype=self.get_features().dtype)
        mask[0,:active_features] = 1.0
        return mask

    def decay_density(self):
        decayed_densities = inverse_sigmoid(self.get_density() * self.conf.model.density_decay.gamma)
        optimizable_tensors = self.replace_tensor_to_optimizer(decayed_densities, "density")
        self.density = optimizable_tensors["density"]

    def reset_density(self):
        updated_densities = inverse_sigmoid(torch.min(self.get_density(), torch.ones_like(self.density) * self.new_max_density))
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

    def reset_rolling_buffers(self):
        self.rolling_error = torch.zeros_like(self.density)
        self.rolling_weight_contrib = torch.zeros_like(self.density)

    def replace_tensor_to_optimizer(self, tensor, name: str):
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = torch.nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def prune_optimizer_tensors(self, mask):
        assert self.optimizer is not None, "Optimizer need to be initialized before concatenating the values"
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != "background":
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = torch.nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = torch.nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
    
        return optimizable_tensors
    

    def concatenate_optimizer_tensors(self, tensors_dict):
        assert self.optimizer is not None, "Optimizer need to be initialized before concatenating the values"

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(group["params"][0].requires_grad))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(group["params"][0].requires_grad))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def init_densification_buffer(self, checkpoint:Optional[dict] = None):
        if checkpoint is not None:
            if self.conf.model.log_rolling_buffers:
                self.rolling_error = checkpoint["rolling_error"]
                self.rolling_weight_contrib = checkpoint["rolling_weight_contrib"]

            if self.conf.model.densify.method == 'gradient-buffer':
                self.positional_grad_norm_accum = checkpoint["positional_grad_norm_accum"]
                self.positional_grad_norm_denom = checkpoint["positional_grad_norm_denom"]
        else: 
            if self.conf.model.log_rolling_buffers:
                num_gaussians = self.positions.shape[0]
                self.rolling_error = torch.zeros((num_gaussians,1), dtype=torch.float, device=self.device)
                self.rolling_weight_contrib = torch.zeros((num_gaussians,1), dtype=torch.float, device=self.device)

            if self.conf.model.densify.method == 'gradient-buffer':
                num_gaussians = self.positions.shape[0]
                self.positional_grad_norm_accum = torch.zeros((num_gaussians,1), dtype=torch.float, device=self.device)
                self.positional_grad_norm_denom = torch.zeros((num_gaussians,1), dtype=torch.int, device=self.device)

    @torch.cuda.nvtx.range("update-gradient-buffer")
    def update_gradient_buffer(self, rays_ori, rays_dir):
        assert self.conf.model.densify.method == 'gradient-buffer'
        hit_cts = self.get_hit_counts(rays_ori, rays_dir)
        mask = (hit_cts > 0).squeeze()

        assert self.positions.grad is not None
        self.positional_grad_norm_accum[mask] += torch.norm(self.positions.grad[mask], dim=-1, keepdim=True)
        self.positional_grad_norm_denom[mask] += 1

    @torch.cuda.nvtx.range("update_rolling_buffers")
    def update_rolling_buffers(self, gaussian_errors, gaussian_weights):
        assert self.conf.model.log_rolling_buffers
        self.rolling_error += gaussian_errors
        self.rolling_weight_contrib += gaussian_weights

    @torch.cuda.nvtx.range("densify_gaussians")
    def densify_gaussians(self, scene_extent):
        if self.conf.model.densify.method in ["adam", "gradient-buffer"]:
            self.densify_positional_grad(scene_extent)
        elif self.conf.model.densify.method == "error":
            self.clone_gaussians_error()

    def densify_positional_grad(self, scene_extent):
        assert self.optimizer is not None, "Optimizer need to be initialized before splitting and cloning the Gaussians"
        assert self.get_positions().requires_grad, "Trying to perform split and clone but the positions are not being optimized"

        positional_grad_norm = None
        match self.conf.model.densify.method:
            case 'adam':
                # TODO: we directly use the gradient of the 3D positions, whereas 3D Gaussian Splatting uses the 2D position gradient after projection (in theory this should be the same as not projecting the gradient)
                # TODO: we always consider all the Gaussians by tapping into the Adam's exponential moving average. 3DGS only considers Gaussians that survived the frustum culling in the last iteration.
                for group in self.optimizer.param_groups:
                    if group["name"] == "positions":
                        positional_grad_norm = torch.norm(self.optimizer.state.get(group['params'][0], None)["exp_avg"], dim=-1)

            case 'gradient-buffer':
                # gsplat implementation
                positional_grad_norm = self.positional_grad_norm_accum / self.positional_grad_norm_denom
                positional_grad_norm[positional_grad_norm.isnan()] = 0.0 
            case _:
                raise ValueError(f"densify.method {self.conf.model.densify.method} not supported")

        assert positional_grad_norm is not None, "Was not able to retrieve the exp average of the positional gradient from Adam"       

        self.clone_gaussians(positional_grad_norm.squeeze(), scene_extent)
        self.split_gaussians(positional_grad_norm.squeeze(), scene_extent)      

        torch.cuda.empty_cache()

    @torch.cuda.nvtx.range("densify_postfix")
    def densify_postfix(self, add_gaussians):
        # Concatenate new tensors to the optimizer variables 
        optimizable_tensors = self.concatenate_optimizer_tensors(add_gaussians)
        self.update_optimizable_parameters(optimizable_tensors)

        if self.conf.model.log_rolling_buffers:
            self.reset_rolling_buffers()

        if self.conf.model.densify.method == 'gradient-buffer':
            self.positional_grad_norm_accum = torch.zeros((self.get_positions().shape[0], 1), 
                                                        device=self.device, 
                                                        dtype=self.positional_grad_norm_accum.dtype)
            self.positional_grad_norm_denom = torch.zeros((self.get_positions().shape[0], 1), 
                                                        device=self.device, 
                                                        dtype=self.positional_grad_norm_denom.dtype)
        
    @torch.cuda.nvtx.range("split_gaussians")
    def split_gaussians(self, positional_grad_norm: torch.Tensor, scene_extent: float):
        n_init_points = self.get_positions().shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        # Here we already have the cloned points in the self.positions so only take the points up to size of the initial grad
        padded_grad[:positional_grad_norm.shape[0]] = positional_grad_norm.squeeze()
        mask = torch.where(padded_grad >= self.split_grad_threshold, True, False)
        mask = torch.logical_and(mask, torch.max(self.get_scale(), dim=1).values > self.relative_size_threshold * scene_extent)


        stds = self.get_scale()[mask].repeat(self.split_n_gaussians, 1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.rotation[mask]).repeat(self.split_n_gaussians,1,1)

        add_gaussians = {
                "positions": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions()[mask].repeat(self.split_n_gaussians, 1),
                "density":  self.density[mask].repeat(self.split_n_gaussians,1),
                "scale":  get_activation_function(self.conf.model.scale_activation, inverse=True)(self.get_scale()[mask].repeat(self.split_n_gaussians,1) / (0.8*self.split_n_gaussians)),
                "rotation": self.rotation[mask].repeat(self.split_n_gaussians,1)
                }
        if self.feature_type == 'sh':
            add_gaussians["features_albedo"] = self.features_albedo[mask].repeat(self.split_n_gaussians, 1)
            add_gaussians["features_specular"] = self.features_specular[mask].repeat(self.split_n_gaussians, 1)

        self.densify_postfix(add_gaussians)
        
        # stats
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f"Splitted {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")
        

        # Prune away the Gaussians that were originally slected
        valid = ~torch.cat((mask, torch.zeros(self.split_n_gaussians * mask.sum(), device="cuda", dtype=bool)))
        self.prune_gaussians(valid)


    @torch.cuda.nvtx.range("clone_gaussians")
    def clone_gaussians(self, positional_grad_norm: torch.Tensor, scene_extent: float):
        assert positional_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(positional_grad_norm >= self.clone_grad_threshold, True, False)

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(mask, torch.max(self.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent)

        # stats
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f"Cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")
        
        # Use the mask to dupicate these points
        add_gaussians = {
                "positions": self.positions[mask],
                "density": self.density[mask],
                "scale": self.scale[mask],
                "rotation": self.rotation[mask]}
        if self.feature_type == 'sh':
            add_gaussians["features_albedo"] = self.features_albedo[mask]
            add_gaussians["features_specular"] = self.features_specular[mask]

        self.densify_postfix(add_gaussians)
    
    @torch.cuda.nvtx.range("clone_gaussians_error")
    def clone_gaussians_error(self):
        assert self.conf.model.log_rolling_buffers

        n_split = 1

        # Clone points with high error ratio
        # This roughly correspond to "what fraction of what this ray contributed was error"
        error_ratio = self.rolling_error[:,0] / self.rolling_weight_contrib[:,0]
        error_ratio_mask = error_ratio > self.conf.model.error_densify.error_ratio_threshold

        mask = error_ratio_mask
        
        # stats
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_clone = mask.sum()
            print(f"Error-cloned {n_clone} / {n_before} ({n_clone/n_before*100:.2f}%) gaussians")

        # sample new locations according to the shape of the sourcea gaussian
        stds = self.get_scale()[mask]
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_so3(self.rotation[mask]).repeat(n_split,1,1)
        new_pos = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_positions()[mask]

        add_gaussians = {
                "positions": new_pos,
                "density":  self.density[mask],
                "scale":  self.scale[mask, :],
                "rotation": self.rotation[mask,:]
            }

        if self.feature_type == 'sh':
            add_gaussians["features_albedo"] = self.features_albedo[mask]
            add_gaussians["features_specular"] = self.features_specular[mask]

        self.densify_postfix(add_gaussians)

    def prune_gaussians_weight(self):
        # Prune the Gaussians based on their weight
        mask = self.rolling_weight_contrib[:,0] >= self.conf.model.prune_weight.weight_threshold
        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            print(f"Weight-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.prune_gaussians(mask)
    
    def prune_gaussians_opacity(self):
        # Prune the Gaussians based on their opacity
        mask = self.get_density().squeeze() >= self.prune_density_threshold

        if self.conf.model.print_stats:
            n_before = mask.shape[0]
            n_prune = n_before - mask.sum()
            print(f"Density-pruned {n_prune} / {n_before} ({n_prune/n_before*100:.2f}%) gaussians")

        self.prune_gaussians(mask)

    def prune_needles(self):
        # Prune small needle gaussians 
        mask = torch.min(self.get_scale(),dim=1).values >= self.conf.model.prune_needles.min_scale_threshold
        self.prune_gaussians(mask.squeeze())

    @torch.cuda.nvtx.range("prune_gaussians")
    def prune_gaussians(self, valid_mask):
        # TODO: consider having a buffer of the contribution of Gaussians to the rendering -> this might avoid the need to reset opacity
        # TODO: we could also consider pruning away some of the large Gaussians?
        optimizable_tensors = self.prune_optimizer_tensors(valid_mask)
        self.update_optimizable_parameters(optimizable_tensors)

        if self.conf.model.densify.method == 'gradient-buffer':
            self.positional_grad_norm_accum = self.positional_grad_norm_accum[valid_mask]
            self.positional_grad_norm_denom = self.positional_grad_norm_denom[valid_mask]

        if self.conf.model.log_rolling_buffers:
            # TODO only touch these based on a densify method?
            self.rolling_error = self.rolling_error[valid_mask]
            self.rolling_weight_contrib = self.rolling_weight_contrib[valid_mask]

        torch.cuda.empty_cache()

    def prune_sky_gaussians(self, sky_rays, hit_count_threshold=0):
        # Iterate over the sky rays and prune away gaussians along them
        num_deleted_sky = 0
        hit_buffer = torch.zeros((self.positions.shape[0],),device=self.device)
        for semantic_rays in tqdm.tqdm(
            sky_rays,
            desc="Pruning the Gaussians with sky rays",
        ):
            ray_o = semantic_rays[:, :3].to(self.device)
            ray_d = semantic_rays[:, 3:6].to(self.device)
            ray_d /= torch.linalg.norm(ray_d, dim=1, keepdim=True)
            
            # Prune Gaussians along ray
            hit_buffer += self.get_hit_counts(rays_o=ray_o[None,...],rays_d=ray_d[None,...]).squeeze()
        mask = (hit_buffer <= hit_count_threshold).squeeze()
        self.delete_hit_gaussians(mask)
        num_deleted_sky += (~mask).sum()
        print(f"culled {num_deleted_sky} using sky rays...")

    def get_scale(self, preactivation=False):
        if preactivation:
           return self.scale
        else:
           return self.scale_activation(self.scale)
    
    def get_rotation(self, preactivation=False):
        if preactivation:
           return self.rotation
        else:
           return self.rotation_activation(self.rotation)

    def get_positions(self, preactivation=False):
        return self.positions
    
    def get_features(self, preactivation=False):
        return torch.cat((self.features_albedo, self.features_specular), dim=1)
    
    def get_density(self, preactivation=False):
        if preactivation:
           return self.density
        else:
           return self.density_activation(self.density)

    def get_model_parameters(self) -> dict:
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"

        model_params = {
            "positions": self.positions,
            "rotation": self.rotation,
            "scale": self.scale,
            "density": self.density,
            "background": self.background.state_dict(),
            # Add other attributes that we need at restore
            "n_active_features": self.n_active_features,
            "max_n_features": self.max_n_features,
            "progressive_training": self.progressive_training,
            "scene_extent": self.scene_extent,

            # Add optimizer state dict
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf
        }

        if self.progressive_training:
            model_params["feature_dim_increase_interval"] = self.feature_dim_increase_interval
            model_params["feature_dim_increase_step"] = self.feature_dim_increase_step

        if self.feature_type == 'sh':
            model_params["features_albedo"] = self.features_albedo
            model_params["features_specular"] = self.features_specular

        if self.conf.model.densify.method == 'gradient-buffer':
            model_params["positional_grad_norm_accum"] = self.positional_grad_norm_accum,
            model_params["positional_grad_norm_denom"] = self.positional_grad_norm_denom,
        
        return model_params

    @torch.no_grad()
    def get_hit_counts(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        mog_counts = optixtracer.count_mog_hits(
            self.optix_ctx, 
            rays_o, 
            rays_d, 
            self.positions, 
            self.get_rotation(), 
            self.get_scale(),
            self.get_density()
        )
        return mog_counts

    def forward_optix_render(self, rays_o: torch.Tensor, rays_d: torch.Tensor, err_target, train: bool) -> dict[str, torch.Tensor]:

        if err_target is None:
            err_target = torch.ones_like(rays_o[:,0])

        num_gsplats = self.positions.shape[0]
        with torch.cuda.nvtx.range(f"model.forward({num_gsplats} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            features = self.get_features()
            if self.progressive_training:
                features *= self.get_active_feature_mask()

            # For spherical harmonics, for optimal performance we set the kernel
            # to avoid computing sh degrees which are not yet used by the model
            if self.feature_type == 'sh':
                self.optix_ctx.set_sph_degree(self.n_active_features)

            pred_rgb, pred_opacity, pred_dist, g_weights, err_backprop_proxy = optixtracer.trace_mog(
                    self.optix_ctx, rays_o, rays_d,
                    self.positions, self.get_rotation(), self.get_scale(),
                    self.get_density(), features, err_target)

            pred_rgb, pred_opacity = self.background(rays_d, pred_rgb, pred_opacity, train)

        return {
            'pred_rgb': pred_rgb,
            'pred_opacity': pred_opacity,
            'pred_dist': pred_dist,
            'g_weights': g_weights,
            'err_backprop_proxy': err_backprop_proxy,
        }
    
    def forward_torch_render(self, rays_o: torch.Tensor, rays_d: torch.Tensor, err_target: torch.Tensor, train: bool) -> dict[str, torch.Tensor]:

        ## Use the optix raycaster to get a list of hit indices
        num_gsplats = self.positions.shape[0]
        with torch.cuda.nvtx.range(f"model.forward_hitinds({num_gsplats} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            features = self.get_features()
            if self.progressive_training:
                features *= self.get_active_feature_mask()

            # Gather data
            gpos = self.positions
            grot = self.get_rotation()
            gscl = self.get_scale()
            gdns = self.get_density()
            gsh  = features

            # Get the hit indices
            dense_hit_gIds, = optixtracer.trace_mog_inds(
                    self.optix_ctx, rays_o, rays_d,
                    self.positions, self.get_rotation(), self.get_scale(),
                    self.get_density())

        ## Evaluate the render pass
        with torch.cuda.nvtx.range(f"model.forward_rendereval({num_gsplats} gaussians)"):
            ray_rgb, ray_opacity, ray_ohit, ray_dist, g_weights, err_backprop_proxy = evaluate_rays(
                dense_hit_gIds,
                rays_o,
                rays_d,
                gpos,
                grot,
                gscl,
                gdns,
                gsh,
                err_target,
                self.max_n_features,
                self.conf.render.torch,
            )

            ray_rgb, ray_opacity = self.background(rays_d, ray_rgb, ray_opacity, train)
        return {
            'pred_rgb': ray_rgb,
            'pred_opacity': ray_opacity,
            'pred_ohit': ray_ohit,
            'pred_dist': ray_dist,
            'g_weights': g_weights,
            'err_backprop_proxy': err_backprop_proxy,
        }

    
    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor, err_target=None, force_method=None, train=False) -> dict[str, torch.Tensor]:
        """
        err_target is a "dummy" input used to implement rolling error accumulation
        """
        
        if err_target is None:
            err_target = torch.ones_like(self.density)

        if force_method is None:
            force_method = self.render_method

        if force_method == 'optix':
            return self.forward_optix_render(rays_o, rays_d, err_target, train)

        elif force_method == 'torch':
            return self.forward_torch_render(rays_o, rays_d, err_target, train)

        else:
            raise ValueError(f"unrecognized render method {self.render_method}")


