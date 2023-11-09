import logging, os

import numpy as np
import torch
from plyfile import PlyData

from libs import optixtracer
from utils import to_torch, get_activation_function, inverse_sigmoid, get_scheduler, quaternion_to_so3, \
    sh_degree_to_num_features, sh_degree_to_specular_dim
from datasets.colmap_utils import read_next_bytes
from datasets.utils import PointCloud
from geometry import nearest_neighbor_dist_cpuKD
from utils import to_np

class MixtureOfGaussians(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.positions = torch.empty([0,3])  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation  = torch.empty([0,4])   # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale     = torch.empty([0,3])     # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density   = torch.empty([0,1])    # Density of each Gaussian [n_gaussians, 1]
        self.features_albedo  = torch.empty([0,3])  # Feature vector of the 0th order SH coefficients [n_gaussians, 3] (We split it into two due to different learning rates)
        self.features_specular  = torch.empty([0,1]) # Features of the higher order SH coefficients [n_gaussians, 3]
        self.device = 'cuda'
        self.optimizer = None
        self.optix_ctx = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.density_activation_inv = get_activation_function(self.conf.model.density_activation, inverse=True)
        self.scale_activation =  get_activation_function(self.conf.model.scale_activation)
        self.scale_activation_inv =  get_activation_function(self.conf.model.scale_activation, inverse=True)
        self.rotation_activation =   get_activation_function("normalize") # The default value of the dim parameter is 1

        # How many degrees of spherical harmonics to use.
        # features tensor must contain matching number of sh weights
        self.sh_degs_to_calculate = 0

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
                sph_degree = self.conf.render.sph_degree,
                gaussian_sigma_threshold = self.conf.render.gaussian_sigma_threshold,
                min_transmittance = self.conf.render.min_transmittance,
            )
        )

    def init_from_colmap(self, root_path: str):
        # TODO this reads from the binary format, also implement the nearly-identical plaintext version?
        points_file = os.path.join(root_path, "sparse/0", "points3D.bin")
        if not os.path.isfile(points_file):
            raise ValueError(f"colomap points file {points_file} not found")

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

        self.default_initialize_from_points(file_pts, file_rgb)
        self.validate_fields()

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
        self.features_specular = torch.nn.Parameter(to_torch(np.transpose(np.stack((
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
            dtype=np.float32)), device=self.device))  # type: ignore

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    @torch.no_grad()
    def init_from_random_point_cloud(self,
                                     num_gsplat: int = 100_000,
                                     dtype=torch.float32,
                                     set_optimizable_parameters: bool = True):

        logging.info(f"Generating random point cloud ({num_gsplat})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz in [-1.3, 1.3]
        fused_point_cloud = torch.rand((num_gsplat, 3), dtype=dtype, device=self.device) * 2.6 - 1.3
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

        if set_optimizable_parameters:
            self.set_optimizable_parameters()
        self.validate_fields()

    def init_from_checkpoint(self, checkpoint: dict):
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features_albedo = checkpoint["features_albedo"]
        self.features_specular = checkpoint["features_specular"]
        self.n_active_features = checkpoint["n_active_features"]
        self.max_n_features = checkpoint["max_n_features"]
        self.set_optimizable_parameters()
        self.setup_optimizer(state_dict=checkpoint['optimizer'])
        self.validate_fields()

    def default_initialize_from_points(self, pts_np, colors_np=None):
        """
        Given an Nx3 array of points (and optionally Nx3 rgb colors), 
        initialize default values for the other parameters of the model
        """
                              
        dtype = torch.float32

        N = pts_np.shape[0]
        positions = to_torch(pts_np, dtype=dtype, device=self.device)
       
        # identity rotations
        rots = torch.zeros((N,4), dtype=dtype, device=self.device)
        rots[:,0] = 1. # they're quaternions

        # set the scale as function of the distance from the origin
        # TODO this might not make sense for large-scale scenes
        dist = torch.clamp_min(nearest_neighbor_dist_cpuKD(positions), 1e-3)
        scales = torch.log(dist)[..., None].repeat(1, 3)

        # set density as a constant
        opacities = self.density_activation_inv(0.1 * torch.ones((N,1), dtype=dtype, device=self.device))

        # set colors, constant if they weren't given
        if colors_np is None:
            features_albedo = 0.5 * torch.ones((N, 3), dtype=dtype, device=self.device)
        else:
            features_albedo = to_torch(colors_np, dtype=dtype, device=self.device)
        num_specular_dims = sh_degree_to_specular_dim(self.max_n_features)
        features_specular = torch.zeros((N, num_specular_dims))

        self.positions = torch.nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features_albedo = torch.nn.Parameter(features_albedo.to(dtype=dtype, device=self.device))
        self.features_specular = torch.nn.Parameter(features_specular.to(dtype=dtype, device=self.device))

        self.set_optimizable_parameters()
        self.setup_optimizer()
        self.validate_fields()
            
    def init_from_lidar(self, 
                        point_cloud : PointCloud, 
                        ):
        
        logging.info(f"Initializing based on lidar point cloud ...")
       
        # only initialize by default from points for now
        self.default_initialize_from_points(to_np(point_cloud.xyz_end), None)

    def setup_optimizer(self, state_dict=None):
        params = []
        for name, args in self.conf.optimizer.params.items():
            module =  getattr(self, name)
            if isinstance(module, torch.nn.Module):
                module_parameters = filter(lambda p: p.requires_grad, module.parameters())
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

    def reset_density(self):
        updated_densities = inverse_sigmoid(torch.min(self.get_density(), torch.ones_like(self.density) * self.new_max_density))
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self.density = optimizable_tensors["density"]

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


    def densify_gaussians(self, scene_extent):
        assert self.optimizer is not None, "Optimizer need to be initialized before splitting and cloning the Gaussians"
        assert self.get_positions().requires_grad, "Trying to perform split and clone but the positions are not being optimized"

        positional_grad_norm = None
        # TODO: we directly use the gradient of the 3D positions, whereas 3D Gaussian Splatting uses the 2D position gradient after projection (in theory this should be the same as not projecting the gradient)
        # TODO: we always consider all the Gaussians by tapping into the Adam's exponential moving average. 3DGS only considers Gaussians that survived the frustum culling in the last iteration.
        for group in self.optimizer.param_groups:
            if group["name"] == "positions":
                positional_grad_norm = torch.norm(self.optimizer.state.get(group['params'][0], None)["exp_avg"], dim=-1)

        assert positional_grad_norm is not None, "Was not able to retrieve the exp average of the positional gradient from Adam"

        self.clone_gaussians(positional_grad_norm, scene_extent)
        self.split_gaussians(positional_grad_norm, scene_extent)      

        torch.cuda.empty_cache()

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

        # Concatenate new tensors to the optimizer variables 
        optimizable_tensors = self.concatenate_optimizer_tensors(add_gaussians)
        self.update_optimizable_parameters(optimizable_tensors)

    def clone_gaussians(self, positional_grad_norm: torch.Tensor, scene_extent: float):
        assert positional_grad_norm is not None, "Positional gradients must be available in order to clone the Gaussians"
        # Extract points that satisfy the gradient condition
        mask = torch.where(positional_grad_norm >= self.clone_grad_threshold, True, False)

        # If the gaussians are larger they shouldn't be cloned, but rather split
        mask = torch.logical_and(mask, torch.max(self.get_scale(), dim=1).values <= self.relative_size_threshold * scene_extent)

        # Use the mask to dupicate these points
        add_gaussians = {
                "positions": self.positions[mask],
                "density": self.density[mask],
                "scale": self.scale[mask],
                "rotation": self.rotation[mask]}
        if self.feature_type == 'sh':
            add_gaussians["features_albedo"] = self.features_albedo[mask]
            add_gaussians["features_specular"] = self.features_specular[mask]

        optimizable_tensors = self.concatenate_optimizer_tensors(add_gaussians)
        self.update_optimizable_parameters(optimizable_tensors)

    def prune_gaussians(self):

        # Prune the Gaussians based on their opacity
        # TODO: consider having a buffer of the contribution of Gaussians to the rendering -> this might avoid the need to reset opacity
        # TODO: we could also consider pruning away some of the large Gaussians?
        mask = self.get_density().squeeze() >= self.prune_density_threshold

        optimizable_tensors = self.prune_optimizer_tensors(mask)
        self.update_optimizable_parameters(optimizable_tensors)

        torch.cuda.empty_cache()

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
            # Add other attributes that we need at restore
            "n_active_features": self.n_active_features,
            "max_n_features": self.max_n_features,
            # Add optimizer state dict
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf
        }
        if self.feature_type == 'sh':
            model_params["features_albedo"] = self.features_albedo
            model_params["features_specular"] = self.features_specular
        return model_params

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> dict[str, torch.Tensor]:
        
        features = self.get_features()
        if self.progressive_training:
            features *= self.get_active_feature_mask()

        pred_rgb, pred_opacity, pred_ohit = optixtracer.trace_mog(
                self.optix_ctx, rays_o, rays_d, 
                self.positions, self.get_rotation(), self.get_scale(),
                self.get_density(), features)

        return {
            'pred_rgb': pred_rgb,
            'pred_opacity': pred_opacity,
            'pred_ohit': pred_ohit
        }
