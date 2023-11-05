import logging
import numpy as np
import torch
from plyfile import PlyData
from utils import to_torch, get_activation_function, inverse_sigmoid
from libs import optixtracer


class MixtureOfGaussians(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        self.positions = torch.empty([0,3])  # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
        self.rotation = torch.empty([0,4])   # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
        self.scale  = torch.empty([0,3])     # Anisotropic scale of each Gaussian [n_gaussians, 3]
        self.density = torch.empty([0,1])    # Density of each Gaussian [n_gaussians, 1]
        self.features  = torch.empty([0,1])  # Feature vector associated to each Gaussian (can be RGB, SH or a latent code) [n_gaussians, n_features]
        self.n_active_features = 0
        self.device = 'cuda'
        self.optimizer = None
        self.optix_ctx = None
        self.density_activation = get_activation_function(self.conf.model.density_activation)
        self.scale_activation =  get_activation_function(self.conf.model.scale_activation)
        self.rotation_activation =   get_activation_function("normalize") # The default value of the dim parameter is 1

    def set_optix_context(self):
        torch.zeros(1, device=self.device) # Create a dummy tensor to force cuda context init
        self.optix_ctx = optixtracer.OptiXContext()

    def set_optimizable_parameters(self):
        if not self.conf.model.optimize_density:
            self.density.requires_grad = False
        if not self.conf.model.optimize_features:
            self.features.requires_grad = False
        if not self.conf.model.optimize_rotation:
            self.rotation.requires_grad = False
        if not self.conf.model.optimize_scale:
            self.scale.requires_grad = False
        if not self.conf.model.optimize_position:
            self.positions.requires_grad = False

    def init_from_checkpoint(self, checkpoint: dict):
        self.positions = checkpoint["positions"]
        self.rotation = checkpoint["rotation"]
        self.scale = checkpoint["scale"]
        self.density = checkpoint["density"]
        self.features = checkpoint["features"]
        self.n_active_features = checkpoint["active_features"]
        self.set_optimizable_parameters()
        self.setup_optimizer(state_dict=checkpoint['optimizer'])
            
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

        # When loading from the checkpoint also load the state dict
        if state_dict is not None:
            self.optimizer.load_state_dict(state_dict)

    def init_from_pretrained_point_cloud(self, pc_path: str, set_optimizable_parameters: bool = True):
        data = PlyData.read(pc_path)
        num_gsplat = len(data['vertex'])
        self.positions = torch.nn.Parameter(to_torch(np.transpose(np.stack((data['vertex']['x'], data['vertex']['y'], data['vertex']['z']), dtype=np.float32)), device=self.device))  # type: ignore
        self.rotation = torch.nn.Parameter(to_torch(np.transpose(np.stack((data['vertex']['rot_0'], data['vertex']['rot_1'], data['vertex']['rot_2'], data['vertex']['rot_3']), dtype=np.float32)), device=self.device))  # type: ignore
        self.scale = torch.nn.Parameter(to_torch(np.transpose(np.stack((data['vertex']['scale_0'], data['vertex']['scale_1'], data['vertex']['scale_2']), dtype=np.float32)), device=self.device))  # type: ignore
        self.density = torch.nn.Parameter(to_torch(data['vertex']['opacity'].astype(np.float32).reshape(num_gsplat, 1), device=self.device))
        self.features = torch.nn.Parameter(to_torch(np.transpose(np.stack((
                            data['vertex']['f_dc_0'], data['vertex']['f_dc_1'], data['vertex']['f_dc_2'],
                            data['vertex']['f_rest_0'],data['vertex']['f_rest_1'],data['vertex']['f_rest_2'],data['vertex']['f_rest_3'],data['vertex']['f_rest_4'],
                            data['vertex']['f_rest_5'],data['vertex']['f_rest_6'],data['vertex']['f_rest_7'],data['vertex']['f_rest_8'],data['vertex']['f_rest_9'],
                            data['vertex']['f_rest_10'],data['vertex']['f_rest_11'],data['vertex']['f_rest_12'],data['vertex']['f_rest_13'],data['vertex']['f_rest_14'],
                            data['vertex']['f_rest_15'],data['vertex']['f_rest_16'],data['vertex']['f_rest_17'],data['vertex']['f_rest_18'],data['vertex']['f_rest_19'],
                            data['vertex']['f_rest_20'],data['vertex']['f_rest_21'],data['vertex']['f_rest_22'],data['vertex']['f_rest_23'],data['vertex']['f_rest_24'],
                            data['vertex']['f_rest_25'],data['vertex']['f_rest_26'],data['vertex']['f_rest_27'],data['vertex']['f_rest_28'],data['vertex']['f_rest_29'],
                            data['vertex']['f_rest_30'],data['vertex']['f_rest_31'],data['vertex']['f_rest_32'],data['vertex']['f_rest_33'],data['vertex']['f_rest_34'],
                            data['vertex']['f_rest_35'],data['vertex']['f_rest_36'],data['vertex']['f_rest_37'],data['vertex']['f_rest_38'],data['vertex']['f_rest_39'],
                            data['vertex']['f_rest_40'],data['vertex']['f_rest_41'],data['vertex']['f_rest_42'],data['vertex']['f_rest_43'],data['vertex']['f_rest_44']),
                            dtype=np.float32)), device=self.device)) # type: ignore

        if set_optimizable_parameters:
            self.set_optimizable_parameters()

    @torch.no_grad()
    def randomize_point_cloud(self,
                              num_gsplat: int = 100_000,
                              max_sh_degree: int = 3,
                              dtype = torch.float32,
                              set_optimizable_parameters: bool = True):
        try:
            from simple_knn._C import distCUDA2
        except ImportError as e:
            logging.error('Cannot import simple_knn. Please check README.md & make sure library is properly installed.')
            logging.error(e)
        logging.info(f"Generating random point cloud ({num_gsplat})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz in [-1.3, 1.3]
        fused_point_cloud = torch.rand((num_gsplat, 3), dtype=dtype, device=self.device) * 2.6 - 1.3
        # sh in [0, 0.0039]
        fused_color = torch.rand((num_gsplat, 3), dtype=dtype, device=self.device) / 255.0

        features = torch.zeros((num_gsplat, 3, (max_sh_degree + 1) ** 2), dtype=dtype, device=self.device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        features = features.transpose(1, 2).reshape(num_gsplat, -1).contiguous()

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((num_gsplat, 4), device=self.device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((num_gsplat, 1), dtype=dtype, device=self.device))

        self.positions = torch.nn.Parameter(fused_point_cloud)  # type: ignore
        self.rotation = torch.nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scale = torch.nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.density = torch.nn.Parameter(opacities.to(dtype=dtype, device=self.device))
        self.features = torch.nn.Parameter(features.to(dtype=dtype, device=self.device))

        if set_optimizable_parameters:
            self.set_optimizable_parameters()

    def build_bvh(self):
        optixtracer.build_mog_bvh(self.optix_ctx, self._positions, self.rotation_activation(self._rotation), self.scale_activation(self._scale), 3, True)

    def init_from_point_cloud(self, pc_path: str):
        pass

    def reset_density(self):
        updated_densities = inverse_sigmoid(torch.min(self.get_density, torch.ones_like(self.density)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(updated_densities, "density")
        self._density = optimizable_tensors["density"]


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



    @property
    def get_scale(self):
        return self.scale_activation(self.scale)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)
    
    @property
    def get_position(self):
        return self.positions
    
    @property
    def get_features(self):
        return self._features
    
    @property
    def get_density(self):
        return self.density_activation(self.density)
    

    def get_model_parameters(self) -> dict:
        assert self.optimizer is not None, "Optimizer need to be initialized when storing the checkpoint"

        return {
            "positions": self.positions,
            "rotation": self.rotation,
            "scale": self.scale,
            "density": self.density,
            "features": self.features,
            # Add other attributes that we need at restore
            "active_features": self.n_active_features,
            # Add optimizer state dict
            "optimizer": self.optimizer.state_dict(),
            "config": self.conf
        }

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor) -> dict[str, torch.Tensor]:
        pred_rgb, pred_opacity, pred_ohit = optixtracer.trace_mog(self.optix_ctx, rays_o, rays_d, self.positions, 
            self.rotation_activation(self.rotation), self.scale_activation(self.scale), self.density_activation(self.density), self.features)

        return {
            'pred_rgb': pred_rgb,
            'pred_opacity': pred_opacity,
            'pred_ohit': pred_ohit
        }
