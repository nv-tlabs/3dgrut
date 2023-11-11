import os
import numpy as np
import torch
import pickle
from collections import defaultdict
from pathlib import Path


class TrainingRecorder:
    """ For collecting detailed information during training """

    RECORDINGS_FOLDER = 'extra_info'

    def __init__(self):
        self.train_info_dict = defaultdict(list)
        self.valid_info_dict = defaultdict(list)
        os.makedirs(TrainingRecorder.RECORDINGS_FOLDER, exist_ok=True)

    def record_metrics(self, iteration: int, psnr, loss):
        self.valid_info_dict['iteration'].append(iteration)
        self.valid_info_dict['psnr'].append(psnr)
        self.valid_info_dict['psnr_mean'].append(np.mean(psnr))
        self.valid_info_dict['loss'].append(loss)
        self.valid_info_dict['loss_mean'].append(np.mean(loss))

    def record_train_step(self, model, iteration: int, iteration_time: int,
                          l1_loss: torch.Tensor, ssim_loss: torch.Tensor, total_loss: torch.Tensor, psnr: int):
        if not (iteration > 0 and iteration % 100 == 0):
            return
        num_gaussians = model.positions.shape[0]

        self.train_info_dict['iteration'].append(iteration)
        self.train_info_dict['iter_time'].append(iteration_time)
        self.train_info_dict['num_gaussians'].append(num_gaussians)
        self.train_info_dict['active_sh_deg'].append(model.n_active_features)

        self.train_info_dict['l1_loss'].append(l1_loss.item())
        self.train_info_dict['ssim_loss'].append(ssim_loss.item())
        self.train_info_dict['total_loss'].append(total_loss.item())
        self.train_info_dict['psnr'].append(psnr)

        self.train_info_dict['opacity_mean'].append(model.get_density().mean().item())
        self.train_info_dict['opacity_std'].append(model.get_density().std().item())

        self.train_info_dict['features_albedo_mean'].append(model.features_albedo.mean().item())
        self.train_info_dict['features_albedo_R_mean'].append(model.features_albedo.mean(dim=0)[0].item())
        self.train_info_dict['features_albedo_G_mean'].append(model.features_albedo.mean(dim=0)[1].item())
        self.train_info_dict['features_albedo_B_mean'].append(model.features_albedo.mean(dim=0)[2].item())
        self.train_info_dict['features_albedo_std'].append(model.features_albedo.std().item())
        self.train_info_dict['features_albedo_R_std'].append(model.features_albedo.std(dim=0)[0].item())
        self.train_info_dict['features_albedo_G_std'].append(model.features_albedo.std(dim=0)[1].item())
        self.train_info_dict['features_albedo_B_std'].append(model.features_albedo.std(dim=0)[2].item())

        features_specular = model.features_specular.reshape(num_gaussians, -1, 3)
        self.train_info_dict['features_specular_mean'].append(features_specular.mean().item())
        self.train_info_dict['features_specular_R_mean'].append(features_specular.mean(dim=(0, 1))[0].item())
        self.train_info_dict['features_specular_G_mean'].append(features_specular.mean(dim=(0, 1))[1].item())
        self.train_info_dict['features_specular_B_mean'].append(features_specular.mean(dim=(0, 1))[2].item())
        self.train_info_dict['features_specular_std'].append(features_specular.std().item())
        self.train_info_dict['features_specular_R_std'].append(features_specular.std(dim=(0, 1))[0].item())
        self.train_info_dict['features_specular_G_std'].append(features_specular.std(dim=(0, 1))[1].item())
        self.train_info_dict['features_specular_B_std'].append(features_specular.std(dim=(0, 1))[2].item())

        self.train_info_dict['pos_x_mean'].append(model.get_positions().mean(dim=0)[0].item())
        self.train_info_dict['pos_y_mean'].append(model.get_positions().mean(dim=0)[1].item())
        self.train_info_dict['pos_z_mean'].append(model.get_positions().mean(dim=0)[2].item())
        self.train_info_dict['pos_x_std'].append(model.get_positions().std(dim=0)[0].item())
        self.train_info_dict['pos_y_std'].append(model.get_positions().std(dim=0)[1].item())
        self.train_info_dict['pos_z_std'].append(model.get_positions().std(dim=0)[2].item())

        self.train_info_dict['scale_x_mean'].append(model.get_scale().mean(dim=0)[0].item())
        self.train_info_dict['scale_y_mean'].append(model.get_scale().mean(dim=0)[1].item())
        self.train_info_dict['scale_z_mean'].append(model.get_scale().mean(dim=0)[2].item())
        self.train_info_dict['scale_x_std'].append(model.get_scale().std(dim=0)[0].item())
        self.train_info_dict['scale_y_std'].append(model.get_scale().std(dim=0)[1].item())
        self.train_info_dict['scale_z_std'].append(model.get_scale().std(dim=0)[2].item())

        self.train_info_dict['rot_x_mean'].append(model.get_rotation().mean(dim=0)[0].item())
        self.train_info_dict['rot_y_mean'].append(model.get_rotation().mean(dim=0)[1].item())
        self.train_info_dict['rot_z_mean'].append(model.get_rotation().mean(dim=0)[2].item())
        self.train_info_dict['rot_w_mean'].append(model.get_rotation().mean(dim=0)[3].item())
        self.train_info_dict['rot_x_std'].append(model.get_rotation().std(dim=0)[0].item())
        self.train_info_dict['rot_y_std'].append(model.get_rotation().std(dim=0)[1].item())
        self.train_info_dict['rot_z_std'].append(model.get_rotation().std(dim=0)[2].item())
        self.train_info_dict['rot_w_std'].append(model.get_rotation().std(dim=0)[3].item())

    def get_gaussians_info(self, gaussians):
        num_gaussians = gaussians.get_positions().shape[0]
        features = gaussians.get_features().reshape(num_gaussians, -1, 3)
        with torch.no_grad():
            data = dict(
                opacity = gaussians.get_density().squeeze(dim=-1).cpu().numpy(),
                pos = gaussians.get_positions().cpu().numpy(),
                rotations = gaussians.get_rotation().cpu().numpy(),
                scale = gaussians.get_scale(False).cpu().numpy(),
                log_scale = gaussians.get_scale(True).cpu().numpy(),
                sh_dc_response = features[:, 0].cpu().numpy(),
                sh_deg1_response = features[:, 1:4].cpu().numpy(),
                sh_deg2_response = features[:, 4:9].cpu().numpy(),
                sh_deg3_response = features[:, 9:].cpu().numpy(),
            )
        return data

    def get_scene_info(self, dataset, scene_extent):
        num_of_cameras = None
        try:
            num_of_cameras = dataset.poses.shape[0]
        except:
            pass

        scene_extents = None
        try:
            scene_extents = scene_extent[0].cpu().numpy(), scene_extent[1].cpu().numpy()
        except:
            pass

        scene_info = dict(
            cameras_extent=scene_extents,
            num_train_cameras=num_of_cameras
        )
        return scene_info

    def submit_recording(self, dataset, scene_extent, train_path: str, model):
        recorded_output = dict(
            scene_info = self.get_scene_info(dataset, scene_extent),
            gaussians=self.get_gaussians_info(model),
            extra_dict=self.train_info_dict
        )
        object_name = Path(train_path).stem
        filename = f'{TrainingRecorder.RECORDINGS_FOLDER}/{object_name}.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(recorded_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
