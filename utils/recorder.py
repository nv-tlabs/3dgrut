import os
import numpy as np
import torch
import pickle
from collections import defaultdict
from pathlib import Path
from datetime import datetime


class TrainingRecorder:
    """ For collecting detailed information during training """

    RECORDINGS_FOLDER = 'extra_info'

    def __init__(self, enabled):
        self.enabled = enabled
        self.train_info_dict = defaultdict(list)
        self.valid_info_dict = defaultdict(list)
        self.train_recording_frequency = 100    # Record training state every N steps
        self._buffered_updates = False          # Contains updates not yet reported to tensorboard / wandb
        self.timestamp = self.get_timestamp()
        if enabled:
            os.makedirs(TrainingRecorder.RECORDINGS_FOLDER, exist_ok=True)

    @torch.cuda.nvtx.range("record_metrics")
    def record_metrics(self, iteration: int, psnr, loss):
        if not self.enabled:
            return  # nop
        self.valid_info_dict['iteration'].append(iteration)
        self.valid_info_dict['psnr'].append(psnr)
        self.valid_info_dict['psnr_mean'].append(np.mean(psnr))
        self.valid_info_dict['loss'].append(loss)
        self.valid_info_dict['loss_mean'].append(np.mean(loss))

    def is_time_to_record(self, iteration):
        return iteration > 0 and iteration % self.train_recording_frequency == 0

    @torch.cuda.nvtx.range("record_train_step")
    def record_train_step(self, model, iteration: int, iteration_time: int,
                          l1_loss: torch.Tensor, ssim_loss: torch.Tensor | None, total_loss: torch.Tensor, psnr: float):
        if not self.enabled or not self.is_time_to_record(iteration):
            return  # nop
        num_gaussians = model.positions.shape[0]

        self.train_info_dict['iteration'].append(iteration)
        self.train_info_dict['iter_time'].append(iteration_time)
        self.train_info_dict['num_gaussians'].append(num_gaussians)
        self.train_info_dict['active_sh_deg'].append(model.n_active_features)

        self.train_info_dict['l1_loss'].append(l1_loss.item())
        if ssim_loss is not None:
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

        try:
            self.train_info_dict['grad_pos_x_mean'].append(model.positions.grad.mean(dim=0)[0].item())
            self.train_info_dict['grad_pos_y_mean'].append(model.positions.grad.mean(dim=0)[1].item())
            self.train_info_dict['grad_pos_z_mean'].append(model.positions.grad.mean(dim=0)[2].item())
            self.train_info_dict['grad_pos_x_std'].append(model.positions.grad.std(dim=0)[0].item())
            self.train_info_dict['grad_pos_y_std'].append(model.positions.grad.std(dim=0)[1].item())
            self.train_info_dict['grad_pos_z_std'].append(model.positions.grad.std(dim=0)[2].item())
            self.train_info_dict['grad_pos_norm'].append(model.positions.grad.norm(dim=1).mean().item())
        except:
            pass
        
        try:
            self.train_info_dict['grad_scale_x_mean'].append(model.scale.grad.mean(dim=0)[0].item())
            self.train_info_dict['grad_scale_y_mean'].append(model.scale.grad.mean(dim=0)[1].item())
            self.train_info_dict['grad_scale_z_mean'].append(model.scale.grad.mean(dim=0)[2].item())
            self.train_info_dict['grad_scale_x_std'].append(model.scale.grad.std(dim=0)[0].item())
            self.train_info_dict['grad_scale_y_std'].append(model.scale.grad.std(dim=0)[1].item())
            self.train_info_dict['grad_scale_z_std'].append(model.scale.grad.std(dim=0)[2].item())
            self.train_info_dict['grad_scale_norm'].append(model.scale.grad.norm(dim=1).mean().item())
        except:
            pass

        try:
            self.train_info_dict['grad_rot_x_mean'].append(model.rotation.grad.mean(dim=0)[0].item())
            self.train_info_dict['grad_rot_y_mean'].append(model.rotation.grad.mean(dim=0)[1].item())
            self.train_info_dict['grad_rot_z_mean'].append(model.rotation.grad.mean(dim=0)[2].item())
            self.train_info_dict['grad_rot_w_mean'].append(model.rotation.grad.mean(dim=0)[3].item())
            self.train_info_dict['grad_rot_x_std'].append(model.rotation.grad.std(dim=0)[0].item())
            self.train_info_dict['grad_rot_y_std'].append(model.rotation.grad.std(dim=0)[1].item())
            self.train_info_dict['grad_rot_z_std'].append(model.rotation.grad.std(dim=0)[2].item())
            self.train_info_dict['grad_rot_w_std'].append(model.rotation.grad.std(dim=0)[3].item())
            self.train_info_dict['grad_rot_norm'].append(model.rotation.grad.norm(dim=1).mean().item())
        except:
            pass

        try:
            self.train_info_dict['grad_opacity_mean'].append(model.density.grad.mean().item())
            self.train_info_dict['grad_opacity_std'].append(model.density.grad.std().item())
            self.train_info_dict['grad_opacity_norm'].append(model.density.grad.norm(dim=1).mean().item())
        except:
            pass

        try:
            self.train_info_dict['grad_feature_albedo_R_mean'].append(model.features_albedo.grad.mean(dim=0)[0].item())
            self.train_info_dict['grad_feature_albedo_G_mean'].append(model.features_albedo.grad.mean(dim=0)[1].item())
            self.train_info_dict['grad_feature_albedo_B_mean'].append(model.features_albedo.grad.mean(dim=0)[2].item())
            self.train_info_dict['grad_feature_albedo_R_std'].append(model.features_albedo.grad.std(dim=0)[0].item())
            self.train_info_dict['grad_feature_albedo_G_std'].append(model.features_albedo.grad.std(dim=0)[1].item())
            self.train_info_dict['grad_feature_albedo_B_std'].append(model.features_albedo.grad.std(dim=0)[2].item())
            self.train_info_dict['grad_feature_albedo_norm'].append(model.features_albedo.grad.norm(dim=1).mean().item())
        except:
            pass
    
        self._buffered_updates = True

    @torch.cuda.nvtx.range("report_statistics")
    def report_statistics(self, writer):
        """ Reports last aggregated training statistics to experiments manager """
        if not self.enabled or not self._buffered_updates:
            return  # nop
        writer.add_scalar("iter_time", self.train_info_dict['iter_time'][-1])
        writer.add_scalar("g_statistics/num_gaussians", self.train_info_dict['num_gaussians'][-1])
        writer.add_scalar("g_statistics/active_sh_deg", self.train_info_dict['active_sh_deg'][-1])

        writer.add_scalar("g_statistics/density/opacity_mean", self.train_info_dict['opacity_mean'][-1])
        writer.add_scalar("g_statistics/density/opacity_std", self.train_info_dict['opacity_std'][-1])
        
        writer.add_scalar("g_statistics/scale/scale_x_mean", self.train_info_dict['scale_x_mean'][-1])
        writer.add_scalar("g_statistics/scale/scale_y_mean", self.train_info_dict['scale_y_mean'][-1])
        writer.add_scalar("g_statistics/scale/scale_z_mean", self.train_info_dict['scale_z_mean'][-1])

        writer.add_scalar("g_statistics/position/pos_x_mean", self.train_info_dict['pos_x_mean'][-1])
        writer.add_scalar("g_statistics/position/pos_y_mean", self.train_info_dict['pos_y_mean'][-1])
        writer.add_scalar("g_statistics/position/pos_z_mean", self.train_info_dict['pos_z_mean'][-1])
        writer.add_scalar("g_statistics/position/pos_x_std", self.train_info_dict['pos_x_std'][-1])
        writer.add_scalar("g_statistics/position/pos_y_std", self.train_info_dict['pos_y_std'][-1])
        writer.add_scalar("g_statistics/position/pos_z_std", self.train_info_dict['pos_z_std'][-1])

        writer.add_scalar("g_statistics/feature_albedo/features_albedo_mean", self.train_info_dict['features_albedo_mean'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_R_mean", self.train_info_dict['features_albedo_R_mean'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_G_mean", self.train_info_dict['features_albedo_G_mean'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_B_mean", self.train_info_dict['features_albedo_B_mean'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_std", self.train_info_dict['features_albedo_std'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_R_std", self.train_info_dict['features_albedo_R_std'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_G_std", self.train_info_dict['features_albedo_G_std'][-1])
        writer.add_scalar("g_statistics/feature_albedo/features_albedo_B_std", self.train_info_dict['features_albedo_B_std'][-1])

        writer.add_scalar("g_statistics/feature_specular/features_specular_mean", self.train_info_dict['features_specular_mean'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_R_mean", self.train_info_dict['features_specular_R_mean'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_G_mean", self.train_info_dict['features_specular_G_mean'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_B_mean", self.train_info_dict['features_specular_B_mean'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_std", self.train_info_dict['features_specular_std'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_R_std", self.train_info_dict['features_specular_R_std'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_G_std", self.train_info_dict['features_specular_G_std'][-1])
        writer.add_scalar("g_statistics/feature_specular/features_specular_B_std", self.train_info_dict['features_specular_B_std'][-1])

        writer.add_scalar("g_statistics/rotation/rot_x_mean", self.train_info_dict['rot_x_mean'][-1])
        writer.add_scalar("g_statistics/rotation/rot_y_mean", self.train_info_dict['rot_y_mean'][-1])
        writer.add_scalar("g_statistics/rotation/rot_z_mean", self.train_info_dict['rot_z_mean'][-1])
        writer.add_scalar("g_statistics/rotation/rot_w_mean", self.train_info_dict['rot_w_mean'][-1])
        writer.add_scalar("g_statistics/rotation/rot_x_std", self.train_info_dict['rot_x_std'][-1])
        writer.add_scalar("g_statistics/rotation/rot_y_std", self.train_info_dict['rot_y_std'][-1])
        writer.add_scalar("g_statistics/rotation/rot_z_std", self.train_info_dict['rot_z_std'][-1])
        writer.add_scalar("g_statistics/rotation/rot_w_std", self.train_info_dict['rot_w_std'][-1])

        try:
            writer.add_scalar("grad_statistics/position/grad_pos_x_mean", self.train_info_dict['grad_pos_x_mean'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_y_mean", self.train_info_dict['grad_pos_y_mean'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_z_mean", self.train_info_dict['grad_pos_z_mean'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_x_std", self.train_info_dict['grad_pos_x_std'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_y_std", self.train_info_dict['grad_pos_y_std'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_z_std", self.train_info_dict['grad_pos_z_std'][-1])
            writer.add_scalar("grad_statistics/position/grad_pos_norm", self.train_info_dict['grad_pos_norm'][-1])
        except:
            pass

        try:
            writer.add_scalar("grad_statistics/scale/grad_scale_x_mean", self.train_info_dict['grad_scale_x_mean'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_y_mean", self.train_info_dict['grad_scale_y_mean'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_z_mean", self.train_info_dict['grad_scale_z_mean'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_x_std", self.train_info_dict['grad_scale_x_std'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_y_std", self.train_info_dict['grad_scale_y_std'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_z_std", self.train_info_dict['grad_scale_z_std'][-1])
            writer.add_scalar("grad_statistics/scale/grad_scale_norm", self.train_info_dict['grad_scale_norm'][-1])
        except:
            pass

        try:
            writer.add_scalar("grad_statistics/rotation/grad_rot_x_mean", self.train_info_dict['grad_rot_x_mean'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_y_mean", self.train_info_dict['grad_rot_y_mean'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_z_mean", self.train_info_dict['grad_rot_z_mean'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_w_mean", self.train_info_dict['grad_rot_w_mean'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_x_std", self.train_info_dict['grad_rot_x_std'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_y_std", self.train_info_dict['grad_rot_y_std'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_z_std", self.train_info_dict['grad_rot_z_std'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_w_std", self.train_info_dict['grad_rot_w_std'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_w_std", self.train_info_dict['grad_rot_w_std'][-1])
            writer.add_scalar("grad_statistics/rotation/grad_rot_norm", self.train_info_dict['grad_rot_norm'][-1])
        except:
            pass

        try:
            writer.add_scalar("grad_statistics/opacity/grad_opacity_mean", self.train_info_dict['grad_opacity_mean'][-1])
            writer.add_scalar("grad_statistics/opacity/grad_opacity_std", self.train_info_dict['grad_opacity_std'][-1])
            writer.add_scalar("grad_statistics/opacity/grad_opacity_norm", self.train_info_dict['grad_opacity_norm'][-1])
        except:
            pass

        try:
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_R_mean", self.train_info_dict['grad_feature_albedo_R_mean'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_G_mean", self.train_info_dict['grad_feature_albedo_G_mean'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_B_mean", self.train_info_dict['grad_feature_albedo_B_mean'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_R_std", self.train_info_dict['grad_feature_albedo_R_std'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_G_std", self.train_info_dict['grad_feature_albedo_G_std'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_B_std", self.train_info_dict['grad_feature_albedo_B_std'][-1])
            writer.add_scalar("grad_statistics/feature_albedo/grad_albedo_norm", self.train_info_dict['grad_feature_albedo_norm'][-1])
        except:
            pass
        
        self._buffered_updates = False

    def _get_gaussians_info(self, gaussians):
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

    def _get_scene_info(self, dataset, scene_extent):
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

    @staticmethod
    def get_run_name(data_path):
        object_name = Path(data_path).stem
        return object_name

    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%d%m_%H%M%S")

    @torch.cuda.nvtx.range("submit_recording")
    def submit_recording(self, dataset, scene_extent, train_path: str, model):
        if not self.enabled:
            return  # nop
        recorded_output = dict(
            scene_info = self._get_scene_info(dataset, scene_extent),
            gaussians=self._get_gaussians_info(model),
            extra_dict=self.train_info_dict
        )
        object_name = self.get_run_name(train_path)
        filename = f'{TrainingRecorder.RECORDINGS_FOLDER}/{object_name}{self.timestamp}.pickle'
        with open(filename, 'wb') as handle:
            pickle.dump(recorded_output, handle, protocol=pickle.HIGHEST_PROTOCOL)