

def record_train_step(info_dict, model, iteration, iteration_time, l1_loss, ssim_loss, total_loss, psnr):
    if not (iteration > 0 and iteration % 100 == 0):
        return
    num_gaussians = model.positions.shape[0]

    info_dict['iteration'].append(iteration)
    info_dict['iter_time'].append(iteration_time)
    info_dict['num_gaussians'].append(num_gaussians)
    info_dict['active_sh_deg'].append(model.n_active_features)

    info_dict['l1_loss'].append(l1_loss.item())
    info_dict['ssim_loss'].append(ssim_loss.item())
    info_dict['total_loss'].append(total_loss.item())
    info_dict['psnr'].append(psnr)

    info_dict['opacity_mean'].append(model.get_density().mean().item())
    info_dict['opacity_std'].append(model.get_density().std().item())

    info_dict['features_albedo_mean'].append(model.features_albedo.mean().item())
    info_dict['features_albedo_R_mean'].append(model.features_albedo.mean(dim=0)[0].item())
    info_dict['features_albedo_G_mean'].append(model.features_albedo.mean(dim=0)[1].item())
    info_dict['features_albedo_B_mean'].append(model.features_albedo.mean(dim=0)[2].item())
    info_dict['features_albedo_std'].append(model.features_albedo.std().item())
    info_dict['features_albedo_R_std'].append(model.features_albedo.std(dim=0)[0].item())
    info_dict['features_albedo_G_std'].append(model.features_albedo.std(dim=0)[1].item())
    info_dict['features_albedo_B_std'].append(model.features_albedo.std(dim=0)[2].item())

    features_specular = model.features_specular.reshape(num_gaussians, -1, 3)
    info_dict['features_specular_mean'].append(features_specular.mean().item())
    info_dict['features_specular_R_mean'].append(features_specular.mean(dim=(0,1))[0].item())
    info_dict['features_specular_G_mean'].append(features_specular.mean(dim=(0,1))[1].item())
    info_dict['features_specular_B_mean'].append(features_specular.mean(dim=(0,1))[2].item())
    info_dict['features_specular_std'].append(features_specular.std().item())
    info_dict['features_specular_R_std'].append(features_specular.std(dim=(0,1))[0].item())
    info_dict['features_specular_G_std'].append(features_specular.std(dim=(0,1))[1].item())
    info_dict['features_specular_B_std'].append(features_specular.std(dim=(0,1))[2].item())

    info_dict['pos_x_mean'].append(model.get_positions().mean(dim=0)[0].item())
    info_dict['pos_y_mean'].append(model.get_positions().mean(dim=0)[1].item())
    info_dict['pos_z_mean'].append(model.get_positions().mean(dim=0)[2].item())
    info_dict['pos_x_std'].append(model.get_positions().std(dim=0)[0].item())
    info_dict['pos_y_std'].append(model.get_positions().std(dim=0)[1].item())
    info_dict['pos_z_std'].append(model.get_positions().std(dim=0)[2].item())

    info_dict['scale_x_mean'].append(model.get_scale().mean(dim=0)[0].item())
    info_dict['scale_y_mean'].append(model.get_scale().mean(dim=0)[1].item())
    info_dict['scale_z_mean'].append(model.get_scale().mean(dim=0)[2].item())
    info_dict['scale_x_std'].append(model.get_scale().std(dim=0)[0].item())
    info_dict['scale_y_std'].append(model.get_scale().std(dim=0)[1].item())
    info_dict['scale_z_std'].append(model.get_scale().std(dim=0)[2].item())

    info_dict['rot_x_mean'].append(model.get_rotation().mean(dim=0)[0].item())
    info_dict['rot_y_mean'].append(model.get_rotation().mean(dim=0)[1].item())
    info_dict['rot_z_mean'].append(model.get_rotation().mean(dim=0)[2].item())
    info_dict['rot_w_mean'].append(model.get_rotation().mean(dim=0)[3].item())
    info_dict['rot_x_std'].append(model.get_rotation().std(dim=0)[0].item())
    info_dict['rot_y_std'].append(model.get_rotation().std(dim=0)[1].item())
    info_dict['rot_z_std'].append(model.get_rotation().std(dim=0)[2].item())
    info_dict['rot_w_std'].append(model.get_rotation().std(dim=0)[3].item())


def get_gaussians_info(gaussians):
    import torch
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

def submit_extra_output(info_dict, dataset, scene_extent, train_path, model):
    import os
    import pickle
    from pathlib import Path

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

    os.makedirs('extra_info', exist_ok=True)
    extra_output = dict(
        scene_info = dict(
            cameras_extent=scene_extents,
            num_train_cameras=num_of_cameras
        ),
        gaussians=get_gaussians_info(model),
        extra_dict=info_dict
    )
    object_name = Path(train_path).stem
    filename = f'extra_info/{object_name}.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(extra_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
