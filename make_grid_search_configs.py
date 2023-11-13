import os
from itertools import product


if __name__ == "__main__":
    base_cfg_path = "./ngc/grid_search_configs"
    exp_name = "densify"

    search_params = {
        'model.densify.clone_grad_threshold': [0.00005 , 0.0002, 0.0005],
        'model.densify.split_grad_threshold': [0.00005 , 0.0002, 0.0005],
        'model.densify.relative_size_threshold': [0.01, 0.02, 0.03],
        'model.prune.density_threshold': [0.01, 0.02, 0.03],

    }

    base_cmd = f"WANDB_API_KEY=eaaec74f703cfe44de835946df755b1008106503 python train.py --config-name=apps/colmap use_wandb=True path=/workspace/data/tandt/truck  experiment_name={exp_name} record_training=true test_last=True val_frequency=10"

    grid_search = list(product(*search_params.values()))

    cfg_path = os.path.join(base_cfg_path, exp_name + ".txt")
    with open(cfg_path, "w") as file:
        for params in grid_search:
            cmd = base_cmd
            for param_value, param_key in zip(params,search_params.keys()):
                cmd += f" {param_key}={param_value}"
            cmd += "\n"
            # print(cmd)
            file.write(cmd)