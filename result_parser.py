import os
import argparse

import yaml
from itertools import zip_longest
import numpy as np

from render import Renderer

REMOTE_WORKSPACE_DATA_PATH = "/workspace/data"

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-path", required=True, type=str, help="Path to the workspace")
    parser.add_argument("--experiment-name", required=True, type=str, help="Experiment name")
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--log-dir", type=str, default="runs", help="Tensorbord log dir")
    parser.add_argument("--local-data-path", type=str, default="/scratch/data/3dgrt_data", help="local data dir")
    parser.add_argument("--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]")
    args = parser.parse_args()

    tensorboard_path = os.path.join(args.workspace_path,"experiments",args.experiment_name,args.log_dir)
    runs = os.listdir(tensorboard_path)

    psnrs = []
    stds = []   
    checkpoints = []
    os.makedirs(args.out_dir, exist_ok=True)
    for run in runs:
        run_path = os.path.join(tensorboard_path,run)
        
        conf_path = os.path.join(run_path,'parsed.yaml')
        if os.path.exists(conf_path):
            with open(conf_path, 'r') as file:
                conf = yaml.safe_load(file)

            relative_path = []
            for a, b in zip_longest(conf['path'].split('/'), REMOTE_WORKSPACE_DATA_PATH.split('/')):
                if a != b:
                    relative_path.append(a if a is not None else b)
            local_data_path = os.path.join(args.local_data_path,"/".join(relative_path))

            ls_dir = sorted(os.listdir(run_path))
            if len(ls_dir)>0:
                filtered_checkpoints = list(filter(lambda x: x.endswith('.pt'), ls_dir))
                epochs = [int(checkpoint.split('.')[0].split('_')[1]) for checkpoint in filtered_checkpoints]

                last_checkpoint = filtered_checkpoints[np.argmax(epochs)]

                print()
                print(f"Rendering for experiment {args.experiment_name}, run {run}, checkpoint {last_checkpoint}")
                renderer = Renderer(checkpoint_path=os.path.join(run_path,last_checkpoint), 
                                    out_dir=os.path.join(args.out_dir,run),
                                    path=local_data_path,
                                    save_gt=args.save_gt)
                mean, std = renderer.render_all()
                psnrs.append(mean)
                stds.append(std)
                checkpoints.append(last_checkpoint)


    print(f"\n\nExperiment {args.experiment_name} summary:\n")
    for psnr, std, run, checkpoint in zip(psnrs, stds, runs, checkpoints):
        print(f"Run: {run}, checkpoint {checkpoint}")
        print(f"PSNR: {psnr} std {std}\n")

    