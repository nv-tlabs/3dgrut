
# Ray Tracing Gaussian Splats

## Dependencies
- __CUDA 11.8 or newer__.
- __Python 3.11 or newer__.

## Set up the environment

To set up the environment using conda, you can run the following

```
conda create -n 3dgrt python=3.11
conda activate 3dgrt 
pip install -r requirements.txt

# Install ray utils cpp module for AV data / packed_ops
pip install ./libs/ray_utils
pip install ./libs/packed_ops

# Install packed_ops

# Install simple-knn submodule
pip install ./thirdparty/simple-knn
```

## Using the GUI

To use the interactive UI, install the following optional dependencies, and run with `--with-gui`

```
pip install git+ssh://git@github.com/nmwsharp/polyscope-py.git@v2
pip install cuda-python cupy
```

The latter two packages `cuda-python` and `cupy` may very slow to install and/or create CUDA versioning problems, which is why we don't install them by default. (In the future we could remove these by writing a few of our own cuda bindings.)

## NGC utils

Based on [ngc-toolbox](https://gitlab-master.nvidia.com/jalucas/ngc-toolbox/-/tree/main?ref_type=heads) see more detailed README in `./utils/ngc/README.md`

### Prerequisites

- [Request access to your NGC team](https://goo.gl/forms/IjKBiZRt4RYZcF3h1) (`omniverse` for the ovx A40 instances on OVC)
    - Once approved, you will receive a sign-in link to set your password
- [Download and install the NGC CLI](https://ngc.nvidia.com/setup/installers/cli)
- [Install docker](https://docs.docker.com/get-docker/)

### Setting up

- [Generate an NGC API key](https://ngc.nvidia.com/configuration/api-key) (and keep it somewhere safe)
- Set key in docker: `docker login -u \$oauthtoken -p <YOUR_API_KEY> nvcr.io `
- Configure NGC CLI: `ngc config set`
    - Input your API key when requested, choose the team `omniverse` and ace `nv-us-west-2`

**You're ready to go!**

### Updating Docker container (only if needed)
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml build_docker
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml push_docker
```
### Workspace

Mounting a workspace locally:
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml mount_workspace
```
Unmounting:
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml unmount_workspace
```

Syncing current version of code to workspace:
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml sync_workspace experiment_name
```
This will create a copy of the code in the workspace, in the directory `experiments/experiment_name`.

### Jobs
Before any job, the command will execute `./utils/ngc/ngc_pre_job.sh`.
After any job, the command will execute `./utils/ngc/ngc_post_job.sh`.

Start an interactive job:
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml run_interactive_job --runtime 4h 
```
The job will spin up and you'll be assigned a job ID (presented in the command output). You can connect to the job via `ngc batch exec <job_id>`.


Runnig a normal job:
```bash
python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml run_job "python train.py --arg1 value1 --arg2 value2" experiment_name
```

The job will spin up on ngc, and should be visible on the [NGC dashboard](https://ngc.nvidia.com/dashboard).

Note that syncing a workspace and running a normal job can be combined using the `submit_job` method:

`python ./utils/ngc/app.py --config ngc_config/3dgrt.toml submit_job "python train.py --arg1 value1 --arg2 value2" experiment_name`

**A large batch of jobs**

First, generate a file where each line corresponds to a different command to be run. For example,

> grid_search.txt
```
python train.py --arg1 a1 --arg2 a2
python train.py --arg1 b1 --arg2 a2
python train.py --arg1 a1 --arg2 b2
python train.py --arg1 b1 --arg2 b2
```

Then execute `python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml  generate_job_array grid_search.txt grid_search grid_search_jobs/`.
The final three arguments are the file containing the commands, a name for the experiment, and the output folder to place the job data.
Running this command creates a new directory called `grid_search_jobs` containing the files `[cmd_0.json, cmd_1.json, cmd_2.json, cmd_3.json]`. You can then dispatch all jobs to NGC via,

`python ./utils/ngc/app.py --config ngc/ngc_config/3dgrt.toml  run_job_array grid_search_jobs`

_This could also be achieve in one step, by adding the `--run` flag to the `generate_job_array` command._




### Sample commands to run on NeRF Synthetic scenes

```
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/lego  with_gui=True test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/lego use_wandb=True experiment_name=test-wandb record_training=true test_last=True
```

Fine-tuning from 3DGS

```
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/lego  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/hotdog  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/chair  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/drums  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/ficus  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/materials  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/mic  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/ship  with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3

python train.py --config-name apps/colmap.yaml path=data/tandt/truck with_gui=True initialization.method="point_cloud" optimizer.params.positions.lr=0.0000016 model.densify.end_iteration=-1 model.prune.end_iteration=-1 model.reset_density.end_iteration=-1 model.progressive_training.init_n_features=3
```

### Running the experiments on ngc

```
rm utils/ngc/grid_search_configs/grid_search/*
EXP_NAME="Ashkan-Feb21-L2-finetune"
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml sync_workspace $EXP_NAME
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml generate_job_array grid_search/finetune.txt grid_search grid_search_jobs/ --run --exp_name $EXP_NAME
```

```
rm utils/ngc/grid_search_configs/grid_search/*
EXP_NAME="Ashkan-Feb28-L1-ssim-Loss"
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml sync_workspace $EXP_NAME
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml generate_job_array grid_search/random_init.txt grid_search grid_search_jobs/ --run --exp_name $EXP_NAME
```

running the ablations

```
rm utils/ngc/grid_search_configs/grid_search/*
EXP_NAME="Ashkan-Feb21-l1-ssim-knn-grad-thresh"
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml sync_workspace $EXP_NAME
python utils/ngc/app.py --config utils/ngc/ngc_config/3dgrt.toml generate_job_array grid_search/ablations.txt grid_search grid_search_jobs/ --run --exp_name $EXP_NAME
```