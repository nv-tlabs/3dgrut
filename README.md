
# Ray Tracing Gaussian Splats

## DISCLAIMER

This branch is intended solely for an internal release and is not maintained in its current form. The code herein may not adhere to our usual standards of cleanliness or best practices.

## Dependencies
- __Python 3.11 or newer__.

## Set up the environment

We use git lfs to track certain files so as a first step after cloning the repo make sure to run:

```
git lfs install
git lfs pull
```

To set up the environment using conda, you can run the following

```
conda create -n 3dgrt python=3.11 -y
conda activate 3dgrt

conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 "numpy<2.0" ninja -c pytorch -c nvidia/label/cuda-11.8.0

pip install -r requirements.txt

# Install ray utils cpp module for AV data
pip install ./libs/ray_utils

# Install packed_ops
pip install ./libs/packed_ops

# Install simple-knn submodule
pip install ./thirdparty/simple-knn
```

## Using the GUI

To use the interactive UI, install the following optional dependencies:

```
pip install cuda-python cupy
```

The latter two packages `cuda-python` and `cupy` may very slow to install and/or create CUDA versioning problems, which is why we don't install them by default. (In the future we could remove these by writing a few of our own cuda bindings.)

## Downloading the datasets

We use the same format for the data as [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). You can refer to their codebase for downloading the dataset. Alternatively, the datasets are also available [here](https://drive.google.com/drive/folders/1MRh6P5B9yBnWozDlHIBQeM-XPNRQ3I3E?usp=drive_link) on our corporate Google drive. We particularly used the following datasets for view-synthesis evaluations:

- nerf_synthetic
- tandt (Tanks and Temples)
- mipNeRF360
- db (Deep Blending)

Download the datasets and place them under `data` with the following structure:

```
data
├── db
│   ├── drjohnson
│   └── playroom
├── mipNeRF360
│   ├── bicycle
│   ├── bonsai
│   ├── counter
│   ├── garden
│   ├── kitchen
│   ├── room
│   ├── stump
├── nerf_synthetic
│   ├── chair
│   ├── drums
│   ├── ficus
│   ├── hotdog
│   ├── lego
│   ├── materials
│   ├── mic
│   ├── ship
└── tandt
    ├── train
    └── truck
```

# Optimizing the scenes

For training scenes using our fast setting ,`ours (fast)`, you can use one of the following commands:

```
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/ficus test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/chair test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/drums test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/hotdog test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/lego test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/materials test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/mic test_last=True
python train.py --config-name apps/nerf_synthetic.yaml path=data/nerf_synthetic/ship test_last=True

python train.py --config-name=apps/colmap.yaml path=data/tandt/truck test_last=True
python train.py --config-name=apps/colmap.yaml path=data/tandt/train test_last=True

python train.py --config-name=apps/colmap.yaml path=data/db/drjohnson test_last=True
python train.py --config-name=apps/colmap.yaml path=data/db/playroom test_last=True

python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/counter test_last=True dataset.downsample_factor=2
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/bicycle test_last=True dataset.downsample_factor=4
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/kitchen test_last=True dataset.downsample_factor=2
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/room test_last=True dataset.downsample_factor=2
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/garden test_last=True dataset.downsample_factor=4
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/stump test_last=True dataset.downsample_factor=4
python train.py --config-name=apps/colmap.yaml path=data/mipNeRF360/bonsai test_last=True dataset.downsample_factor=2
```

Alternatively, you can use the reference setting, `ours (reference)`, by using adding `optimizer.params.density.lr=0.05 render.kernel_function=gaussian` to the commands. Note that you can export the optimized set of particles (Gaussians) by adding `export_ingp.enabled=True export_ingp.path="<out_path>.ingp"` (replace `<out_path>` with your desired file name). Then, for getting the final inference times, you can load the ingp file using `import_ingp.enabled=True import_ingp.path="<out_path>.ingp`, and set `render.adaptive_kernel_clamping=True render.min_transmittance=0.01`. 

To run each command with the Polyscope visualizer, use `with_gui=True`. 

# Checkpoints in the form of ingp files

You can find our pretrained `.ingp` checkpoints here:
- [Ours (reference)](https://drive.google.com/drive/folders/1aIOW9vTiqIo3vVMhkAxlnj5DWG4t7JoY?usp=drive_link)
- [Ours (fast)](https://drive.google.com/drive/folders/1H_ShVx_uGr3Imlj0I5uOEBzic5Dz2UBB?usp=drive_link)

As mentioned above, you can load and visualize them using the following arguments: 
`import_ingp.enabled=True import_ingp.path="<ingp_path>.ingp render.adaptive_kernel_clamping=True render.min_transmittance=0.01 with_gui=True`. Note that for `ours (reference)` you'll need `render.kernel_function=gaussian` too. 