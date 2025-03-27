<p align="center">
  <img height="100" src="assets/3dgrut_logo.png">
</p>

---
<p align="center">
  <img width="100%" src="assets/nvidia-hq-playground.gif">
</p>

This repository provides the official implementations of **3D Gaussian Ray Tracing (3DGRT)** and **3D Gaussian Unscented Transform (3DGUT)**. Unlike traditional methods that rely on splatting, 3DGRT performs ray tracing of volumetric Gaussian particles instead. This enables support for distorted cameras with complex, time-dependent effects such as rolling shutters, while also efficiently simulating secondary rays required for rendering phenomena like reflection, refraction, and shadows. However, 3DGRT requires dedicated ray-tracing hardware and remains slower than 3DGS.

To mitigate this limitation, we also propose 3DGUT, which enables support for distorted cameras with complex, time-dependent effects within a rasterization framework, maintaining the efficiency of rasterization methods. By aligning the rendering formulations of 3DGRT and 3DGUT, we introduce a hybrid approach called **3DGRUT**. This technique allows for rendering primary rays via rasterization and secondary rays via ray tracing, combining the strengths of both methods for improved performance and flexibility.


> __3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes__  
> [Nicolas Moenne-Loccoz*](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Ashkan Mirzaei*](https://ashmrz.github.io), [Or Perel](https://orperel.github.io/), [Riccardo De Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://jme.pub/),   
> [Gavriel State](https://www.linkedin.com/in/gavstate/?originalSubdomain=ca), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Nicholas Sharp^](https://nmwsharp.com/), [Zan Gojcic^](https://zgojcic.github.io/) _(*,^ indicates equal contribution)_  
> _SIGGRAPH Asia 2024 (Journal Track)_  
> __[Project page](https://research.nvidia.com/labs/toronto-ai/3DGRT)&nbsp;/ [Paper](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf)&nbsp;/ [Video](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_supplementary_video.mp4)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgrt2024.bib)__

> __3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting__  
> [Qi Wu*](https://wilsoncernwq.github.io/), [Janick Martinez Esturo*](https://jme.pub/), [Ashkan Mirzaei](https://ashmrz.github.io),   
> [Nicolas Moenne-Loccoz](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Zan Gojcic](https://zgojcic.github.io/)  _(* indicates equal contribution)_  
> _CVPR 2025_  
> __[Project page](https://research.nvidia.com/labs/toronto-ai/3DGUT)&nbsp;/ [Paper](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf)&nbsp;/ [Video](https://research.nvidia.com/labs/toronto-ai/3DGUT/#supp_video)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgut2025.bib)__


## 🔥 News
- ✅[2025/03] Initial code release!
- ✅[2025/02] [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf) was accepted to CVPR 2025!
- ✅[2024/08] [3DGRT](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf) was accepted to SIGGRAPH Asia 2024!

## Contents

- [🔥 News](#-news)
- [Contents](#contents)
- [🔧 1 Dependencies and Installation](#-1-dependencies-and-installation)
  - [Running with Docker](#running-with-docker)
- [💻 2. Train 3DGRT or 3DGUT scenes](#-2-train-3dgrt-or-3dgut-scenes)
- [🎥 3. Rendering from Checkpoints](#-3-rendering-from-checkpoints)
  - [To visualize training progress interactively](#to-visualize-training-progress-interactively)
  - [To visualize a pre-trained checkpoint](#to-visualize-a-pre-trained-checkpoint)
- [📋 4. Evaluations](#-4-evaluations)
  - [3DGRT](#3dgrt)
    - [MipNeRF360 Dataset](#mipnerf360-dataset)
  - [3DGUT](#3dgut)
    - [MipNeRF360 Dataset](#mipnerf360-dataset-1)
    - [Scannet++ Dataset](#scannet-dataset)
- [🛝 5. Interactive Playground GUI](#-5-interactive-playground-gui)
- [🎓 6. Citations](#-6-citations)
- [🙏 7. Acknowledgements](#-7-acknowledgements)

## 🔧 1 Dependencies and Installation
- CUDA 11.8+ Compatible System
- For good performance with 3DGRT, we recommend using an NVIDIA GPU with Ray Tracing (RT) cores.
- Currently, only Linux environments are supported by the included install script (Windows support coming soon!)

<details> 
<summary> NOTE: gcc versions >11 (expand for details)</summary>

Currently the codebase requires gcc <= 11.  If your machine uses the compiler gcc-12 or newer (i.e., in Ubuntu 24.04), you may need to install and use gcc-11. 

First, install gcc 11:
```sh
sudo apt-get install gcc-11 g++-11
```

Then run the install script with the optional `WITH_GCC11` flag, which additionally configures the conda environment to use gcc-11:
```sh
./install_env.sh 3dgrut WITH_GCC11
```
</details>
</br>

To set up the environment using conda, first clone the repository and run `./install_env.sh` script as:

```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# You can install each component step by step following install_env.sh
chmod +x install_env.sh
./install_env.sh 3dgrut
conda activate 3dgrut
```

### Running with Docker

Build the docker image:
```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
docker build . -t 3dgrut
````

Run it:
```bash
xhost +local:root
docker run -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace --runtime=nvidia -e DISPLAY 3dgrut
```
> [!NOTE]
> Remember to set DISPLAY environment variable if you are running on a remote server from command line.

## 💻 2. Train 3DGRT or 3DGUT scenes

We provide different configurations for training using 3DGRT and 3DGUT models on common benchmark datasets. 
For example you can download [NeRF Synthetic dataset](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset), 
[MipNeRF360 dataset](https://jonbarron.info/mipnerf360/) or [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), 
and then run one of the following commands:

```bash
# Train Lego with 3DGRT & 3DGUT
python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgrt
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgut

# Train Bonsai
python train.py --config-name apps/colmap_3dgrt.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgrt dataset.downsample_factor=2 
python train.py --config-name apps/colmap_3dgut.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgut dataset.downsample_factor=2 

# Train Scannet++
python train.py --config-name apps/scannetpp_3dgut.yaml path=data/scannetpp/0a5c013435/dslr out_dir=runs experiment_name=0a5c013435_3dgut
```

> [!Note] 
> For ScanNet++, we expect the dataset to be preprocessed following [FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)'s method.

> [!Note]  
> If you're running from PyCharm IDE, enable rich console through:
> Run Configuration > Modify Options > Emulate terminal in output console*

## 🎥 3. Rendering from Checkpoints
Evaluate Checkpoint with Splatting / OptiX Tracer / Torch
```bash
python render.py --checkpoint runs/lego/ckpt_last.pt --out-dir outputs/eval
```


### To visualize training progress interactively
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True 
```

### To visualize a pre-trained checkpoint
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True test_last=False export_ingp.enabled=False resume=runs/lego/ckpt_last.pt 
```
> [!NOTE]
> Remember to set DISPLAY environment variable if you are running on a remote server from command line.

On start up, you might see a black screen, but you can use the GUI to navigate to correct camera views:
<img src="assets/train_gui_initial.jpg" height="400"/> 
<img src="assets/render_lego.jpg" height="400"/>

## 📋 4. Evaluations

We provide scripts to reproduce results reported in publications.

```bash
# Training
bash ./benchmark/mipnerf360_3dgut.sh <config-yaml>
# Rendering
bash ./benchmark/mipnerf360_3dgut_render.sh <results-folder>
```

### 3DGRT

**NeRF Synthestic Dataset**

```bash
bash ./benchmark/nerf_synthestic.sh paper/3dgrt/colmap_ours.yaml
bash ./benchmark/nerf_synthestic_render.sh results/nerf_synthetic
```

Results for 3DGRT (Produced on RTX 5090):

|            | PSNR	  | SSIM	| Train (s) |	FPS |
|------------|--------|-------|-------|------|
| Chair      | 35.85	| 0.988	| 556.4	| 299 |
| Drums      | 25.87	| 0.953	| 462.8	| 389 |
| Ficus      | 36.57	| 0.989	| 331.0	| 465 |
| Hotdog     | 37.88	| 0.986	| 597.0	| 270 |
| Lego       | 36.70	| 0.985	| 469.8	| 360 |
| Materials  | 30.42	| 0.962	| 463.3	| 347 |
| Mic        | 35.90	| 0.992	| 443.4	| 291 |
| Ship       | 31.73	| 0.909	| 510.7	| 360 |
| *Average*  | 33.87	| 0.971	| 479.3	| 347 |

**MipNeRF360 Dataset**

```bash
bash ./benchmark/mipnerf360.sh paper/3dgrt/colmap_ours.yaml
bash ./benchmark/mipnerf360_render.sh results/mipnerf360
```

Results for 3DGRT (Produced on RTX 5090):

|          | Bicycle | Bonsai  | Counter | Flower  | Garden  | Kitchen | Room    | Stump   | Treehill | *Average* |
|----------|---------|---------|---------|---------|---------|---------|---------|---------|----------|-----------|
| train(s) | 5417.39 | 4107.31 | 4157.14 | 4150.21 | 4660.56 | 5230.30 | 3884.14 | 5197.60 | 5169.49  | 4663.79   |
| FPS      | 63.8    | 96.1    | 74.3    | 92.9    | 73.2    | 48.4    | 91.5    | 71.2    | 66.7     | 75.3      |
| PSNR     | 24.633  | 31.724  | 27.978  | 21.327  | 26.594  | 30.057  | 30.194  | 26.323  | 22.412   | 26.805    |


### 3DGUT

**NeRF Synthestic Dataset**

```bash
bash ./benchmark/nerf_synthestic.sh paper/3dgut/unsorted_nerf_synthestic.yaml
bash ./benchmark/nerf_synthestic_render.sh results/nerf_synthetic
```

Results for 3DGUT (Produced on RTX 5090):

|            | PSNR	  | SSIM	| Train (s) |	FPS |
|------------|--------|-------|-------|------|
| Chair      | 35.61	| 0.988	| 265.6	| 599  |
| Drums      | 25.99	| 0.953	| 254.1	| 694  |
| Ficus      | 36.43	| 0.988	| 183.5	| 1053 |
| Hotdog     | 38.11	| 0.986	| 184.8	| 952  |
| Lego       | 36.47	| 0.984	| 221.7	| 826  |
| Materials  | 30.39	| 0.960	| 194.3	| 1000 |
| Mic        | 36.32	| 0.992	| 204.7	| 775  |
| Ship       | 31.72	| 0.908	| 208.5	| 870  |
| *Average*  | 33.88	| 0.970	| 214.6	| 846  |

**MipNeRF360 Dataset**

```bash
bash ./benchmark/mipnerf360.sh paper/3dgut/unsorted_colmap.yaml
bash ./benchmark/mipnerf360_render.sh results/mipnerf360
```

Results for unsorted 3DGUT (Produced on RTX 5090):

|           | PSNR  | SSIM	| Train (s) |	FPS |
|-----------|-------|-------|-------|------|
| Bicycle   | 25.01	| 0.759	| 949.8	| 275 |
| Bonsai    | 32.46	| 0.945	| 485.3	| 362 |
| Counter   | 29.14	| 0.911	| 484.5	| 380 |
| Flowers   | 21.45	| 0.612	| 782.0	| 253 |
| Garden    | 27.18	| 0.856	| 810.2	| 316 |
| Kitchen   | 31.16	| 0.928	| 664.8	| 275 |
| Room      | 31.63	| 0.920	| 448.8	| 370 |
| Stump     | 26.50	| 0.773	| 742.6	| 319 |
| Treehill  | 22.35	| 0.627	| 809.6	| 299 |
| *Average* | 27.43	| 0.815	| 686.4	| 317 |

**Scannet++ Dataset**

```bash
bash ./benchmark/scannetpp.sh paper/3dgut/unsorted_scannetpp.yaml
bash ./benchmark/scannetpp_render.sh results/scannetpp
```
> [!Note]
> We followed [FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)'s convention to prepare the dataset for fair comparisons

Results for unsorted 3DGUT (Produced on RTX 5090):

|           | PSNR  | SSIM	| Train (s) |	FPS |
|-----------|-------|-------|-------|------|
| 0a5c013435 | 29.67	| 0.930	| 292.3	| 389 | 
| 8d563fc2cc | 26.88	| 0.912	| 286.1	| 439 | 
| bb87c292ad | 31.58	| 0.941	| 316.9	| 448 | 
| d415cc449b | 28.12	| 0.871	| 394.6	| 483 | 
| e8ea9b4da8 | 33.47	| 0.954	| 280.8	| 394 | 
| fe1733741f | 25.60	| 0.858	| 355.8	| 450 | 
| *Average*  | 29.22	| 0.911	| 321.1	| 434 |

## 🛝 5. Interactive Playground GUI

The playground allows interactive exploration of pretrained scenes, with raytracing effects such as inserted objects, 
reflections, refractions, depth of field, and more.

Run the playground UI to visualize a pretrained scene with:
```
python playground.py --gs_object <ckpt_path>
```

See [Playground README](playground/README.md) for details.



## 🎓 6. Citations

```
@article{loccoz20243dgrt,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```

```
@article{wu20253dgut,
    title={3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting},
    author={Wu, Qi and Martinez Esturo, Janick and Mirzaei, Ashkan and Moenne-Loccoz, Nicolas and Gojcic, Zan},
    journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

## 🙏 7. Acknowledgements

We sincerely thank our colleagues for their valuable contributions to this project.

Hassan Abu Alhaija, Ronnie Sharif, Beau Perschall and Lars Fabiunke for assistance with assets.
Greg Muthler, Magnus Andersson, Maksim Eisenstein, Tanki Zhang, Nathan Morrical, Dietger van Antwerpen and John Burgess for performance feedback.
Thomas Müller, Merlin Nimier-David, and Carsten Kolve for inspiration and pointers. 
Ziyu Chen, Clement Fuji-Tsang, Masha Shugrina, and George Kopanas for technical & experiment assistance,
and to Ramana Kiran and Shailesh Mishra for typo fixes.
