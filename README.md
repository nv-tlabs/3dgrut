# FullCircle: Effortless 3D Reconstruction from Casual 360° Captures
**Abstract**: We propose a practical pipeline for reconstructing 3D scenes directly from raw 360° camera captures. Our pipeline requires no special capture protocols or pre-processing, and exhibits robustness to a prevalent source of reconstruction errors: the human operator that is visible in all 360° imagery. To facilitate evaluation, we introduce a multi-tiered dataset of scenes captured as raw dual-fisheye images, establishing a benchmark for robust casual 360° reconstruction.

[🌍 Project page](https://theialab.github.io/fullcircle/)  
[📄 Paper](https://theialab.github.io/fullcircle/paper.pdf)


### 1. Install FullCircle
```bash
git clone git@github.com:theialab/fullcircle.git
cd fullcircle
chmod +x install_env.sh
./install_env.sh fullcircle
conda activate fullcircle
```

## 2. Data preparation
### 2.1 Camera calibration
Run COLMAP-based calibration using:
```bash
bash preprocessing/colmap.sh
```
This step estimates camera intrinsics and extrinsics for the dual-fisheye (360°) capture.

Organize each scene in COLMAP format under:
```bash
data/<scene_name>/
```
### 2.2 Automated masking
```bash
bash preprocessing/run.sh
```

## 3. Training
```
python train.py \
  --config-name apps/colmap_3dgrt.yaml \
  path=data/${scene_name} \
  out_dir=runs \
  dataset.downsample_factor=4
```

## 4. Rendering
```
python render.py \
  --checkpoint runs/${scene_name}/ckpt_last.pt \
  --out-dir outputs/${scene_name}
```

## BibTeX
```bibtex
@article{foroutan2026fullcircle,
  title   = {{FullCircle}: Effortless 3D Reconstruction from Casual 360° Captures},
  author  = {Foroutan, Yalda and Oztas, Ipek and Rebain, Daniel and Dundar, Aysegul and Yi, Kwang Moo and Goli, Lily and Tagliasacchi, Andrea},
  journal = {arXiv preprint},
  year    = {2026}
}
```
