# NHT Parity Handoff

Date: 2026-07-01

## Objective

Make the 3DGRUT 3DGUT NHT implementation reproduce the reference Neural Harmonic Textures results, not just recover quality with 3DGUT-specific approximations.

Reference results are available at:

```text
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000
```

Current 3DGRUT parity run is available at:

```text
results/mipnerf360_2293c73d_nht_mcmc
```

## Current Status

The latest 3DGRUT run did use the intended basic parity switches:

- `normalize_world_space: true`
- `initialization.use_observation_points: false`
- `render.particle_feature_half: false`
- `render.feature_output_half: false`
- rebuilt 3DGUT with `PARTICLE_FEATURE_HALF=0` and `FEATURE_OUTPUT_HALF=0`
- `render.splat.k_buffer_size: 0`

The run logs confirm GSplat-style normalization was applied for train and validation splits.

Reference NHT uses:

- `with_ut: true`
- `with_eval3d: true`
- `nht_fused: true`
- `normalize_world_space: true`
- `init_type: sfm`
- `init_opa: 0.5`
- `init_scale: 0.1`
- `test_every: 8`
- `deferred_opt_feature_dim: 48`
- `deferred_features_lr: 0.015`
- `deferred_mlp_lr: 0.00068`
- `deferred_mlp_ema_decay: 0.95`
- `color_refine_steps: 3000`

Important: reference NHT does not use a 3DGUT k-buffer. It sorts/intersects through gsplat's tile intersection path. `k_buffer_size=16` is only a diagnostic ablation for 3DGUT ordering, not reference parity.

## Metric Gap

The comparison below uses the same nine Mip-NeRF 360 scenes.

| Scene | 3DGRUT PSNR | Ref PSNR | Delta | 3DGRUT SSIM | Ref SSIM | Delta |
|---|---:|---:|---:|---:|---:|---:|
| bicycle | 25.094 | 25.544 | -0.450 | 0.765 | 0.779 | -0.014 |
| bonsai | 34.120 | 34.458 | -0.337 | 0.954 | 0.960 | -0.006 |
| counter | 30.469 | 30.576 | -0.108 | 0.924 | 0.930 | -0.007 |
| flowers | 21.418 | 21.746 | -0.328 | 0.606 | 0.625 | -0.020 |
| garden | 27.662 | 28.119 | -0.457 | 0.865 | 0.875 | -0.011 |
| kitchen | 32.861 | 33.050 | -0.189 | 0.937 | 0.943 | -0.005 |
| room | 33.011 | 33.281 | -0.270 | 0.935 | 0.943 | -0.008 |
| stump | 26.355 | 26.737 | -0.382 | 0.765 | 0.781 | -0.015 |
| treehill | 22.827 | 23.197 | -0.370 | 0.647 | 0.668 | -0.021 |

| Aggregate | 3DGRUT | Reference | Delta |
|---|---:|---:|---:|
| PSNR | 28.2019 | 28.5231 | -0.3212 |
| SSIM | 0.8220 | 0.8339 | -0.0119 |
| LPIPS | 0.2450 | 0.2312 | +0.0138 |

## Working Hypotheses

1. Renderer semantics are still different.
   - Reference NHT uses gsplat ordered tile intersections and `rasterize_to_pixels_eval3d_extra`.
   - 3DGUT uses its own ray traversal/compositing path.
   - NHT composites latent features before an MLP, so small ordering/transmittance differences can become visible after decoding.

2. Training dynamics may diverge even if the renderer is close.
   - MCMC relocation/add behavior, opacity evolution, scale evolution, and color refine timing must be compared against reference checkpoints/stats.

3. Metric/evaluation differences are less likely but still need to be ruled out.
   - Reference stats live under each scene's `stats/val_step29999.json`.
   - 3DGRUT metrics live under each scene's `metrics.json`.

## Execution Plan

### Step 1: Lock One-Scene Target

Use `garden` first.

Reference artifacts:

```text
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/cfg.yml
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/ckpts/ckpt_29999_rank0.pt
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/val_step29999.json
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/val_step29999_per_image.json
```

3DGRUT artifacts:

```text
results/mipnerf360_2293c73d_nht_mcmc/garden/garden-3006_012105/parsed.yaml
results/mipnerf360_2293c73d_nht_mcmc/garden/garden-3006_012105/ckpt_last.pt
results/mipnerf360_2293c73d_nht_mcmc/garden/garden-3006_012105/metrics.json
results/mipnerf360_2293c73d_nht_mcmc/train_garden.log
```

Acceptance for one-scene parity: final `garden` PSNR within about `0.05 dB` of reference, with SSIM/LPIPS close enough to be explained by metric implementation noise.

### Step 2: Render-Only A/B On The Same Checkpoint

Goal: decide whether the gap is renderer semantics or training dynamics.

Use one checkpoint and render it through both paths:

- 3DGRUT 3DGUT NHT renderer
- reference gsplat NHT renderer

Dump these for the same validation frames:

- rasterized encoded features before decoder
- alpha
- ray directions used by the decoder
- decoded RGB before background
- final RGB after background
- per-frame PSNR/SSIM/LPIPS

If the same checkpoint scores differently depending on renderer, prioritize renderer parity. If both renderers score similarly, prioritize training dynamics.

Implementation note: start with one frame from `garden` to keep iteration fast. Add a temporary debug script under `tools/` or `scripts/`; do not refactor production training code until the blocker is identified.

Execution update:

- Added `tools/nht_step2_parity.py` for checkpoint conversion, saved-PNG comparison, command generation, and runtime preflight.
- Converted the 3DGRUT garden checkpoint to reference gsplat/NHT checkpoint format:

```text
results/nht_step2_parity/garden_grut_as_gsplat/ckpts/ckpt_29999_rank0.pt
```

- The converted checkpoint loads with the expected reference schema:
  - `splats.features`: `(1000000, 48)` float32
  - `splats.means`: `(1000000, 3)` float32
  - `splats.opacities`: `(1000000,)` float32
  - `splats.quats`: `(1000000, 4)` float32
  - `splats.scales`: `(1000000, 3)` float32
  - `deferred_module.backbone.params`: `(40960,)`
  - `deferred_ema.backbone.params`: `(40960,)`
  - `deferred_module_config`: feature dim 48, SH view encoding, hidden dim 128, 3 layers, SH degree 3, SH scale 3.0, center-ray encoding false.

- Local runtime preflight found Torch 2.9.1+cu128 but `torch.cuda.is_available() == False`, so the actual CUDA render-only A/B could not be completed in this session.
- CUDA run supplied later by the user completed under:

```text
results/nht_step2_parity/garden_grut_as_gsplat_eval
```

- Same 3DGRUT checkpoint rendered through reference gsplat/NHT scored much worse than the original 3DGRUT render:

| Metric | 3DGRUT renderer | Converted checkpoint via reference renderer | Reference trained+rendered | Converted ref - 3DGRUT |
|---|---:|---:|---:|---:|
| PSNR | 27.6621 | 25.8323 | 28.1193 | -1.8298 |
| SSIM | 0.8645 | 0.7965 | 0.8753 | -0.0680 |
| LPIPS | 0.1311 | 0.1522 | 0.1227 | +0.0212 |

- Tensor conversion was verified to be exact for positions, scales, rotations, densities/opacities, features, and EMA decoder parameters.
- A concrete renderer-constant mismatch was found:
  - Current 3DGRUT NHT run compiled with `GAUSSIAN_PARTICLE_MAX_ALPHA=0.999`.
  - Reference gsplat/NHT hard-codes `MAX_ALPHA 0.99`.
  - Updated 3DGUT NHT configs to set `render.particle_kernel_max_alpha: 0.99`:

```text
configs/apps/colmap_3dgut_mcmc_nht.yaml
configs/apps/nerf_synthetic_3dgut_mcmc_nht.yaml
```

Conclusion: Step 2 currently points to renderer/constants parity as a blocker. Rerun `garden` with `particle_kernel_max_alpha: 0.99` before spending time on training-trajectory debugging.

Alpha-0.99 rerun update:

```text
results/mipnerf360_nht_alpha099
```

This run confirmed the new 3DGUT compile flags:

- `GAUSSIAN_PARTICLE_MAX_ALPHA=0.99`
- `PARTICLE_FEATURE_HALF=0`
- `FEATURE_OUTPUT_HALF=0`
- `GAUSSIAN_K_BUFFER_SIZE=0`
- `normalize_world_space: true`
- `use_observation_points: false`

Garden metrics:

| Metric | Previous 3DGRUT | Alpha-0.99 3DGRUT | Reference NHT | Alpha-0.99 - Previous | Alpha-0.99 - Reference |
|---|---:|---:|---:|---:|---:|
| PSNR | 27.6621 | 27.6995 | 28.1193 | +0.0375 | -0.4197 |
| SSIM | 0.8645 | 0.8647 | 0.8753 | +0.0001 | -0.0106 |
| LPIPS | 0.1311 | 0.1306 | 0.1227 | -0.0005 | +0.0079 |

Conclusion: matching the alpha cap helped only slightly. The remaining `garden` gap is still about `0.42 dB`.

Converted the alpha-0.99 checkpoint to reference gsplat/NHT format:

```text
results/nht_step2_parity/garden_alpha099_as_gsplat/ckpts/ckpt_29999_rank0.pt
```

Run this next to check whether same-checkpoint reference-render parity improved:

```bash
cd /home/qiwu/Work/3dgrt-external/thirdparty/neural-harmonic-textures
CUDA_VISIBLE_DEVICES=0 python gsplat/examples/simple_trainer_nht.py default \
  --disable_viewer --disable_video \
  --data_dir /media/data0/datasets/mipnerf360/garden \
  --data_factor 4 \
  --result_dir /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/garden_alpha099_as_gsplat_eval \
  --strategy.cap-max 1000000 \
  --render_traj_path ellipse \
  --ckpt /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/garden_alpha099_as_gsplat/ckpts/ckpt_29999_rank0.pt
```

Then compare:

```bash
cd /home/qiwu/Work/3dgrt-external
python tools/nht_step2_parity.py saved-png-compare \
  --grut-renders results/mipnerf360_nht_alpha099/garden/eval/garden/garden-0107_115458/ours_30000/renders \
  --ref-renders results/nht_step2_parity/garden_alpha099_as_gsplat_eval/renders \
  --ref-glob 'val_step*.png'
```

Alpha-0.99 same-checkpoint reference render result:

| Metric | Alpha-0.99 3DGRUT renderer | Alpha-0.99 converted checkpoint via reference renderer | Reference NHT |
|---|---:|---:|---:|
| PSNR | 27.6995 | 25.8528 | 28.1193 |
| SSIM | 0.8647 | 0.7967 | 0.8753 |
| LPIPS | 0.1306 | 0.1514 | 0.1227 |

This is essentially unchanged from the previous converted-checkpoint reference render. The alpha cap fixed only a small constant mismatch; it did not fix the same-checkpoint renderer/convention mismatch.

New concrete mismatch found after the alpha-0.99 run:

- Reference gsplat/NHT creates `UnscentedTransformParameters()` with defaults `alpha=0.1`, `beta=2.0`, `kappa=0.0`.
- 3DGRUT 3DGUT NHT was inheriting `render.splat.ut_alpha: 1.0`, confirmed in `parsed.yaml` and the compile line as `GAUSSIAN_UT_ALPHA=1.0`.
- Patched the NHT 3DGUT app configs to override only NHT parity runs to `render.splat.ut_alpha: 0.1`:

```text
configs/apps/colmap_3dgut_mcmc_nht.yaml
configs/apps/nerf_synthetic_3dgut_mcmc_nht.yaml
```

Next run should confirm the extension rebuilds with:

```text
GAUSSIAN_UT_ALPHA=0.1
GAUSSIAN_UT_DELTA=0.17320508075688773
```

UT-alpha rerun update:

```text
results/mipnerf360_nht_utalpha01
```

This run confirmed:

- `GAUSSIAN_UT_ALPHA=0.1`
- `GAUSSIAN_UT_DELTA=0.17320508075688776`
- `GAUSSIAN_PARTICLE_MAX_ALPHA=0.99`
- `normalize_world_space: true`
- `use_observation_points: false`

Garden metrics:

| Metric | Previous 3DGRUT | Alpha-0.99 3DGRUT | UT-alpha-0.1 3DGRUT | Reference NHT |
|---|---:|---:|---:|---:|
| PSNR | 27.6621 | 27.6995 | 27.6651 | 28.1193 |
| SSIM | 0.8645 | 0.8647 | 0.8646 | 0.8753 |
| LPIPS | 0.1311 | 0.1306 | 0.1308 | 0.1227 |

Conclusion: matching gsplat's UT alpha did not recover the quality gap in the 3DGRUT renderer. Keep it as a reference-parity constant for now, but stop doing scalar ablations and move to same-frame tensor dumps.

Added `tools/nht_step2_parity.py` dump commands:

- `dump-grut-frame`: dumps 3DGRUT raw encoded features, alpha, decoder ray directions, decoded RGB, final RGB, and GT for one val frame.
- `dump-ref-frame`: dumps the same tensors from the reference gsplat/NHT rasterization path for a converted checkpoint.
- `compare-dumps`: reports PSNR/MAE/max-abs for matching tensors.

Converted the UT-alpha-0.1 checkpoint to reference gsplat/NHT format:

```text
results/nht_step2_parity/garden_utalpha01_as_gsplat/ckpts/ckpt_29999_rank0.pt
```

Run the tensor dumps next:

```bash
# In the 3DGRUT environment
CUDA_VISIBLE_DEVICES=0 python tools/nht_step2_parity.py dump-grut-frame \
  --grut-ckpt results/mipnerf360_nht_utalpha01/garden/garden-0107_122327/ckpt_last.pt \
  --data-dir /media/data0/datasets/mipnerf360/garden \
  --out results/nht_step2_parity/dumps/garden_utalpha01_frame000_grut.pt \
  --frame-index 0
```

```bash
# In the reference NHT environment
CUDA_VISIBLE_DEVICES=0 python /home/qiwu/Work/3dgrt-external/tools/nht_step2_parity.py dump-ref-frame \
  --nht-root /home/qiwu/Work/3dgrt-external/thirdparty/neural-harmonic-textures \
  --ref-ckpt /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/garden_utalpha01_as_gsplat/ckpts/ckpt_29999_rank0.pt \
  --data-dir /media/data0/datasets/mipnerf360/garden \
  --out /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/dumps/garden_utalpha01_frame000_ref.pt \
  --frame-index 0 \
  --data-factor 4
```

```bash
python tools/nht_step2_parity.py compare-dumps \
  --grut-dump results/nht_step2_parity/dumps/garden_utalpha01_frame000_grut.pt \
  --ref-dump results/nht_step2_parity/dumps/garden_utalpha01_frame000_ref.pt
```

Tensor dump result for `garden`, frame 0, UT-alpha-0.1 same checkpoint:

| Tensor | PSNR | MAE | Max Abs |
|---|---:|---:|---:|
| GT | 38.2753 | 0.0080 | 0.1608 |
| alpha | 33.5441 | 0.0121 | 0.3127 |
| features | 30.1541 | 0.0189 | 0.6721 |
| ray_dirs_mlp | 67.1446 | 0.0003 | 0.0011 |
| ray_dirs_world | 70.6664 | 0.0002 | 0.0007 |
| decoded_rgb | 29.4219 | 0.0189 | 0.6157 |
| final_rgb | 29.4219 | 0.0189 | 0.6157 |

Conclusion: decoder ray-direction convention is not the blocker. The mismatch enters before the decoder, in alpha and integrated encoded features.

Reference NHT UT rendering uses:

- `fully_fused_projection_with_ut(... opacities=opacities, eps2d=0.3, global_z_order=True)`
- `isect_tiles(... conics=None, opacities=None)` for UT paths

That means reference uses opacity-aware projected radii, then plain AABB tile intersection. 3DGRUT was still doing additional per-tile conic pruning through `render.splat.tile_based_culling: true`.

Applied next parity config change, NHT app configs only:

```yaml
render:
  splat:
    tile_based_culling: false
```

Keep these enabled for now because they match reference projection intent:

```yaml
render:
  splat:
    rect_bounding: true
    tight_opacity_bounding: true
```

Next run should confirm:

```text
GAUSSIAN_TILE_BASED_CULLING=false
GAUSSIAN_RECT_BOUNDING=true
GAUSSIAN_TIGHT_OPACITY_BOUNDING=true
GAUSSIAN_UT_ALPHA=0.1
GAUSSIAN_PARTICLE_MAX_ALPHA=0.99
```

No-tile-cull rerun update:

```text
results/mipnerf360_nht_no_tile_cull
```

The compile flags were correct:

- `GAUSSIAN_TILE_BASED_CULLING=false`
- `GAUSSIAN_RECT_BOUNDING=true`
- `GAUSSIAN_TIGHT_OPACITY_BOUNDING=true`
- `GAUSSIAN_UT_ALPHA=0.1`
- `GAUSSIAN_PARTICLE_MAX_ALPHA=0.99`
- `PARTICLE_FEATURE_HALF=0`
- `FEATURE_OUTPUT_HALF=0`

Garden metrics:

| Metric | No Tile Culling 3DGRUT | Reference NHT | Delta |
|---|---:|---:|---:|
| PSNR | 27.6864 | 28.1193 | -0.4329 |
| SSIM | 0.8646 | 0.8753 | -0.0108 |
| LPIPS | 0.1310 | 0.1227 | +0.0083 |

Conclusion: extra 3DGUT tile-level culling is not the main cause of the gap.

Exclusive-termination ablation:

```text
results/mipnerf360_nht_exclusive_term
```

Garden metrics:

| Metric | Exclusive Termination 3DGRUT | Reference NHT | Delta |
|---|---:|---:|---:|
| PSNR | 27.677 | 28.1193 | -0.4423 |
| SSIM | 0.8647 | 0.8753 | -0.0106 |
| LPIPS | 0.1310 | 0.1227 | +0.0083 |

Conclusion: exclusive transmittance termination is not the missing parity fix. The diagnostic compile switch and renderer branches were removed to keep the final parity patch focused.

New strongest lead: dataset image protocol.

- Tensor dumps for the same frame showed ray directions match closely, but `gt` itself differs (`PSNR=38.2753`, `MAE=0.0080`).
- Reference NHT's default `data_factor` protocol does not load `images_4` directly for Mip-NeRF 360 if it contains JPGs. It resizes full-resolution `images/` with PIL bicubic into `images_4_png/` and loads those PNGs.
- 3DGRUT was loading `images_4` directly. That creates a different training/eval target and slightly different intrinsics because the reference uses rounded PNG dimensions.

Applied COLMAP dataset parity switch, enabled for the NHT COLMAP app only:

```yaml
dataset:
  normalize_world_space: true
  gsplat_image_downscale: true
```

Implementation files:

```text
configs/dataset/colmap.yaml
configs/apps/colmap_3dgut_mcmc_nht.yaml
threedgrut/datasets/__init__.py
threedgrut/datasets/dataset_colmap.py
```

Next run should confirm the dataset logs use `images_4_png` behavior. If `images_4_png` already exists from a reference run, 3DGRUT will reuse it. If it is missing and `images_4` contains JPGs, 3DGRUT will generate it from `images/` with bicubic resizing and `.png` names.

Suggested rerun:

```bash
GPU=0 \
  DATA_ROOT=/media/data0/datasets/mipnerf360 \
  SCENE_LIST=garden \
  RESULT_DIR=results/mipnerf360_nht_gsplat_images \
  FEATURE_DIM=48 \
  TRAIN_PARTICLE_FEATURE_HALF=false \
  TRAIN_FEATURE_OUTPUT_HALF=false \
  CAP_MAX=1000000 \
  bash scripts/benchmark/mipnerf360_nht.sh apps/colmap_3dgut_mcmc_nht
```

GSplat image-protocol rerun update:

```text
results/mipnerf360_nht_gsplat_images
```

Garden metrics:

| Metric | GSplat Image Protocol 3DGRUT | Reference NHT | Delta |
|---|---:|---:|---:|
| PSNR | 28.124 | 28.1193 | +0.0047 |
| SSIM | 0.8728 | 0.8753 | -0.0025 |
| LPIPS | 0.1228 | 0.1227 | +0.0001 |

Conclusion: the dominant quality regression was the dataset image/downsample protocol. Matching GSplat's `images_{factor}_png` path closes the garden PSNR and LPIPS gap. Remaining SSIM difference is small enough to check on the full sweep before spending more time on renderer internals.

One-frame tensor parity check after the GSplat image-protocol fix:

```text
results/nht_step2_parity/dumps/garden_gsplat_images_frame000_grut.pt
results/nht_step2_parity/dumps/garden_gsplat_images_frame000_ref.pt
```

| Tensor | Exact | Diff Elements | PSNR | MAE | Max Abs |
|---|---:|---:|---:|---:|---:|
| GT | true | 0 / 3,268,440 | 120.0000 | 0.00000000 | 0.00000000 |
| alpha | false | 1,056,270 / 1,089,480 | 76.2430 | 0.00001508 | 0.01431763 |
| features | false | 26,144,323 / 26,147,520 | 70.2700 | 0.00005342 | 0.12407559 |
| ray_dirs_mlp | false | 2,918,226 / 3,268,440 | 120.0000 | 0.00000023 | 0.00000075 |
| ray_dirs_world | false | 2,976,065 / 3,268,440 | 120.0000 | 0.00000014 | 0.00000048 |
| decoded_rgb | false | 929,956 / 3,268,440 | 72.0049 | 0.00007912 | 0.05395508 |
| final_rgb | false | 929,956 / 3,268,440 | 72.0049 | 0.00007912 | 0.05395508 |

Conclusion: this is not byte-for-byte renderer parity, but it is strong numerical renderer parity. The previous large tensor mismatch was dominated by the image/GT protocol. After fixing that, the remaining same-checkpoint final RGB mismatch is about `72 dB` PSNR with very small MAE.

- As a saved-artifact sanity check, comparing existing 3DGRUT garden renders to the reference side-by-side PNGs gives:

| Comparison | Count | Mean PSNR | Min PSNR | Max PSNR | Mean MAE |
|---|---:|---:|---:|---:|---:|
| 3DGRUT render vs reference GT half | 24 | 27.7231 | 22.1510 | 30.5506 | 0.026638 |
| 3DGRUT render vs reference predicted half | 24 | 31.5792 | 24.2574 | 34.3149 | 0.017426 |

The right-half comparison confirms frame ordering/alignment, but it is not the decisive same-checkpoint renderer parity check.

Run this on a CUDA-visible machine to finish Step 2:

```bash
cd /home/qiwu/Work/3dgrt-external/thirdparty/neural-harmonic-textures
CUDA_VISIBLE_DEVICES=0 python gsplat/examples/simple_trainer_nht.py default \
  --disable_viewer --disable_video \
  --data_dir /media/data0/datasets/mipnerf360/garden \
  --data_factor 4 \
  --result_dir /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/garden_grut_as_gsplat_eval \
  --strategy.cap-max 1000000 \
  --render_traj_path ellipse \
  --ckpt /home/qiwu/Work/3dgrt-external/results/nht_step2_parity/garden_grut_as_gsplat/ckpts/ckpt_29999_rank0.pt
```

Then compare against the existing or regenerated 3DGRUT render:

```bash
python tools/nht_step2_parity.py saved-png-compare \
  --grut-renders results/mipnerf360_2293c73d_nht_mcmc/garden/garden-3006_012105/ours_30000/renders \
  --ref-renders results/nht_step2_parity/garden_grut_as_gsplat_eval/renders \
  --ref-glob 'val_step*.png'
```

### Step 3: If Renderer Gap, Localize With A Tiny Deterministic Scene

Construct a tiny fixed scene:

- one camera
- a small number of Gaussians
- fixed opacities/scales/rotations/features
- overlapping pixels where ordering matters

Compare reference gsplat vs 3DGUT outputs at each stage:

- hit/intersection ordering
- alpha per contribution
- accumulated transmittance
- encoded feature accumulation
- final alpha
- decoded RGB

Do not call `k_buffer_size=16` reference parity. It can be used only as an ablation to test whether 3DGUT ordering explains the error.

### Step 4: If Training Gap, Compare State Trajectories

Add or extract comparable stats at:

```text
init
step 6999
step 29999 / 30000
```

Compare:

- number of Gaussians
- opacity histogram
- scale histogram
- feature norm histogram
- decoder LR and feature LR
- MCMC relocate/add counts
- validation PSNR per image
- color refine start and freeze behavior

Reference stat files already exist:

```text
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/train_step6999_rank0.json
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/train_step29999_rank0.json
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/val_step6999.json
thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000/garden/stats/val_step29999.json
```

### Step 5: Full Sweep Only After One-Scene Parity

Once `garden` is explained, rerun all nine Mip-NeRF 360 scenes with the same fix.

Report:

- per-scene PSNR/SSIM/LPIPS
- aggregate mean
- training time
- whether the fix is renderer-only, training-only, or both

## Useful Commands

Aggregate current and reference metrics:

```bash
python3 - <<'PY'
import json, pathlib, statistics
cur = pathlib.Path("results/mipnerf360_2293c73d_nht_mcmc")
ref = pathlib.Path("thirdparty/neural-harmonic-textures/results/nht_mcmc_1000000")
scenes = ["bicycle", "bonsai", "counter", "flowers", "garden", "kitchen", "room", "stump", "treehill"]
for metric_cur, metric_ref in [("mean_psnr", "psnr"), ("mean_ssim", "ssim"), ("mean_lpips", "lpips")]:
    cvals, rvals = [], []
    for scene in scenes:
        cvals.append(json.loads(next((cur / scene).glob("*/metrics.json")).read_text())[metric_cur])
        rvals.append(json.loads((ref / scene / "stats" / "val_step29999.json").read_text())[metric_ref])
    print(metric_cur, statistics.mean(cvals), statistics.mean(rvals), statistics.mean(cvals) - statistics.mean(rvals))
PY
```

Check current 3DGRUT resolved config:

```bash
rg -n "normalize_world_space|use_observation_points|particle_feature_half|feature_output_half|k_buffer_size" \
  results/mipnerf360_2293c73d_nht_mcmc/*/*/parsed.yaml
```

Check 3DGUT compile flags from the run:

```bash
rg -n "PARTICLE_FEATURE_HALF|FEATURE_OUTPUT_HALF|GAUSSIAN_K_BUFFER_SIZE" \
  results/mipnerf360_2293c73d_nht_mcmc/train_bicycle.log
```

## Do Not Conflate These

- `k_buffer_size=16`: useful diagnostic for 3DGUT sorted compositing, not reference parity.
- Reference gsplat NHT: ordered tile-intersection rasterization with eval3D NHT.
- 3DGUT NHT: separate renderer implementation; achieving parity means matching behavior, not matching config field names.
