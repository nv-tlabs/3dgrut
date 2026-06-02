#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reference-style NHT validation on Mip-NeRF 360.
#
# Usage:
#   bash benchmark/mipnerf360_nht.sh
#   bash benchmark/mipnerf360_nht.sh apps/colmap_3dgut_mcmc_nht
#   GPU=1 FEATURE_DIM=48 SCENE_LIST="garden bonsai" bash benchmark/mipnerf360_nht.sh
#   GPU=1 SCENE_LIST=bonsai PAPER_HIGH=1 NHT_FEATURE_LR=0.03 bash benchmark/mipnerf360_nht.sh
#
# Extra Hydra overrides may be appended after the optional config name.

set -euo pipefail

is_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

CONFIG=${CONFIG:-"apps/colmap_3dgut_mcmc_nht"}
if [[ $# -gt 0 && "$1" != *=* ]]; then
    CONFIG="$1"
    shift
fi
CONFIG=${CONFIG%.yaml}
EXTRA_ARGS=("$@")

DATA_ROOT=${DATA_ROOT:-"/mnt/gogn/data/nerf_datasets/nerf_360"}
RESULT_DIR=${RESULT_DIR:-"results/mipnerf360_nht"}
GPU=${GPU:-0}
CAP_MAX=${CAP_MAX:-1000000}
MAX_STEPS=${MAX_STEPS:-30000}
FEATURE_DIM=${FEATURE_DIM:-64}
LAMBDA_L1=${LAMBDA_L1:-0.8}
LAMBDA_SSIM=${LAMBDA_SSIM:-0.2}
PAPER_HIGH=${PAPER_HIGH:-0}
CENTER_RAY_ENCODING=${CENTER_RAY_ENCODING:-$PAPER_HIGH}
PERTURB_START_ITERATION=${PERTURB_START_ITERATION:-0}
PARTICLE_FEATURE_HALF=${PARTICLE_FEATURE_HALF:-true}
FEATURE_OUTPUT_HALF=${FEATURE_OUTPUT_HALF:-true}
TRAIN_PARTICLE_FEATURE_HALF=${TRAIN_PARTICLE_FEATURE_HALF:-$PARTICLE_FEATURE_HALF}
TRAIN_FEATURE_OUTPUT_HALF=${TRAIN_FEATURE_OUTPUT_HALF:-$FEATURE_OUTPUT_HALF}
NHT_FEATURES_BWD_LOCAL_GRAD_CUDA=${NHT_FEATURES_BWD_LOCAL_GRAD_CUDA:-1}
if is_enabled "$TRAIN_PARTICLE_FEATURE_HALF"; then
    TRAIN_PARTICLE_FEATURE_HALF=true
else
    TRAIN_PARTICLE_FEATURE_HALF=false
fi
if is_enabled "$TRAIN_FEATURE_OUTPUT_HALF"; then
    TRAIN_FEATURE_OUTPUT_HALF=true
else
    TRAIN_FEATURE_OUTPUT_HALF=false
fi
if is_enabled "$CENTER_RAY_ENCODING"; then
    CENTER_RAY_ENCODING=true
else
    CENTER_RAY_ENCODING=false
fi
NHT_MLP_LR=${NHT_MLP_LR:-}
NHT_FEATURE_LR=${NHT_FEATURE_LR:-}
NHT_SCALE_REG=${NHT_SCALE_REG:-}
NHT_OPACITY_REG=${NHT_OPACITY_REG:-}
if is_enabled "$PAPER_HIGH"; then
    NHT_MLP_LR=${NHT_MLP_LR:-0.0072}
    NHT_SCALE_REG=${NHT_SCALE_REG:-0.01}
    NHT_OPACITY_REG=${NHT_OPACITY_REG:-0.02}
fi
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_RENDER=${RUN_RENDER:-1}
SKIP_EXISTING=${SKIP_EXISTING:-0}

M360_INDOOR=("bonsai" "counter" "kitchen" "room")
M360_OUTDOOR=("garden" "bicycle" "stump" "treehill" "flowers")
SCENE_LIST=${SCENE_LIST:-"${M360_INDOOR[*]} ${M360_OUTDOOR[*]}"}

declare -A PAPER_HIGH_CAPS=(
    [bonsai]=1300000
    [counter]=1200000
    [kitchen]=1800000
    [room]=1500000
    [garden]=5200000
    [bicycle]=5900000
    [stump]=4750000
    [treehill]=3500000
    [flowers]=3000000
)

get_factor() {
    case "$1" in
        bonsai|counter|kitchen|room) echo 2 ;;
        *) echo 4 ;;
    esac
}

get_data_dir() {
    local scene=$1
    for p in \
        "$DATA_ROOT/$scene" \
        "$DATA_ROOT/mipnerf360/$scene" \
        "$DATA_ROOT/360_v2/$scene"; do
        if [[ -d "$p" ]]; then
            echo "$p"
            return
        fi
    done
    echo ""
}

latest_ckpt() {
    local scene=$1
    [[ -d "$RESULT_DIR/$scene" ]] || return 0
    find "$RESULT_DIR/$scene" -name ckpt_last.pt 2>/dev/null | sort | tail -n 1
}

mkdir -p "$RESULT_DIR"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"results/.torch_extensions_nht"}
export NHT_FEATURES_BWD_LOCAL_GRAD_CUDA

echo "============================================================"
echo "Mip-NeRF 360 NHT validation"
echo "  Config:   $CONFIG"
echo "  Data:     $DATA_ROOT"
echo "  Results:  $RESULT_DIR"
echo "  Build cache: $TORCH_EXTENSIONS_DIR"
echo "  Features: $FEATURE_DIM"
if is_enabled "$PAPER_HIGH"; then
    echo "  Mode:     paper high-count caps"
    echo "  Cap:      per scene"
else
    echo "  Cap:      $CAP_MAX"
fi
echo "  Steps:    $MAX_STEPS"
echo "  Center ray encoding: $CENTER_RAY_ENCODING"
echo "  Perturb start: $PERTURB_START_ITERATION"
echo "  Train particle feature half:  $TRAIN_PARTICLE_FEATURE_HALF"
echo "  Train feature output half:    $TRAIN_FEATURE_OUTPUT_HALF"
echo "  Render precision: checkpoint config"
echo "  CUDA NHT feature bwd: $NHT_FEATURES_BWD_LOCAL_GRAD_CUDA"
echo "  Loss:     L1=$LAMBDA_L1 SSIM=$LAMBDA_SSIM"
[[ -n "$NHT_FEATURE_LR" ]] && echo "  NHT feature LR: $NHT_FEATURE_LR"
[[ -n "$NHT_MLP_LR" ]] && echo "  NHT MLP LR: $NHT_MLP_LR"
[[ -n "$NHT_SCALE_REG" ]] && echo "  Scale reg:  $NHT_SCALE_REG"
[[ -n "$NHT_OPACITY_REG" ]] && echo "  Opacity reg:$NHT_OPACITY_REG"
echo "  Scenes:   $SCENE_LIST"
echo "============================================================"

for SCENE in $SCENE_LIST; do
    DATA_DIR=$(get_data_dir "$SCENE")
    DATA_FACTOR=$(get_factor "$SCENE")
    SCENE_CAP=$CAP_MAX
    if is_enabled "$PAPER_HIGH"; then
        SCENE_CAP=${PAPER_HIGH_CAPS[$SCENE]:-$CAP_MAX}
    fi

    if [[ -z "$DATA_DIR" ]]; then
        echo "WARNING: data not found for $SCENE under $DATA_ROOT; skipping"
        continue
    fi

    echo
    echo ">>> $SCENE (factor=$DATA_FACTOR, cap=$SCENE_CAP) <<<"

    if [[ "$RUN_TRAIN" == "1" ]]; then
        if [[ "$SKIP_EXISTING" == "1" && -n "$(latest_ckpt "$SCENE")" ]]; then
            echo "  [skip] existing checkpoint: $(latest_ckpt "$SCENE")"
        else
            TRAIN_LOG="$RESULT_DIR/train_$SCENE.log"
            nvidia-smi > "$TRAIN_LOG" 2>&1 || true

            TRAIN_OVERRIDES=(
                "path=$DATA_DIR" \
                "out_dir=$RESULT_DIR" \
                "experiment_name=$SCENE" \
                "dataset.downsample_factor=$DATA_FACTOR" \
                "dataset.load_exif=false" \
                "n_iterations=$MAX_STEPS" \
                "test_last=false" \
                "val_frequency=999999" \
                "strategy.add.max_n_gaussians=$SCENE_CAP" \
                "strategy.perturb.start_iteration=$PERTURB_START_ITERATION" \
                "model.nht_features.dim=$FEATURE_DIM" \
                "model.nht_decoder.center_ray_encoding=$CENTER_RAY_ENCODING" \
                "model.nht_decoder.scheduler.max_steps=$MAX_STEPS" \
                "scheduler.positions.max_steps=$MAX_STEPS" \
                "scheduler.features.max_steps=$MAX_STEPS" \
                "loss.lambda_l1=$LAMBDA_L1" \
                "loss.lambda_ssim=$LAMBDA_SSIM" \
                "render.particle_feature_half=$TRAIN_PARTICLE_FEATURE_HALF" \
                "render.feature_output_half=$TRAIN_FEATURE_OUTPUT_HALF" \
                "checkpoint.iterations=[$MAX_STEPS]" \
                "use_wandb=false" \
                "with_gui=false" \
                "with_viser_gui=false"
            )
            [[ -n "$NHT_FEATURE_LR" ]] && TRAIN_OVERRIDES+=("optimizer.params.features.lr=$NHT_FEATURE_LR")
            [[ -n "$NHT_MLP_LR" ]] && TRAIN_OVERRIDES+=("model.nht_decoder.learning_rate=$NHT_MLP_LR")
            [[ -n "$NHT_SCALE_REG" ]] && TRAIN_OVERRIDES+=("loss.lambda_scale=$NHT_SCALE_REG")
            [[ -n "$NHT_OPACITY_REG" ]] && TRAIN_OVERRIDES+=("loss.lambda_opacity=$NHT_OPACITY_REG")

            CUDA_VISIBLE_DEVICES="$GPU" python train.py --config-name "$CONFIG" \
                "${TRAIN_OVERRIDES[@]}" \
                "${EXTRA_ARGS[@]}" >> "$TRAIN_LOG" 2>&1
        fi
    fi

    if [[ "$RUN_RENDER" == "1" ]]; then
        CKPT=$(latest_ckpt "$SCENE")
        if [[ -z "$CKPT" ]]; then
            echo "WARNING: no ckpt_last.pt found for $SCENE; skipping render"
            continue
        fi

        RENDER_LOG="$RESULT_DIR/render_$SCENE.log"
        CUDA_VISIBLE_DEVICES="$GPU" python render.py \
            --checkpoint "$CKPT" \
            --path "$DATA_DIR" \
            --out-dir "$RESULT_DIR/$SCENE/eval" > "$RENDER_LOG" 2>&1
    fi
done

echo
echo "============================================================"
echo "Results Summary"
echo "============================================================"
RESULT_DIR="$RESULT_DIR" SCENE_LIST="$SCENE_LIST" python - <<'PY'
import glob
import json
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
scenes = os.environ["SCENE_LIST"].split()
indoor = {"bonsai", "counter", "kitchen", "room"}
rows = []

for scene in scenes:
    metric_paths = glob.glob(str(result_dir / scene / "eval" / "**" / "metrics.json"), recursive=True)
    if not metric_paths:
        continue
    latest = max(metric_paths, key=lambda p: os.path.getmtime(p))
    with open(latest) as f:
        metrics = json.load(f)
    rows.append((scene, metrics))

print("| Scene | PSNR | SSIM | LPIPS | Time ms/frame |")
print("|---|---:|---:|---:|---:|")
for scene, m in rows:
    print(
        f"| {scene} | {m.get('mean_psnr', float('nan')):.3f} | "
        f"{m.get('mean_ssim', float('nan')):.4f} | "
        f"{m.get('mean_lpips', float('nan')):.4f} | "
        f"{m.get('mean_inference_time_ms', float('nan')):.2f} |"
    )

def avg(label, subset):
    vals = [(s, m) for s, m in rows if s in subset]
    if not vals:
        return
    def mean(key):
        return sum(m.get(key, 0.0) for _, m in vals) / len(vals)
    print(
        f"{label}: PSNR={mean('mean_psnr'):.3f}, "
        f"SSIM={mean('mean_ssim'):.4f}, "
        f"LPIPS={mean('mean_lpips'):.4f}, "
        f"time={mean('mean_inference_time_ms'):.2f} ms/frame"
    )

avg("M360-In", indoor)
avg("M360-Out", set(scenes) - indoor)
avg("M360", set(scenes))
PY
