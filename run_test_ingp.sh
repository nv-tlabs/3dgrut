#!/bin/bash
INGP_DIR=$1
DATA_DIR=$2
OUT_DIR=$3

COLMAP_CMD="python train.py --config-name apps/colmap_test_ingp.yaml out_dir=${OUT_DIR}"

NERF360_INDOOR_SEQ=(
    room
    bonsai
    counter
    kitchen
)
for i in "${NERF360_INDOOR_SEQ[@]}"; do
    echo "$COLMAP_CMD path=${DATA_DIR}/nerf_360/$i dataset.downsample_factor=2 group_name=mipnerf360 import_ingp.path=${INGP_DIR}/$i.ingp ${@:4}";
    $COLMAP_CMD "path=${DATA_DIR}/nerf_360/$i" "dataset.downsample_factor=2" "group_name=mipnerf360" "import_ingp.path=${INGP_DIR}/$i.ingp" "${@:4}";
done

NERF360_OUTDOOR_SEQ=(
    bicycle
    garden
    stump
)
for i in "${NERF360_OUTDOOR_SEQ[@]}"; do
    echo "$COLMAP_CMD dataset.downsample_factor=4 path=${DATA_DIR}/nerf_360/$i group_name=mipnerf360 import_ingp.path=${INGP_DIR}/$i.ingp ${@:4}";
    $COLMAP_CMD "path=${DATA_DIR}/nerf_360/$i" "dataset.downsample_factor=4" "group_name=mipnerf360" "import_ingp.path=${INGP_DIR}/$i.ingp" "${@:4}";
done

SYNTH_CMD="python train.py --config-name apps/nerf_synthetic_test_ingp.yaml"

SYNTH_SEQ=(
    chair
    drums
    ficus
    hotdog
    lego
    materials
    mic
    ship
)
for i in "${SYNTH_SEQ[@]}"; do
    echo "$SYNTH_CMD path=${DATA_DIR}/nerf_synthetic/$i group_name=nerf_synthetic import_ingp.path=${INGP_DIR}/$i.ingp ${@:4}";
    $SYNTH_CMD "path=${DATA_DIR}/nerf_synthetic/$i" "group_name=nerf_synthetic" "import_ingp.path=${INGP_DIR}/$i.ingp" "${@:4}";
done

TANDT_SEQ=(
    train
    truck
)
for i in "${TANDT_SEQ[@]}"; do
    echo "$COLMAP_CMD path=${DATA_DIR}/nerf_tandt/$i dataset.downsample_factor=1 group_name=tandt import_ingp.path=${INGP_DIR}/$i.ingp ${@:4}";
    $COLMAP_CMD "path=${DATA_DIR}/nerf_tandt/$i" "dataset.downsample_factor=1" "group_name=tandt" "import_ingp.path=${INGP_DIR}/$i.ingp" "${@:4}";
done

DB_SEQ=(
    drjohnson
    playroom
)
for i in "${DB_SEQ[@]}"; do
    echo "$COLMAP_CMD path=${DATA_DIR}/nerf_tandt/$i dataset.downsample_factor=1 group_name=db import_ingp.path=${INGP_DIR}/$i.ingp ${@:4}";
    $COLMAP_CMD "path=${DATA_DIR}/nerf_tandt/$i" "dataset.downsample_factor=1" "group_name=db" "import_ingp.path=${INGP_DIR}/$i.ingp" "${@:4}";
done