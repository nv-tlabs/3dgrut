#!/usr/bin/env bash

DATAPATH="$HOME/360/omni/3dgrut/data/bimodel/drawer"
GPU=1
##################################################################################################
# 1
# Feature extraction
CUDA_VISIBLE_DEVICES=2 colmap312 feature_extractor \
    --image_path ${DATAPATH}/images \
    --ImageReader.mask_path ${DATAPATH}/masks \
    --database_path ${DATAPATH}/database.db \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_model OPENCV_FISHEYE

# 2
# Matching
CUDA_VISIBLE_DEVICES=${GPU} colmap312 exhaustive_matcher \
    --database_path ${DATAPATH}/database.db

# 3
# Sparse reconstruction
mkdir -p ${DATAPATH}/sparse
CUDA_VISIBLE_DEVICES=${GPU} colmap312 mapper \
    --image_path ${DATAPATH}/images \
    --database_path ${DATAPATH}/database.db \
    --output_path ${DATAPATH}/sparse