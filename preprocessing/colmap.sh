#!/usr/bin/env bash

SCENE="flat_yalda"
PROJECT="fullcircle"

DATAROOT="data/${PROJECT}"
DATAPATH="${DATAROOT}/${SCENE}"

## step 1
## feature extraction
CUDA_VISIBLE_DEVICES=2 colmap312 feature_extractor \
    --image_path ${DATAPATH}/images \
    # --ImageReader.mask_path ${DATAPATH}/masks \
    --database_path ${DATAPATH}/database.db \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_model OPENCV_FISHEYE

## step 2
## matching
CUDA_VISIBLE_DEVICES=2 colmap312 exhaustive_matcher \
    --database_path ${DATAPATH}/database.db

## step 3
## sparse reconstruction
mkdir -p ${DATAPATH}/sparse
CUDA_VISIBLE_DEVICES=2 colmap312 mapper \
    --image_path ${DATAPATH}/images \
    --database_path ${DATAPATH}/database.db \
    --output_path ${DATAPATH}/sparse