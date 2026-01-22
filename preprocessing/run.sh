#!/usr/bin/env bash

SCENE="bedroom"
PROJECT="fullcircle"
METHOD="ui"

DATAROOT="data/${PROJECT}"
DATAPATH="${DATAROOT}/${SCENE}"

## omni → 16 prespectives
conda activate 3dgrut_pycolmap
cd colmap
python python/examples/16/pano_camera.py \
    --input_image_path ${DATAPATH}/omni \
    --output_path      ${DATAPATH}/pre_masking/images

## mask perspectives
conda activate sam
python python/examples/16/mask_frames.py \
    ${DATAPATH}/pre_masking/images \
    ${DATAPATH}/pre_masking/masks

## prespective → omni
conda activate 3dgrut_pycolmap
python python/examples/16/pano_masks_reverse.py \
    --input_path ${DATAPATH}/pre_masking/masks \
    --output_path ${DATAPATH}/pre_masking/reconstructed_masks

## omni → synthetic fisheye
cd ..
python step1-o2s.py --scene "${SCENE}" --project "${PROJECT}"

## mask synthetic fisheye
# python scripts/tracking_gui.py --frames-path /home/youlenda/360/omni/3dgrut/data/flat_maria/pre_masking/fisheye_syn --output-path /home/youlenda/360/omni/3dgrut/data/flat_maria/out-8

## synthetic fisheye → omni
python step2-s2o.py --scene "${SCENE}" --method "${METHOD}" --project "${PROJECT}"

## omni → fisheye
python step3-o2f.py --scene "${SCENE}" --method "${METHOD}" --project "${PROJECT}"

## dilate masks
for CAM in front rear; do
  python step4-dilate.py --scene "${SCENE}" --method "${METHOD}" --camera "${CAM}" --iter 1  --project "${PROJECT}"
  python step4-dilate.py --scene "${SCENE}" --method "${METHOD}" --camera "${CAM}" --iter 20 --project "${PROJECT}"
done
