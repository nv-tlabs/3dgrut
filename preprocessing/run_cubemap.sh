METHOD="ui-6"

PROJECT="plus"
SCENE="bd"
DATAROOT="/home/youlenda/360/omni/3dgrut/data/${PROJECT}/"
DATAPATH="${DATAROOT}/${SCENE}"

cd preprocessing

conda activate 3dgrut_pycolmap
CUDA_VISIBLE_DEVICES=2 python cubemap/pano_camera.py \
    --input_image_path ${DATAPATH}/omni \
    --output_path      ${DATAPATH}/pre_masking_6/images_6

conda activate sam
CUDA_VISIBLE_DEVICES=2 python cubemap/mask_frames.py ${DATAPATH}/pre_masking_6/images_6 ${DATAPATH}/pre_masking_6/masks

conda activate 3dgrut_pycolmap
CUDA_VISIBLE_DEVICES=2 python cubemap/pano_masks_reverse.py \
    --input_path ${DATAPATH}/pre_masking_6/masks \
    --output_path ${DATAPATH}/pre_masking_6/reconstructed_masks



python step1-o2s.py --scene "${SCENE}" --project "${PROJECT}"

# masking
SCENE="party_room_shadow_2"
python scripts/tracking_gui.py --frames-path /home/youlenda/360/omni/3dgrut/data/fullcircle/${SCENE}/pre_masking_6/fisheye_syn --output-path /home/youlenda/360/omni/3dgrut/data/fullcircle/${SCENE}/out-ui-6/querry

#########


python step2-s2o.py --scene "${SCENE}" --method "${METHOD}" --project "${PROJECT}"

python step3-o2f.py   --scene "${SCENE}" --method "${METHOD}" --project "${PROJECT}"

for CAM in front rear; do
  python step4-dilate.py --scene "${SCENE}" --method "${METHOD}" --camera "${CAM}" --iter 1  --project "${PROJECT}"
  python step4-dilate.py --scene "${SCENE}" --method "${METHOD}" --camera "${CAM}" --iter 20 --project "${PROJECT}"
done

# CUDA_VISIBLE_DEVICES=0 python python/examples/6/pano_masks_reverse.py \
#     --input_path ${DATAPATH}/pre_masking_6/images_6 \
#     --output_path ${DATAPATH}/pre_masking_6/reconstructed_imags