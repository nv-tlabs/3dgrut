#!/bin/bash

scenes=("lego" "chair" "drums" "ficus" "hotdog" "materials" "mic" "ship")

for scene in "${scenes[@]}"; do
    echo "Profiling scene: $scene"
    nsys profile -o profile_report_$scene \
        --trace=nvtx \
        --stop-on-exit=true \
        --force-overwrite=true \
        python train.py --config-name paper/3dgrt/nerf_synthetic_ours_reference_inference.yaml path=data/nerf_synthetic/$scene
done

for scene in "${scenes[@]}"; do
    nsys export --type json --output profile_report_${scene}.json profile_report_${scene}.nsys-rep --force-overwrite=true
done
