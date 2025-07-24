#!/bin/bash

OUT_DIR=results/scivis

export TORCH_EXTENSIONS_DIR=$OUT_DIR/.cache
mkdir -p $OUT_DIR

PREFIX=0155_00225_055_den

# Pre-generate all the scene IDs
NUMBERS=($(seq -w 200 2 1398))
runjob() {
    local id=$1

    local scene=${PREFIX}_0${NUMBERS[id]}
    
    # skip if "Training Complete." is in the log
    if grep -q "Training Complete." $OUT_DIR/nyx_${PREFIX}_$scene.log; then
        echo "=== Skipping scene: $scene ==="
        return
    fi

    echo "=== Running scene: $scene ==="

    python train.py --config-name apps/scivis_3dgut_gs.yaml \
        path=data/scivisgs/v0.1/nyx/$scene out_dir=$OUT_DIR > $OUT_DIR/nyx_${PREFIX}_$scene.log
}

# Command to submit: sbatch --array=0-1 benchmark/slurm_scivis.sh
# 600 in total

# for i in {3..200}; do
#     runjob $i
# done

for i in {0..599}; do
    runjob $i
done
