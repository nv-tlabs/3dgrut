#!/bin/bash

# List of primitive types to profile
PRIMITIVES=("icosahedron" "octahedron" "trihexa" "trisurfel" "tetrahedron" "sphere" "diamond")

# Dataset and output configuration
DATA_PATH="data/360_extra_scenes/flowers"
OUT_DIR="runs/experiment"
TRACE_DIR="traces/primitive_comparison"

# Create trace output base directory if needed
mkdir -p "$TRACE_DIR"

# Loop through each primitive type
for primitive in "${PRIMITIVES[@]}"
do
    echo "Profiling training with primitive: $primitive"

    # Set output paths
    PRIM_TRACE_DIR="${TRACE_DIR}/${primitive}"
    mkdir -p "$PRIM_TRACE_DIR"

    # Profile the Python training command
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --output="${PRIM_TRACE_DIR}/nsys_trace" \
        python train.py --config-name paper/3dgrt/${primitive}_7000.yaml \
            path="$DATA_PATH" \
            out_dir="$OUT_DIR" \
            experiment_name="nerf_${primitive}" || {
                echo "Training failed for primitive: $primitive"
                continue
            }
done
