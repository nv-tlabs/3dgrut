#!/bin/bash

# Usage: ./run_train.sh <dataset_path> <experiment_name> [method]
# method can be '3dgut' (default) or '3dgrt'

if [ -z "$1" ]; then
  echo "Error: Dataset path is required."
  echo "Usage: ./run_train.sh <dataset_path> <experiment_name> [method]"
  echo "Example: ./run_train.sh /path/to/my_data my_first_scene"
  exit 1
fi

if [ -z "$2" ]; then
  echo "Error: Experiment name is required."
  echo "Usage: ./run_train.sh <dataset_path> <experiment_name> [method]"
  exit 1
fi

DATASET_PATH=$1
EXPERIMENT_NAME=$2
METHOD=${3:-3dgut} # Default to 3dgut if not provided
OUT_DIR="runs"

echo "----------------------------------------"
echo "Starting training with the following settings:"
echo "Dataset:    $DATASET_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo "Method:     $METHOD"
echo "Output Dir: $OUT_DIR"
echo "----------------------------------------"

# Select the appropriate configuration file based on the method
if [ "$METHOD" == "3dgrt" ]; then
    CONFIG="apps/colmap_3dgrt.yaml"
elif [ "$METHOD" == "3dgut" ]; then
    CONFIG="apps/colmap_3dgut.yaml"
else
    echo "Error: Method must be '3dgrt' or '3dgut'"
    exit 1
fi

# Run the training command
python train.py --config-name "$CONFIG" \
    path="$DATASET_PATH" \
    out_dir="$OUT_DIR" \
    experiment_name="$EXPERIMENT_NAME"
