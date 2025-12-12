#!/bin/bash

# Usage: ./run_export.sh <input_ply_file> [output_usdz_file]
#
# <input_ply_file>: Path to the input PLY file.
# [output_usdz_file]: Optional path for the output USDZ file. If not provided,
#                     it will be derived from the input PLY file (e.g., model.ply -> model.usdz).

if [ -z "$1" ]; then
  echo "Error: Input PLY file path is required."
  echo "Usage: ./run_export.sh <input_ply_file> [output_usdz_file]"
  echo "Example: ./run_export.sh my_model.ply my_model.usdz"
  exit 1
fi

INPUT_PLY_FILE=$1
OUTPUT_USDZ_FILE=$2

# If output USDZ file is not provided, derive it from the input PLY file
if [ -z "$OUTPUT_USDZ_FILE" ]; then
  BASE_NAME=$(basename -- "$INPUT_PLY_FILE")
  DIR_NAME=$(dirname -- "$INPUT_PLY_FILE")
  FILENAME_NO_EXT="${BASE_NAME%.*}"
  OUTPUT_USDZ_FILE="${DIR_NAME}/${FILENAME_NO_EXT}.usdz"
  echo "Output USDZ file not specified. Defaulting to: $OUTPUT_USDZ_FILE"
fi

echo "----------------------------------------"
echo "Starting PLY to USDZ conversion"
echo "Input PLY:   $INPUT_PLY_FILE"
echo "Output USDZ: $OUTPUT_USDZ_FILE"
echo "----------------------------------------"

# Run the conversion command
python -m threedgrut.export.scripts.ply_to_usd "$INPUT_PLY_FILE" --output_file "$OUTPUT_USDZ_FILE"

echo "----------------------------------------"
echo "Conversion complete."
echo "----------------------------------------"
