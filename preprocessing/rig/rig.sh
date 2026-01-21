#!/bin/bash

# ==========================================
# CONFIGURATION
# Users should modify these paths before running
# ==========================================

# Path to the specific dataset root folder
DATASET_PATH="./data/my_dataset"

# Path to COLMAP executable (or alias)
# If colmap is in your global PATH, just use "colmap"
COLMAP_EXE="colmap"

# Path to COLMAP python scripts (for panorama_sfm.py)
COLMAP_SCRIPTS_PATH="./colmap/python/examples"

# Camera Model settings
CAMERA_MODEL="OPENCV_FISHEYE"

# ==========================================
# WORKFLOWS
# ==========================================

# Workflow 1: Unknown Rig Configuration
# Pipeline: Feature Extract -> Match -> Initial Sparse -> Compute Rig -> Match with Rig -> Final Sparse -> BA
run_unknown_rig_pipeline() {
    echo "Starting Unknown Rig Pipeline for: ${DATASET_PATH}"

    # Feature Extraction
    ${COLMAP_EXE} feature_extractor \
        --image_path "${DATASET_PATH}/images" \
        --database_path "${DATASET_PATH}/database.db" \
        --ImageReader.single_camera_per_folder 1 \
        --ImageReader.camera_model ${CAMERA_MODEL}

    # Sequential Matching (Initial)
    ${COLMAP_EXE} sequential_matcher \
        --database_path "${DATASET_PATH}/database.db"

    # Optional: Exhaustive matcher for enhanced scenes (Uncomment if needed)
    # ${COLMAP_EXE} exhaustive_matcher --database_path "${DATASET_PATH}/database.db"

    # Initial Reconstruction (No Rig)
    mkdir -p "${DATASET_PATH}/sparse-initial"
    ${COLMAP_EXE} mapper \
        --image_path "${DATASET_PATH}/images" \
        --database_path "${DATASET_PATH}/database.db" \
        --output_path "${DATASET_PATH}/sparse-initial"

    # Compute Rig Poses
    mkdir -p "${DATASET_PATH}/sparse-with-rigs"
    ${COLMAP_EXE} rig_configurator \
        --database_path "${DATASET_PATH}/database.db" \
        --input_path "${DATASET_PATH}/sparse-initial/0" \
        --rig_config_path "${DATASET_PATH}/rig_config.json" \
        --output_path "${DATASET_PATH}/sparse-with-rigs"

    # Re-match with Rig Configuration
    ${COLMAP_EXE} sequential_matcher \
        --database_path "${DATASET_PATH}/database.db"

    # Final Mapper Run
    mkdir -p "${DATASET_PATH}/sparse-final"
    ${COLMAP_EXE} mapper \
        --image_path "${DATASET_PATH}/images" \
        --database_path "${DATASET_PATH}/database.db" \
        --output_path "${DATASET_PATH}/sparse-final"

    # Bundle Adjustment (Refinement)
    mkdir -p "${DATASET_PATH}/sparse-refined"
    ${COLMAP_EXE} bundle_adjuster \
        --input_path "${DATASET_PATH}/sparse-final/0" \
        --output_path "${DATASET_PATH}/sparse-refined"
        
    # Convert model to TXT (Optional)
    mkdir -p "${DATASET_PATH}/colmap_txt"
    ${COLMAP_EXE} model_converter \
        --input_path "${DATASET_PATH}/sparse-refined" \
        --output_path "${DATASET_PATH}/colmap_txt" \
        --output_type TXT
}

# Workflow 2: Known Rig Poses & Intrinsics
# Pipeline: Feature Extract -> Rig Config -> Match -> Mapper (Fixed Intrinsics) -> BA
run_known_rig_pipeline() {
    echo "Starting Known Rig Pipeline for: ${DATASET_PATH}"

    # Feature Extraction
    ${COLMAP_EXE} feature_extractor \
        --image_path "${DATASET_PATH}/images" \
        --database_path "${DATASET_PATH}/database.db" \
        --ImageReader.single_camera_per_folder 1 \
        --ImageReader.camera_model ${CAMERA_MODEL}

    # Rig Configuration
    ${COLMAP_EXE} rig_configurator \
        --database_path "${DATASET_PATH}/database.db" \
        --rig_config_path "${DATASET_PATH}/rig_config.json"

    # Sequential Matching
    ${COLMAP_EXE} sequential_matcher \
        --database_path "${DATASET_PATH}/database.db"

    # Mapper (Keeping intrinsics fixed)
    mkdir -p "${DATASET_PATH}/sparse-final"
    ${COLMAP_EXE} mapper \
        --database_path "${DATASET_PATH}/database.db" \
        --image_path "${DATASET_PATH}/images" \
        --output_path "${DATASET_PATH}/sparse-final" \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.ba_refine_extra_params 0 \
        --Mapper.ba_refine_sensor_from_rig 0

    # Bundle Adjustment
    mkdir -p "${DATASET_PATH}/sparse-refined"
    ${COLMAP_EXE} bundle_adjuster \
        --input_path "${DATASET_PATH}/sparse-final/0" \
        --output_path "${DATASET_PATH}/sparse-refined"
}


# ==========================================
# EXECUTION
# Uncomment the pipeline you wish to run
# ==========================================

# run_unknown_rig_pipeline
# run_known_rig_pipeline

echo "Please uncomment a function at the bottom of the script to run a specific pipeline."