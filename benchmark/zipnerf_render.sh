# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e


COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

RESULT_DIR=$1
if [[ -z $RESULT_DIR ]]; then
    echo "Error: Result directory is not provided. Aborting execution."
    echo "Usage: $0 <result-directory> <optional-path-to-data>"
    exit 1
fi

DATA_PATH=$2

export TORCH_EXTENSIONS_DIR=$RESULT_DIR/.cache  # make sure we have a clean build

SCENE_LIST="alameda berlin london nyc"

for SCENE in $SCENE_LIST;
do
    DATA_FACTOR=8

    if [[ -z $DATA_PATH ]]; then

        echo -e "${COLOR_GREEN}Running $SCENE${COLOR_RESET}"

        python render.py --checkpoint $(find $RESULT_DIR/fisheye_$SCENE -name ckpt_last.pt) \
            --out-dir $RESULT_DIR/fisheye_$SCENE/eval > $RESULT_DIR/render_fisheye_$SCENE.log

        python render.py --checkpoint $(find $RESULT_DIR/undistorted_$SCENE -name ckpt_last.pt) \
            --out-dir $RESULT_DIR/undistorted_$SCENE/eval > $RESULT_DIR/render_undistorted_$SCENE.log

    else

        # find the last directory name in path
        TEST=$(basename $DATA_PATH)

        # train on fisheye    
        echo -e "${COLOR_GREEN}Trained on fisheye/$SCENE, Testing on $TEST/$SCENE${COLOR_RESET}"
        python render.py --checkpoint $(find $RESULT_DIR/fisheye_$SCENE -name ckpt_last.pt) \
            --path $DATA_PATH/$SCENE \
            --out-dir $RESULT_DIR/fisheye_$SCENE/eval_${TEST} > $RESULT_DIR/render_${SCENE}_train_fisheye_test_${TEST}.log

        # train on undistorted
        echo -e "${COLOR_GREEN}Trained on undistorted/$SCENE, Testing on $TEST/$SCENE${COLOR_RESET}"
        python render.py --checkpoint $(find $RESULT_DIR/undistorted_$SCENE -name ckpt_last.pt) \
            --path $DATA_PATH/$SCENE \
            --out-dir $RESULT_DIR/undistorted_$SCENE/eval_${TEST} > $RESULT_DIR/render_${SCENE}_train_undistorted_test_${TEST}.log

    fi

done
