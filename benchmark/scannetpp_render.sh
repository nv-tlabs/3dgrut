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

RESULT_DIR=$1
if [[ -z $RESULT_DIR ]]; then
    echo "Error: Result directory is not provided. Aborting execution."
    echo "Usage: $0 <result-directory>"
    exit 1
fi

SCENE_LIST="0a5c013435 8d563fc2cc bb87c292ad d415cc449b e8ea9b4da8 fe1733741f"
for SCENE in $SCENE_LIST;
do
    echo "Running $SCENE"

    python render.py --checkpoint $(find $RESULT_DIR/$SCENE -name ckpt_last.pt) --out-dir $RESULT_DIR/$SCENE/eval > $RESULT_DIR/render_$SCENE.log

done

# To grep results from log files, run the following command:
# grep "Test Metrics"        -A 5 train_*.log | awk 'NR % 7 == 5'
