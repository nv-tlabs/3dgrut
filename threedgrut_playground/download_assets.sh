#!/bin/bash

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


assets=(
    armadillo.obj
    xyzrgb_dragon.obj
    beast.obj
    happy.obj
    horse.obj
    lucy.obj
    nefertiti.obj
    teapot.obj
)

# Source: https://github.com/alecjacobson/common-3d-test-models/blob/master/data
base_url="https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data"
output_dir="./threedgrut_playground/assets"

if command -v wget > /dev/null; then
    downloader="wget"
elif command -v curl > /dev/null; then
    downloader="curl"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them and re-run the script."
    exit 1
fi

download_file() {
    local file=$1
    local url="$base_url/$file"
    local output_path="$output_dir/$file"

    if [[ -f "$output_path" ]]; then
        echo "File '$file' already exists. Skipping download."
    else
        echo "Downloading '$file'..."

        if [[ "$downloader" == "wget" ]]; then
            wget -q --show-progress -O "$output_path" "$url"
        else
            curl -L -o "$output_path" "$url"
        fi
        if [[ $? -ne 0 ]]; then
            echo "❌ Failed to download '$file'."
            rm -f "$output_path"  # If an empty file was created, delete it
        fi
    fi
}

mkdir -p "$output_dir"
echo "Downloading 3D mesh assets from '$base_url'"

for asset in "${assets[@]}"; do
    download_file "$asset"
done

echo "Downloading sample envmaps from PolyHaven"
wget -q --show-progress -O "$output_dir/drackenstein_quarry_puresky_2k.hdr" "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/drackenstein_quarry_puresky_2k.hdr"
wget -q --show-progress -O "$output_dir/drackenstein_quarry_puresky_4k.hdr" "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/drackenstein_quarry_puresky_4k.hdr"

echo "✅ Download complete. Files saved in '$output_dir'"
