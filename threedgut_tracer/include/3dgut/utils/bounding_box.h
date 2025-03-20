// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   bounding_box.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 *  @brief  CUDA/C++ AABB implementation.
 */

#pragma once

#include <tiny-cuda-nn/common.h>

namespace threedgut {

struct BoundingBox {
	TCNN_HOST_DEVICE BoundingBox() {}

	TCNN_HOST_DEVICE BoundingBox(const tcnn::vec3& a, const tcnn::vec3& b) : min{a}, max{b} {}

	TCNN_HOST_DEVICE BoundingBox(tcnn::vec3* begin, tcnn::vec3* end, float radius) {
		min = max = *begin;
		for (auto it = begin; it != end; ++it) {
			enlarge(*it);
		}
		inflate(radius);
	}

	TCNN_HOST_DEVICE void enlarge(const BoundingBox& other) {
		min = tcnn::min(min, other.min);
		max = tcnn::max(max, other.max);
	}

	TCNN_HOST_DEVICE void enlarge(const tcnn::vec3& point) {
		min = tcnn::min(min, point);
		max = tcnn::max(max, point);
	}

	TCNN_HOST_DEVICE void inflate(float amount) {
		min -= tcnn::vec3(amount);
		max += tcnn::vec3(amount);
	}

	TCNN_HOST_DEVICE tcnn::vec3 diag() const {
		return max - min;
	}

	TCNN_HOST_DEVICE tcnn::vec3 relative_pos(const tcnn::vec3& pos) const {
		return (pos - min) / diag();
	}

	TCNN_HOST_DEVICE tcnn::vec3 center() const {
		return 0.5f * (max + min);
	}

	TCNN_HOST_DEVICE BoundingBox intersection(const BoundingBox& other) const {
		BoundingBox result = *this;
		result.min = tcnn::max(result.min, other.min);
		result.max = tcnn::min(result.max, other.max);
		return result;
	}

	TCNN_HOST_DEVICE bool intersects(const BoundingBox& other) const {
		return !intersection(other).is_empty();
	}

	TCNN_HOST_DEVICE tcnn::vec2 ray_intersect(const tcnn::vec3& pos, const tcnn::vec3& dir) const {
		float tmin = (min.x - pos.x) / dir.x;
		float tmax = (max.x - pos.x) / dir.x;

		if (tmin > tmax) {
			tcnn::host_device_swap(tmin, tmax);
		}

		float tymin = (min.y - pos.y) / dir.y;
		float tymax = (max.y - pos.y) / dir.y;

		if (tymin > tymax) {
			tcnn::host_device_swap(tymin, tymax);
		}

		if (tmin > tymax || tymin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tymin > tmin) {
			tmin = tymin;
		}

		if (tymax < tmax) {
			tmax = tymax;
		}

		float tzmin = (min.z - pos.z) / dir.z;
		float tzmax = (max.z - pos.z) / dir.z;

		if (tzmin > tzmax) {
			tcnn::host_device_swap(tzmin, tzmax);
		}

		if (tmin > tzmax || tzmin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tzmin > tmin) {
			tmin = tzmin;
		}

		if (tzmax < tmax) {
			tmax = tzmax;
		}

		return { tmin, tmax };
	}

	TCNN_HOST_DEVICE bool is_empty() const {
		return max.x < min.x || max.y < min.y || max.z < min.z;
	}

	TCNN_HOST_DEVICE bool contains(const tcnn::vec3& p) const {
		return
			p.x >= min.x && p.x <= max.x &&
			p.y >= min.y && p.y <= max.y &&
			p.z >= min.z && p.z <= max.z;
	}

	/// Calculate the squared point-AABB distance
	TCNN_HOST_DEVICE float distance(const tcnn::vec3& p) const {
		return tcnn::sqrt(distance_sq(p));
	}

	TCNN_HOST_DEVICE float distance_sq(const tcnn::vec3& p) const {
		return tcnn::length2(tcnn::max(tcnn::max(min - p, p - max), tcnn::vec3(0.0f)));
	}

	TCNN_HOST_DEVICE float signed_distance(const tcnn::vec3& p) const {
		tcnn::vec3 q = abs(p - min) - diag();
		return tcnn::length(tcnn::max(q, tcnn::vec3(0.0f))) + std::min(tcnn::max(q), 0.0f);
	}

	TCNN_HOST_DEVICE void get_vertices(tcnn::vec3 v[8]) const {
		v[0] = {min.x, min.y, min.z};
		v[1] = {min.x, min.y, max.z};
		v[2] = {min.x, max.y, min.z};
		v[3] = {min.x, max.y, max.z};
		v[4] = {max.x, min.y, min.z};
		v[5] = {max.x, min.y, max.z};
		v[6] = {max.x, max.y, min.z};
		v[7] = {max.x, max.y, max.z};
	}

	tcnn::vec3 min = tcnn::vec3(std::numeric_limits<float>::infinity());
	tcnn::vec3 max = tcnn::vec3(-std::numeric_limits<float>::infinity());
};

}
