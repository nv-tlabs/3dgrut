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

import logging
import os
from enum import IntEnum
import torch
import torch.utils.cpp_extension

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
#

_playground_plugin = None


def load_playground_plugin(conf):
    global _playground_plugin
    if _playground_plugin is None:
        try:
            from . import libplayground_cc as tdgrt  # type: ignore
        except ImportError:
            from .setup_playground import setup_playground

            setup_playground(conf)
            import libplayground_cc as tdgrt  # type: ignore
        _playground_plugin = tdgrt


# ----------------------------------------------------------------------------
#

class Tracer:

    def __init__(self, conf):

        self.device = "cuda"
        self.conf = conf
        self.num_update_bvh = 0

        logger.info(f'🔆 Creating Optix tracing pipeline.. Using CUDA path: "{torch.utils.cpp_extension.CUDA_HOME}"')
        torch.zeros(1, device=self.device)  # Create a dummy tensor to force cuda context init
        load_playground_plugin(conf)

        playground_module_path = os.path.dirname(__file__)
        threedgrt_tracer_module_path = os.path.abspath(os.path.join(playground_module_path, '..', 'threedgrt_tracer'))

        self.tracer_wrapper = _playground_plugin.HybridOptixTracer(
            threedgrt_tracer_module_path,
            playground_module_path,
            torch.utils.cpp_extension.CUDA_HOME,
            self.conf.render.pipeline_type,
            self.conf.render.backward_pipeline_type,
            self.conf.render.primitive_type,
            self.conf.render.particle_kernel_degree,
            self.conf.render.particle_kernel_min_response,
            self.conf.render.particle_kernel_density_clamping,
            self.conf.render.particle_radiance_sph_degree,
            self.conf.render.enable_normals,
            self.conf.render.enable_hitcounts,
        )

    def build_gs_acc(self, gaussians, rebuild=True):
        with torch.cuda.nvtx.range(f"build-bvh-full-build-{rebuild}"):
            allow_bvh_update = (
                self.conf.render.max_consecutive_bvh_update > 1
            ) and not self.conf.render.particle_kernel_density_clamping
            rebuild_bvh = (
                rebuild
                or self.conf.render.particle_kernel_density_clamping
                or self.num_update_bvh >= self.conf.render.max_consecutive_bvh_update
            )
            self.tracer_wrapper.build_bvh(
                gaussians.positions.view(-1, 3).contiguous(),
                gaussians.rotation_activation(gaussians.rotation).view(-1, 4).contiguous(),
                gaussians.scale_activation(gaussians.scale).view(-1, 3).contiguous(),
                gaussians.density_activation(gaussians.density).view(-1, 1).contiguous(),
                rebuild_bvh,
                allow_bvh_update
            )
            self.num_update_bvh = 0 if rebuild_bvh else self.num_update_bvh + 1

    def build_mesh_acc(self, mesh_vertices, mesh_faces, rebuild=True, allow_update=True):
        with torch.cuda.nvtx.range(f"build-mesh-bvh-full-build-{rebuild}"):
            self.tracer_wrapper.build_mesh_bvh(
                mesh_vertices.view(-1, 3).contiguous(),
                mesh_faces.view(-1, 3).contiguous(),
                rebuild,
                allow_update
            )

    def render(
        self,
        gaussians,
        gpu_batch,
        train=False,
        frame_id=0
    ):
        assert ('rays_o_cam' in gpu_batch) and ('rays_d_cam' in gpu_batch) and ('poses' in gpu_batch)
        
        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            features = gaussians.get_features()
            if gaussians.progressive_training:
                features *= gaussians.get_active_feature_mask()

            mog_pos = gaussians.positions.contiguous()
            mog_dns = gaussians.get_density().contiguous()
            mog_rot = gaussians.get_rotation().contiguous()
            mog_scl = gaussians.get_scale().contiguous()
            particle_density = torch.concat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1)

            (
                pred_rgb,
                pred_opacity,
                pred_dist,
                pred_normals,
                hits_count
            ) = self.tracer_wrapper.trace(
                frame_id,
                gpu_batch['poses'].contiguous(),
                gpu_batch['rays_o_cam'].contiguous(),
                gpu_batch['rays_d_cam'].contiguous(),
                particle_density,
                features.contiguous(),
                gaussians.n_active_features,
                self.conf.render.min_transmittance,
            )

            # NOTE: disable background
            pred_rgb, pred_opacity = gaussians.background(
                gpu_batch['poses'].contiguous(),
                gpu_batch['rays_d_cam'].contiguous(),
                pred_rgb,
                pred_opacity,
                train
            )

        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count
        }

    @staticmethod
    def to_native_pbr_material(pbr_mat):
        make_contiguous_tex = lambda x, c: x.contiguous() if x is not None else torch.empty([0, 0, c],
                                                                                            dtype=torch.float32)
        make_contiguous_diffuse = lambda x: x.cpu().contiguous() if x is not None else torch.ones(4,
                                                                                                  dtype=torch.float32)
        make_contiguous_emissive = lambda x: x.cpu().contiguous() if x is not None else torch.zeros(4,
                                                                                                    dtype=torch.float32)
        cpbr_material = _playground_plugin.CPBRMaterial()
        cpbr_material.material_id = pbr_mat.material_id
        cpbr_material.diffuseMap = make_contiguous_tex(pbr_mat.diffuse_map, 4)
        cpbr_material.emissiveMap = make_contiguous_tex(pbr_mat.emissive_map, 4)
        cpbr_material.metallicRoughnessMap = make_contiguous_tex(pbr_mat.metallic_roughness_map, 2)
        cpbr_material.normalMap = make_contiguous_tex(pbr_mat.normal_map, 4)
        cpbr_material.diffuseFactor = make_contiguous_diffuse(pbr_mat.diffuse_factor)
        cpbr_material.emissiveFactor = make_contiguous_emissive(pbr_mat.emissive_factor)
        cpbr_material.metallicFactor = pbr_mat.metallic_factor or 0.0
        cpbr_material.roughnessFactor = pbr_mat.roughness_factor or 0.0
        cpbr_material.alphaMode = pbr_mat.alpha_mode or 0
        cpbr_material.alphaCutoff = pbr_mat.alpha_cutoff or 0.5
        cpbr_material.transmissionFactor = pbr_mat.transmission_factor or 0.0
        cpbr_material.ior = pbr_mat.ior or 0.0
        return cpbr_material

    def render_playground(
        self,
        gaussians,
        ray_o,      # world coords
        ray_d,      # world coords
        playground_opts,
        mesh_faces,
        vertex_normals,
        vertex_tangents,
        vertex_tangents_mask,
        primitive_type,
        frame_id=0,
        ray_max_t=None,
        material_uv=None,
        material_id=None,
        materials=None,
        refractive_index=None,
        background_color=None,
        envmap=None,
        enable_envmap=False,
        use_envmap_as_background=False,
        max_pbr_bounces=7
    ):
        if ray_max_t is None:
            ray_max_t = ray_o.new_full(size=ray_o.shape[0:3], fill_value=1e9)
        if refractive_index is None:
            refractive_index = torch.ones_like(primitive_type, dtype=torch.float)
        if material_uv is None:
            material_uv = torch.empty([0, 0], dtype=torch.float)
        if material_id is None:
            material_id = torch.empty([0, 0], dtype=torch.int)
        if materials is None:
            materials = []
        else:
            materials = [self.to_native_pbr_material(m) for m in materials]
        if background_color is None:
            background_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if envmap is None:
            envmap = torch.empty([0, 0, 0], dtype=torch.float32)

        poses = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=torch.float32)

        num_gaussians = gaussians.num_gaussians
        with torch.cuda.nvtx.range(f"model.forward({num_gaussians} gaussians)"):
            # The feature mask zeros out feature dims the model shouldn't use yet.
            # That introduces a curriculum way of optimizing the model
            features = gaussians.get_features()
            if gaussians.progressive_training:
                features *= gaussians.get_active_feature_mask()

            mog_pos = gaussians.positions.contiguous()
            mog_dns = gaussians.get_density().contiguous()
            mog_rot = gaussians.get_rotation().contiguous()
            mog_scl = gaussians.get_scale().contiguous()
            particle_density = torch.concat([mog_pos, mog_dns, mog_rot, mog_scl, torch.zeros_like(mog_dns)], dim=1)
            sph_degree = gaussians.n_active_features
            min_transmittance = self.conf.render.min_transmittance

            (
                pred_rgb,
                pred_opacity,
                pred_dist,
                pred_normals,
                hits_count
            ) = self.tracer_wrapper.trace_hybrid(
                frame_id,
                poses,
                ray_o,
                ray_d,
                particle_density,
                features,
                sph_degree,
                min_transmittance,
                ray_max_t,
                playground_opts,
                mesh_faces,
                vertex_normals,
                vertex_tangents,
                vertex_tangents_mask,
                primitive_type,
                material_uv,
                material_id,
                materials,
                refractive_index,
                background_color,
                envmap,
                enable_envmap,
                use_envmap_as_background,
                max_pbr_bounces
            )

            pred_dist = pred_dist[:, :, :, 0:1]  # return only the hit distance

        return {
            "pred_rgb": pred_rgb,
            "pred_opacity": pred_opacity,
            "pred_dist": pred_dist,
            "pred_normals": torch.nn.functional.normalize(pred_normals, dim=3),
            "hits_count": hits_count,
            "last_ray_o": ray_o,    # Rewritten by tracer
            "last_ray_d": ray_d     # Rewritten by tracer
        }

    @torch.cuda.nvtx.range("denoise")
    def denoise(self, ray_radiance):
        return self.tracer_wrapper.denoise(ray_radiance.contiguous())
