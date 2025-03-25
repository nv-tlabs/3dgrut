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

from __future__ import annotations
import copy
import numpy as np
import torch
import os
import polyscope as ps
import polyscope.imgui as psim
import traceback
import cv2
from typing import List, Tuple, Optional
from enum import IntEnum
from pathlib import Path
from tqdm import tqdm
from scipy.special import comb
from dataclasses import dataclass
from threedgrut.utils.logger import logger
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.background import BackgroundColor
from playground.utils.mesh_io import load_mesh, load_materials, load_missing_material_info, create_procedural_mesh
from playground.utils.depth_of_field import DepthOfField
from playground.utils.spp import SPP
from playground.utils.transform import ObjectTransform
from playground.tracer import Tracer


#################################
##       --- Common ---        ##
#################################

@dataclass
class RayPack:
    rays_ori: torch.FloatTensor
    rays_dir: torch.FloatTensor
    pixel_x: Optional[torch.IntTensor] = None
    pixel_y: Optional[torch.IntTensor] = None
    mask: Optional[torch.BoolTensor] = None

    def split(self, size=None) -> List[RayPack]:
        if size is None:
            return [self]
        assert self.rays_ori.ndim == 2 and self.rays_dir.ndim == 2, 'Only 1D ray packs can be split'
        rays_orig = torch.split(self.rays_ori, size, dim=0)
        rays_dir = torch.split(self.rays_dir, size, dim=0)
        return [RayPack(ray_ori, ray_dir) for ray_ori, ray_dir in zip(rays_orig, rays_dir)]


@dataclass
class PBRMaterial:
    material_id: int
    diffuse_map: Optional[torch.Tensor] = None  # (H, W, 4)
    emissive_map: Optional[torch.Tensor] = None  # (H, W, 4)
    metallic_roughness_map: Optional[torch.Tensor] = None  # (H, W, 2)
    normal_map: Optional[torch.Tensor] = None  # (H, W, 4)
    diffuse_factor: torch.Tensor = None  # (4,)
    emissive_factor: torch.Tensor = None  # (3,)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.0
    alpha_mode: int = 0
    alpha_cutoff: float = 0.5
    transmission_factor: float = 0.0
    ior: float = 1.0


#################################
##     --- OPTIX STRUCTS ---   ##
#################################


class OptixPlaygroundRenderOptions(IntEnum):
    NONE = 0
    SMOOTH_NORMALS = 1
    DISABLE_GAUSSIAN_TRACING = 2
    DISABLE_PBR_TEXTURES = 4


class OptixPrimitiveTypes(IntEnum):
    NONE = 0
    MIRROR = 1
    GLASS = 2
    DIFFUSE = 3

    @classmethod
    def names(cls):
        return ['None', 'Mirror', 'Glass', 'Diffuse Mesh']


@dataclass
class OptixPrimitive:
    geometry_type: str = None
    vertices: torch.Tensor = None
    triangles: torch.Tensor = None
    vertex_normals: torch.Tensor = None
    has_tangents: torch.Tensor = None
    vertex_tangents: torch.Tensor = None
    material_uv: Optional[torch.Tensor] = None
    material_id: Optional[torch.Tensor] = None
    primitive_type: Optional[OptixPrimitiveTypes] = None
    primitive_type_tensor: torch.Tensor = None

    # Mirrors
    reflectance_scatter: torch.Tensor = None
    # Glass
    refractive_index: Optional[float] = None
    refractive_index_tensor: torch.Tensor = None

    transform: ObjectTransform() = None

    @classmethod
    def stack(cls, primitives):
        device = primitives[0].vertices.device
        vertices = torch.cat([p.vertices for p in primitives], dim=0)
        v_offset = torch.tensor([0] + [p.vertices.shape[0] for p in primitives[:-1]], device=device)
        v_offset = torch.cumsum(v_offset, dim=0)
        triangles = torch.cat([p.triangles + offset for p, offset in zip(primitives, v_offset)], dim=0)

        return OptixPrimitive(
            vertices=vertices.float(),
            triangles=triangles.int(),
            vertex_normals=torch.cat([p.vertex_normals for p in primitives], dim=0).float(),
            has_tangents=torch.cat([p.has_tangents for p in primitives], dim=0).bool(),
            vertex_tangents=torch.cat([p.vertex_tangents for p in primitives], dim=0).float(),
            material_uv=torch.cat([p.material_uv for p in primitives if p.material_uv is not None], dim=0).float(),
            material_id=torch.cat([p.material_id for p in primitives if p.material_id is not None], dim=0).int(),
            primitive_type_tensor=torch.cat([p.primitive_type_tensor for p in primitives], dim=0).int(),
            reflectance_scatter=torch.cat([p.reflectance_scatter for p in primitives], dim=0).float(),
            refractive_index_tensor=torch.cat([p.refractive_index_tensor for p in primitives], dim=0).float()
        )

    def apply_transform(self):
        model_matrix = self.transform.model_matrix()
        rs_comp = model_matrix[None, :3, :3]
        t_comp = model_matrix[None, :3, 3:]
        transformed_verts = (rs_comp @ self.vertices[:, :, None] + t_comp).squeeze(2)

        normal_matrix = self.transform.rotation_matrix()[None, :3, :3]
        transformed_normals = (normal_matrix @ self.vertex_normals[:, :, None]).squeeze(2)
        transformed_normals = torch.nn.functional.normalize(transformed_normals)

        transformed_tangents = (normal_matrix @ self.vertex_tangents[:, :, None]).squeeze(2)
        transformed_tangents = torch.nn.functional.normalize(transformed_tangents)

        return OptixPrimitive(
            vertices=transformed_verts,
            triangles=self.triangles,
            vertex_normals=transformed_normals,
            vertex_tangents=transformed_tangents,
            has_tangents=self.has_tangents,
            material_uv=self.material_uv,
            material_id=self.material_id,
            primitive_type=self.primitive_type,
            primitive_type_tensor=self.primitive_type_tensor,
            reflectance_scatter=self.reflectance_scatter,
            refractive_index=self.refractive_index,
            refractive_index_tensor=self.refractive_index_tensor,
            transform=ObjectTransform(device=self.transform.device)
        )


class Primitives:
    SUPPORTED_MESH_EXTENSIONS = ['.obj', '.glb']
    MAX_MATERIALS = 32
    DEFAULT_REFRACTIVE_INDEX = 1.33
    SCALE_OF_NEW_MESH_TO_SMALL_SCENE = 0.5    # Mesh will be this percent of the scene on longest axis

    def __init__(self, tracer, mesh_assets_folder, enable_envmap=False, use_envmap_as_background=False,
                 scene_scale=None):
        # str -> str ; shape name to filename + extension
        self.assets = self.register_available_assets(assets_folder=mesh_assets_folder)
        self.tracer = tracer
        self.enabled = True
        self.objects = dict()
        self.use_smooth_normals = True
        self.disable_gaussian_tracing = False
        self.disable_pbr_textures = False
        self.force_white_bg = False
        self.enable_envmap = enable_envmap
        self.use_envmap_as_background = use_envmap_as_background

        if scene_scale is None:
            self.scene_scale = torch.tensor([1.0, 1.0, 1.0], device='cpu')
        else:
            self.scene_scale = scene_scale.cpu()

        self.stacked_fields = None
        self.dirty = True

        self.instance_counter = dict()  # Counts number of primitives of each geometry

        device = 'cuda'

        self.registered_materials = self.register_default_materials(device)  # str -> PBRMaterial

    def register_available_assets(self, assets_folder):
        available_assets = {Path(asset).stem.capitalize(): os.path.join(assets_folder, asset)
                            for asset in os.listdir(assets_folder)
                            if Path(asset).suffix in Primitives.SUPPORTED_MESH_EXTENSIONS}
        # Procedural shapes are added manually
        available_assets['Quad'] = None
        return available_assets # i.e. {MeshName: /path/to/mesh.glb}

    def register_default_materials(self, device):
        checkboard_res = 512
        checkboard_square = 20
        checkboard_texture = torch.tensor([0.25, 0.25, 0.25, 1.0],
                                          device=device, dtype=torch.float32).repeat(checkboard_res, checkboard_res, 1)
        for i in range(checkboard_res // checkboard_square):
            for j in range(checkboard_res // checkboard_square):
                start_x = (2 * i + j % 2) * checkboard_square
                end_x = min((2 * i + 1 + j % 2) * checkboard_square, checkboard_res)
                start_y = j * checkboard_square
                end_y = min((j + 1) * checkboard_square, checkboard_res)
                checkboard_texture[start_y:end_y, start_x:end_x, :3] = 0.5
        default_materials = dict(
            solid=PBRMaterial(
                material_id=0,
                diffuse_map=torch.tensor([130 / 255.0, 193 / 255.0, 255 / 255.0, 1.0],
                                         device=device, dtype=torch.float32).expand(2, 2, 4),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0
            ),
            checkboard=PBRMaterial(
                material_id=1,
                diffuse_map=checkboard_texture.contiguous(),
                diffuse_factor=torch.ones(4, device=device, dtype=torch.float32),
                emissive_factor=torch.zeros(3, device=device, dtype=torch.float32),
                metallic_factor=0.0,
                roughness_factor=0.0,
                transmission_factor=0.0,
                ior=1.0
            )
        )
        return default_materials

    def set_mesh_scale_to_scene(self, mesh, transform):
        """
        Uses heuristics to scale the mesh so it appears nice:
        1) mesh rescaled to unit size
        2) if the scene is small, the mesh is rescaled to SCALE_OF_NEW_MESH_TO_SMALL_SCENE of the scene scale
        """
        mesh_scale = ((mesh.vertices.max(dim=0)[0] - mesh.vertices.min(dim=0)[0]).cpu()).to(transform.device)
        transform.scale(1.0 / mesh_scale.max())
        if self.scene_scale.max() > 5.0:    # Don't scale for large scenes
            return
        adjusted_scale = self.SCALE_OF_NEW_MESH_TO_SMALL_SCENE * self.scene_scale.to(transform.device)
        largest_axis_scale = adjusted_scale.max()
        transform.scale(largest_axis_scale)

    def add_primitive(self, geometry_type: str, primitive_type: OptixPrimitiveTypes, device):
        if geometry_type not in self.instance_counter:
            self.instance_counter[geometry_type] = 1
        else:
            self.instance_counter[geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"

        mesh = self.create_geometry(geometry_type, device)

        # Generate tangents mas, if available
        num_verts = len(mesh.vertices)
        num_faces = len(mesh.faces)
        has_tangents = torch.ones([num_verts, 1], device=device, dtype=torch.bool) \
            if mesh.vertex_tangents is not None \
            else torch.zeros([num_verts, 1], device=device, dtype=torch.bool)
        # Create identity transform and set scale to scene size
        transform = ObjectTransform(device=device)
        self.set_mesh_scale_to_scene(mesh, transform)
        # Face attributes
        prim_type_tensor = mesh.faces.new_full(size=(num_faces,), fill_value=primitive_type.value)
        reflectance_scatter = mesh.faces.new_zeros(size=(num_faces,))
        refractive_index = Primitives.DEFAULT_REFRACTIVE_INDEX
        refractive_index_tensor = mesh.faces.new_full(size=(num_faces,), fill_value=refractive_index)

        self.objects[name] = OptixPrimitive(
            geometry_type=geometry_type,
            vertices=mesh.vertices.float(),
            triangles=mesh.faces.int(),
            vertex_normals=mesh.vertex_normals.float(),
            has_tangents=has_tangents.bool(),
            vertex_tangents=mesh.vertex_tangents.float(),
            material_uv=mesh.uvs.float(),
            material_id=mesh.material_assignments.unsqueeze(1).int(),
            primitive_type=primitive_type,
            primitive_type_tensor=prim_type_tensor.int(),
            reflectance_scatter=reflectance_scatter.float(),
            refractive_index=refractive_index,
            refractive_index_tensor=refractive_index_tensor.float(),
            transform=transform
        )

    def remove_primitive(self, name: str):
        del self.objects[name]
        self.rebuild_bvh_if_needed(True, True)

    def duplicate_primitive(self, name: str):
        prim = self.objects[name]
        geometry_type = prim.geometry_type
        self.instance_counter[prim.geometry_type] += 1
        name = f"{geometry_type} {self.instance_counter[geometry_type]}"
        self.objects[name] = copy.deepcopy(prim)
        self.rebuild_bvh_if_needed(True, True)

    def register_materials(self, materials, model_name: str):
        """ Registers list of material dictionaries.
        """
        mat_idx_to_mat_id = torch.full([len(materials)], -1)
        for mat_idx, mat in enumerate(materials):
            material_name = f'{model_name}${mat["material_name"]}'
            if material_name not in self.registered_materials:
                if len(self.registered_materials) >= self.MAX_MATERIALS:
                    print('WARNING: Maximum number of supported materials reached. Mesh will not render correctly.')
                    break
                else:
                    self.registered_materials[material_name] = PBRMaterial(
                        material_id=len(self.registered_materials),
                        diffuse_map=mat['diffuse_map'],
                        emissive_map=mat['emissive_map'],
                        metallic_roughness_map=mat['metallic_roughness_map'],
                        normal_map=mat['normal_map'],
                        diffuse_factor=mat['diffuse_factor'],
                        emissive_factor=mat['emissive_factor'],
                        metallic_factor=mat['metallic_factor'],
                        roughness_factor=mat['roughness_factor'],
                        alpha_mode=mat['alpha_mode'],
                        alpha_cutoff=mat['alpha_cutoff'],
                        transmission_factor=mat['transmission_factor'],
                        ior=mat['ior']
                    )
            mat_idx_to_mat_id[mat_idx] = self.registered_materials[material_name].material_id
        return mat_idx_to_mat_id

    def create_geometry(self, geometry_type: str, device):
        match geometry_type:
            case 'Quad':
                MS = 1.0
                MZ = 2.5
                v0 = [-MS, -MS, MZ]
                v1 = [-MS, +MS, MZ]
                v2 = [+MS, -MS, MZ]
                v3 = [+MS, +MS, MZ]
                mesh = create_procedural_mesh(
                    vertices=torch.tensor([v0, v1, v2, v3]),
                    faces=torch.tensor([[0, 1, 2], [2, 1, 3]]),
                    uvs=torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
                    device=device
                )
            case _:
                mesh_path = self.assets[geometry_type]
                mesh = load_mesh(mesh_path, device)
                materials = load_materials(mesh, device)
                if len(materials) > 0:
                    load_missing_material_info(mesh_path, materials, device)
                    material_index_mapping = self.register_materials(materials=materials, model_name=geometry_type)
                    # Update material assignments to match playground material registry
                    material_index_mapping = material_index_mapping.to(device=device)
                    material_id = mesh.material_assignments.to(device=device, dtype=torch.long)
                    mesh.material_assignments = material_index_mapping[material_id].int()
        return mesh

    def recompute_stacked_buffers(self):
        objects = [p.apply_transform() for p in self.objects.values()]
        # Recompute primitive type tensor
        for obj in objects:
            f = obj.triangles
            num_faces = f.shape[0]
            obj.primitive_type_tensor = f.new_full(size=(num_faces,), fill_value=obj.primitive_type.value)
            obj.refractive_index_tensor = f.new_full(size=(num_faces,),
                                                     fill_value=obj.refractive_index, dtype=torch.float)

        # Stack fields again
        self.stacked_fields = None
        if self.has_visible_objects():
            self.stacked_fields = OptixPrimitive.stack([p for p in objects if p.primitive_type != OptixPrimitiveTypes.NONE])

    def has_visible_objects(self):
        return len([p for p in self.objects.values() if p.primitive_type != OptixPrimitiveTypes.NONE]) > 0

    @torch.cuda.nvtx.range("rebuild_bvh (prim)")
    def rebuild_bvh_if_needed(self, force=False, rebuild=True):
        if self.dirty or force:
            if self.has_visible_objects():
                self.recompute_stacked_buffers()
                self.tracer.build_mesh_acc(
                    mesh_vertices=self.stacked_fields.vertices,
                    mesh_faces=self.stacked_fields.triangles,
                    rebuild=rebuild,
                    allow_update=True
                )
            else:
                self.tracer.build_mesh_acc(
                    mesh_vertices=torch.zeros([3, 3], dtype=torch.float, device='cuda'),
                    mesh_faces=torch.zeros([1, 3], dtype=torch.int, device='cuda'),
                    rebuild=True,
                    allow_update=True
                )
        self.dirty = False

#################################
##       --- Renderer      --- ##
#################################

class Playground:
    DEFAULT_DEVICE = torch.device('cuda')
    AVAILABLE_CAMERAS = ['Pinhole', 'Fisheye']
    ANTIALIASING_MODES = ['4x MSAA', '8x MSAA', '16x MSAA', 'Quasi-Random (Sobol)']
    trajectory = []
    continuous_trajectory = False
    trajectory_fps = 30
    frames_between_cameras = 60
    trajectory_output_path = "output.mp4"
    cameras_save_path = "cameras.npy"
    window_w, window_h = 1920, 1080
    use_dof_in_trajectory = False
    min_dof = 2.5
    max_dof = 24

    def __init__(self, gs_object, mesh_assets_folder, default_config, buffer_mode="device2device"):

        self.scene_mog, self.scene_name = self.load_object(gs_object, config_name=default_config)
        self.tracer = Tracer(self.scene_mog.conf)
        device = self.scene_mog.device

        self.envmap = None  # Currently disabled

        self.frame_id = 0
        self.camera_type = 'Pinhole'
        self.camera_fov = 120.0

        self.use_depth_of_field = False
        self.depth_of_field = DepthOfField(aperture_size=0.01, focus_z=1.0)

        """ Outwards facing, these are the useful modes to configure """
        self.use_spp = True
        self.antialiasing_mode = '4x MSAA'
        self.spp = SPP(mode='msaa', spp=4, device=device)

        """ Gamma correction factor """
        self.gamma_correction = 1.0

        """ Maximum number of PBR material bounces (transmissions & refractions, reflections) """
        self.max_pbr_bounces = 15

        """ If enabled, will use the optix denoiser as post-processing """
        self.use_optix_denoiser = True

        scene_scale = self.scene_mog.positions.max(dim=0)[0] - self.scene_mog.positions.min(dim=0)[0]
        self.primitives = Primitives(
            tracer=self.tracer,
            mesh_assets_folder=mesh_assets_folder,
            enable_envmap=self.envmap is not None,
            use_envmap_as_background=self.envmap is not None,
            scene_scale=scene_scale
        )
        self.primitives.add_primitive(geometry_type='Sphere', primitive_type=OptixPrimitiveTypes.GLASS, device=device)
        self.rebuild_bvh(self.scene_mog)
        if self.envmap is not None:
            self.primitives.force_white_bg = False

        self.last_state = dict(
            camera=None,
            rgb=None,
            opacity=None
        )
        self.is_force_canvas_dirty = False
        self.gui_aux_fields = dict()

        self.is_running = True
        self.density_buffer = copy.deepcopy(self.scene_mog.density)  # For slicing planes
        self.ps_buffer_mode = buffer_mode
        self.init_polyscope(buffer_mode)

    def _accumulate_to_buffer(self, prev_frames, new_frame, num_frames_accumulated, gamma, batch_size=1):
        prev_frames = torch.pow(prev_frames, gamma)
        buffer = ((prev_frames * num_frames_accumulated) + new_frame) / (num_frames_accumulated + batch_size)
        buffer = torch.pow(buffer, 1.0 / gamma)
        return buffer

    @torch.cuda.nvtx.range("_render_depth_of_field_buffer")
    def _render_depth_of_field_buffer(self, rb, view_params, rays):
        if self.use_depth_of_field and self.depth_of_field.has_more_to_accumulate():
            # Store current spp index
            i = self.depth_of_field.spp_accumulated_for_frame
            extrinsics_R = torch.from_numpy(view_params.get_R()).to(device=self.DEFAULT_DEVICE)

            dof_rays_ori, dof_rays_dir = self.depth_of_field(extrinsics_R, rays)
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                dof_rb = self.scene_mog.trace(rays_o=dof_rays_ori, rays_d=dof_rays_dir)
            else:
                dof_rb = self._render_playground_hybrid(dof_rays_ori, dof_rays_dir)

            rb['rgb'] = self._accumulate_to_buffer(rb['rgb'], dof_rb['pred_rgb'], i, self.gamma_correction)
            rb['opacity'] = (rb['opacity'] * i + dof_rb['pred_opacity']) / (i + 1)

    def _render_spp_buffer(self, rb, rays):
        if self.use_spp and self.spp.has_more_to_accumulate():
            # Store current spp index
            i = self.spp.spp_accumulated_for_frame

            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                spp_rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                spp_rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)
            batch_rgb = spp_rb['pred_rgb'].sum(dim=0).unsqueeze(0)
            rb['rgb'] = self._accumulate_to_buffer(rb['rgb'], batch_rgb, i, self.gamma_correction,
                                                   batch_size=self.spp.batch_size)
            rb['opacity'] = (rb['opacity'] * i + spp_rb['pred_opacity']) / (i + self.spp.batch_size)

    @torch.cuda.nvtx.range(f"playground._render_playground_hybrid")
    def _render_playground_hybrid(self, rays_o: torch.Tensor, rays_d) -> dict[str, torch.Tensor]:
        mog = self.scene_mog
        playground_render_opts = 0
        if self.primitives.use_smooth_normals:
            playground_render_opts |= OptixPlaygroundRenderOptions.SMOOTH_NORMALS
        if self.primitives.disable_gaussian_tracing:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_GAUSSIAN_TRACING
        if self.primitives.disable_pbr_textures:
            playground_render_opts |= OptixPlaygroundRenderOptions.DISABLE_PBR_TEXTURES

        self.primitives.rebuild_bvh_if_needed()

        envmap = self.envmap
        if self.primitives.force_white_bg:
            background_color = torch.ones(3)
            envmap = None
        elif isinstance(mog.background, BackgroundColor):
            background_color = mog.background.color
        else:
            background_color = torch.zeros(3)

        rendered_results = self.tracer.render_playground(
            gaussians=mog,
            ray_o=rays_o,
            ray_d=rays_d,
            playground_opts=playground_render_opts,
            mesh_faces=self.primitives.stacked_fields.triangles,
            vertex_normals=self.primitives.stacked_fields.vertex_normals,
            vertex_tangents=self.primitives.stacked_fields.vertex_tangents,
            vertex_tangents_mask=self.primitives.stacked_fields.has_tangents,
            primitive_type=self.primitives.stacked_fields.primitive_type_tensor[:, None],
            frame_id=self.frame_id,
            ray_max_t=None,
            material_uv=self.primitives.stacked_fields.material_uv,
            material_id=self.primitives.stacked_fields.material_id,
            materials=sorted(self.primitives.registered_materials.values(), key=lambda mat: mat.material_id),
            refractive_index=self.primitives.stacked_fields.refractive_index_tensor[:, None],
            background_color=background_color,
            envmap=envmap,
            enable_envmap=self.primitives.enable_envmap,
            use_envmap_as_background=self.primitives.use_envmap_as_background,
            max_pbr_bounces=self.max_pbr_bounces
        )

        pred_rgb = rendered_results['pred_rgb']
        pred_opacity = rendered_results['pred_opacity']

        if envmap is None or not self.primitives.use_envmap_as_background:
            if self.primitives.force_white_bg:
                pred_rgb += (1.0 - pred_opacity)
            else:
                poses = torch.tensor([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                ], dtype=torch.float32)
                pred_rgb, pred_opacity = mog.background(
                    poses.contiguous(),
                    rendered_results['last_ray_d'].contiguous(),
                    pred_rgb,
                    pred_opacity,
                    False
                )

        # Advance frame id (for i.e., random number generator) and avoid int32 overflow
        self.frame_id = self.frame_id + self.spp.batch_size if self.frame_id <= (2 ** 31 - 1) else 0

        pred_rgb = torch.clamp(pred_rgb, 0.0, 1.0)  # Make sure image pixels are in valid range

        rendered_results['pred_rgb'] = pred_rgb
        return rendered_results

    @torch.cuda.nvtx.range("render")
    @torch.no_grad()
    def render(self, view_params, window_w, window_h):

        is_canvas_dirty = self.is_dirty()
        is_use_spp = not is_canvas_dirty and not self.use_depth_of_field and self.use_spp
        rays = self.raygen(view_params, window_w, window_h, use_spp=is_use_spp)

        if is_canvas_dirty:
            if not self.primitives.enabled or not self.primitives.has_visible_objects():
                rb = self.scene_mog.trace(rays_o=rays.rays_ori, rays_d=rays.rays_dir)
            else:
                rb = self._render_playground_hybrid(rays.rays_ori, rays.rays_dir)

            rb = dict(rgb=rb['pred_rgb'], opacity=rb['pred_opacity'])
            rb['rgb'] = torch.pow(rb['rgb'], 1.0 / self.gamma_correction)
            rb['rgb'] = rb['rgb'].mean(dim=0).unsqueeze(0)
            rb['opacity'] = rb['opacity'].mean(dim=0).unsqueeze(0)
            self.spp.reset_accumulation()
            self.depth_of_field.reset_accumulation()
        else:
            # Render accumulated effects, i.e. depth of field
            rb = dict(rgb=self.last_state['rgb_buffer'], opacity=self.last_state['opacity'])
            if self.use_depth_of_field:
                self._render_depth_of_field_buffer(rb, view_params, rays)
            elif self.use_spp:
                self._render_spp_buffer(rb, rays)

        # Keep a noisy version of the accumulated rgb buffer so we don't repeat denoising per frame
        rb['rgb_buffer'] = rb['rgb']
        if self.use_optix_denoiser:
            rb['rgb'] = self.tracer.denoise(rb['rgb'])

        if rays.mask is not None:  # mask is for masking away pixels out of view for, i.e. fisheye
            mask = rays.mask[None, :, :, 0]
            rb['rgb'][mask] = 0.0
            rb['rgb_buffer'][mask] = 0.0
            rb['opacity'][mask] = 0.0

        return rb

    @torch.cuda.nvtx.range("load_object")
    def load_object(self, object_path, config_name='apps/colmap_3dgrt.yaml'):
        """
        Loads object from object_path.
        If object is in .ingp, .ply format, the model will be initialized with a config loaded from config_name.
        """
        def load_default_config():
            from hydra.compose import compose
            from hydra.initialize import initialize
            with initialize(version_base=None, config_path='../configs'):
                conf = compose(config_name=config_name)
            return conf

        if object_path.endswith('.pt'):
            checkpoint = torch.load(object_path)
            conf = checkpoint["config"]
            if conf.render['method'] != '3dgrt':
                conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_checkpoint(checkpoint, setup_optimizer=False)
            object_name = conf.experiment_name
        elif object_path.endswith('.ingp'):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ingp(object_path, init_model=False)
            object_name = Path(object_path).stem
        elif object_path.endswith('.ply'):
            conf = load_default_config()
            model = MixtureOfGaussians(conf)
            model.init_from_ply(object_path, init_model=False)
            object_name = Path(object_path).stem
        else:
            raise ValueError(f"Unknown object type: {object_path}")

        if object_name is None or len(object_name) == 0:
            object_name = Path(object_path).stem    # Fallback to pick object name from path, if none specified

        model.build_acc(rebuild=True)

        return model, object_name

    @torch.cuda.nvtx.range("rebuild_bvh (mog)")
    def rebuild_bvh(self, scene_mog):
        rebuild = True
        self.tracer.build_gs_acc(gaussians=scene_mog, rebuild=rebuild)
        self.primitives.rebuild_bvh_if_needed()

    def run(self):
        while self.is_running:
            ps.frame_tick()

            if ps.window_requests_close():
                self.is_running = False
                os._exit(0)

    @torch.cuda.nvtx.range("init_polyscope")
    def init_polyscope(self, buffer_mode):
        def set_polyscope_buffering_mode():
            if buffer_mode == "host2device":
                logger.info("polyscope set to host2device mode.")
            else:  # device2device
                from threedgrt_tracer.gui.ps_extension import initialize_cugl_interop
                initialize_cugl_interop()
                logger.info("polyscope set to device2device mode.")
        set_polyscope_buffering_mode()

        ps.set_use_prefs_file(False)

        ps.set_up_dir("neg_y_up")
        ps.set_front_dir("neg_z_front")
        ps.set_navigation_style("free")

        ps.set_enable_vsync(False)
        ps.set_max_fps(-1)
        ps.set_background_color((0., 0., 0.))
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(1920, 1080)
        ps.set_give_focus_on_show(True)

        ps.set_automatically_compute_scene_extents(False)
        ps.set_bounding_box(np.array([-1.5, -1.5, -1.5]), np.array([1.5, 1.5, 1.5]))

        # Toggle off default polyscope menus
        ps.set_build_default_gui_panels(False)

        self.viz_do_train = False
        self.viz_bbox = False
        self.live_update = True  # if disabled , will skip rendering updates to accelerate background training loop
        self.viz_render_styles = ['color', 'density']
        self.viz_render_style_ind = 0
        self.viz_curr_render_size = None
        self.viz_curr_render_style_ind = None
        self.viz_render_color_buffer = None
        self.viz_render_scalar_buffer = None
        self.viz_render_name = 'render'
        self.viz_render_enabled = True

        ps.init()
        ps.set_user_callback(self.ps_ui_callback)

        self.slice_planes = [ps.add_scene_slice_plane() for _ in range(6)]
        self.slice_plane_enabled = [False for _ in range(6)]
        self.slice_plane_pos = [
            np.array([-5.0, 0.0, 0.0]),
            np.array([0.0, -5.0, 0.0]),
            np.array([0.0, 0.0, -5.0]),
            np.array([5.0, 0.0, 0.0]),
            np.array([0.0, 5.0, 0.0]),
            np.array([0.0, 0.0, 5.0]),
        ]
        self.slice_plane_normal = [
            180.0 * np.array([1.0, 0.0, 0.0]),
            180.0 * np.array([0.0, 1.0, 0.0]),
            180.0 * np.array([0.0, 0.0, 1.0]),
            180.0 * np.array([-1.0, 0.0, 0.0]),
            180.0 * np.array([0.0, -1.0, 0.0]),
            180.0 * np.array([0.0, 0.0, -1.0]),
        ]

        # Update once to popualte lazily-created structures
        self.update_render_view_viz(force=True)

    @torch.no_grad()
    def render_trajectory(self):
        def smoothstep(x, x_min=0, x_max=1, N=1):
            x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

            result = 0
            for n in range(0, N + 1):
                result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

            result *= x ** (N + 1)

            return result


        if len(self.trajectory) < 2:
            return

        out_video = None

        if self.use_dof_in_trajectory:
            old_use_dof = self.use_depth_of_field
            self.use_depth_of_field = True
            eye, target, up = self.trajectory[0]
            ps.look_at_dir(eye, target, up, fly_to=True)
            dofs = np.linspace(self.min_dof, self.max_dof, self.frames_between_cameras)
            for dof in tqdm(dofs):
                self.depth_of_field.focus_z = dof
                self.is_force_canvas_dirty = True

                rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)
                while self.has_progressive_effects_to_render():
                    rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)

                if out_video is None:
                    out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.trajectory_fps, (rgb.shape[2], rgb.shape[1]), True)
                data = rgb[0].clip(0, 1).detach().cpu().numpy()
                data = (data * 255).astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                out_video.write(data)
            self.use_depth_of_field = old_use_dof

        elif self.continuous_trajectory:
            eyes = np.stack([eye for eye, target, up in self.trajectory])
            targets = np.stack([target for eye, target, up in self.trajectory])
            ups = np.stack([up for eye, target, up in self.trajectory])

            from scipy.interpolate import splprep, splev
            tck, u = splprep(eyes.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), self.frames_between_cameras * len(self.trajectory))
            eyes_new = np.stack(splev(u_new, tck, der=0)).T

            tck, u = splprep(targets.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), self.frames_between_cameras * len(self.trajectory))
            targets_new = np.stack(splev(u_new, tck, der=0)).T

            tck, u = splprep(ups.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), self.frames_between_cameras * len(self.trajectory))
            ups_new = np.stack(splev(u_new, tck, der=0)).T

            for eye, target, up in tqdm(zip(eyes_new, targets_new, ups_new)):
                ps.look_at_dir(eye, target, up)

                rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)
                while self.has_progressive_effects_to_render():
                    rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)

                if out_video is None:
                    out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.trajectory_fps, (rgb.shape[2], rgb.shape[1]), True)
                data = rgb[0].clip(0, 1).detach().cpu().numpy()
                data = (data * 255).astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                out_video.write(data)
        else:
            for traj_idx in tqdm(range(1, len(self.trajectory))):
                eye_1, target_1, up_1 = self.trajectory[traj_idx - 1]
                eye_2, target_2, up_2 = self.trajectory[traj_idx]

                Xs = smoothstep(np.linspace(0.0, 1.0, self.frames_between_cameras), N=3)

                for x in Xs:
                    eye = eye_1 * (1 - x) + eye_2 * x
                    target = target_1 * (1 - x) + target_2 * x
                    up = up_1 * (1 - x) + up_2 * x
                    ps.look_at_dir(eye, target, up)

                    rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)
                    while self.has_progressive_effects_to_render():
                        rgb, _ = self.render_from_current_ps_view(window_w=self.window_w, window_h=self.window_h)

                    if out_video is None:
                        out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                    self.trajectory_fps, (rgb.shape[2], rgb.shape[1]), True)
                    data = rgb[0].clip(0, 1).detach().cpu().numpy()
                    data = (data * 255).astype(np.uint8)
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    out_video.write(data)
        out_video.release()

    def did_camera_change(self):
        current_view_matrix = ps.get_view_camera_parameters().get_E()
        cached_camera = self.last_state.get('camera')
        is_camera_changed = cached_camera is not None and (cached_camera != current_view_matrix).any()
        return is_camera_changed

    def has_cached_buffers(self):
        return self.last_state.get('rgb') is not None and self.last_state.get('opacity') is not None

    def has_progressive_effects_to_render(self):
        has_dof_buffers_to_render = self.use_depth_of_field and \
                                    self.depth_of_field.spp_accumulated_for_frame <= self.depth_of_field.spp
        has_spp_buffers_to_render = not self.use_depth_of_field and \
                                    self.use_spp and self.spp.spp_accumulated_for_frame <= self.spp.spp
        return has_dof_buffers_to_render or has_spp_buffers_to_render

    def is_dirty(self):
        # Force dirty flag is on
        if self.is_force_canvas_dirty:
            return True
        if self.did_camera_change():
            return True
        if not self.has_cached_buffers():
            return True
        return False

    def cache_last_state(self, view_params, outputs, window_size):
        self.last_state['window_size'] = window_size
        self.last_state['camera'] = copy.deepcopy(view_params.get_E())
        self.last_state['rgb'] = outputs['rgb']
        self.last_state['rgb_buffer'] = outputs['rgb_buffer']
        self.last_state['opacity'] = outputs['opacity']

    def _raygen_pinhole(self, view_params, window_w, window_h, jitter=None) -> RayPack:
        cam_center = view_params.get_position()
        corner_rays = view_params.generate_camera_ray_corners()
        c_ul, c_ur, c_ll, c_lr = [torch.tensor(a, device=self.DEFAULT_DEVICE, dtype=torch.float32) for a in corner_rays]

        # generate view camera ray origins and directions
        rays_ori = torch.tensor(
            cam_center, device=self.DEFAULT_DEVICE,
            dtype=torch.float32).reshape(1, 1, 1, 3).expand(1, window_h, window_w, 3
                                                            )
        interp_x, interp_y = torch.meshgrid(
            torch.linspace(0., 1., window_w, device=self.DEFAULT_DEVICE, dtype=torch.float32),
            torch.linspace(0., 1., window_h, device=self.DEFAULT_DEVICE, dtype=torch.float32),
            indexing='xy'
        )
        if jitter is not None:
            # jitter has values of [-0.5, +0.5], transform to [-1/window_dim, +1/window_dim]
            interp_x = interp_x * (1.0 + jitter[:, :, 0] / window_w)
            interp_y = interp_y * (1.0 + jitter[:, :, 1] / window_h)
        interp_x = interp_x.unsqueeze(-1)
        interp_y = interp_y.unsqueeze(-1)
        rays_dir = c_ul + interp_x * (c_ur - c_ul) + interp_y * (c_ll - c_ul)
        rays_dir = torch.nn.functional.normalize(rays_dir, dim=-1)
        rays_dir = rays_dir.unsqueeze(0)
        return RayPack(
            rays_ori=rays_ori,
            rays_dir=rays_dir,
            pixel_x=torch.round(interp_x * window_w).squeeze(-1),
            pixel_y=torch.round(interp_y * window_h).squeeze(-1)
        )

    @torch.cuda.nvtx.range("_raygen_fisheye")
    def _raygen_fisheye(self, view_params, window_w, window_h, jitter) -> RayPack:
        eps = 1e-9
        fisheye_fov_rad = torch.deg2rad(
            torch.tensor([self.camera_fov], device=self.DEFAULT_DEVICE, dtype=torch.float32)
        )
        cx = window_w / 2
        cy = window_h / 2

        interp_x, interp_y = torch.meshgrid(
            torch.linspace(0., window_w, window_w, device=self.DEFAULT_DEVICE, dtype=torch.float32),
            torch.linspace(0., window_h, window_h, device=self.DEFAULT_DEVICE, dtype=torch.float32),
            indexing='xy'
        )

        u = (interp_x - cx) * 2. / window_w
        v = (interp_y - cy) * 2. / window_h

        r = torch.sqrt(u * u + v * v)
        out_of_fov_mask = (r > 1.0)[:, :, None]

        phi_cos = torch.where(torch.abs(r) > eps, u / r, 0.0)
        phi_cos = torch.clamp(phi_cos, -1.0, 1.0)
        phi = torch.arccos(phi_cos)
        phi = torch.where(v < 0, -phi, phi)
        theta = r * fisheye_fov_rad * 0.5

        rays_dir = torch.stack(
            [torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)], dim=2
        )
        mock_dir = torch.zeros_like(rays_dir)
        mock_dir[:, :, 0] = -1.0
        mock_dir[:, :, 1] = -.05
        rays_dir = torch.where(out_of_fov_mask, mock_dir, rays_dir).unsqueeze(0)

        # generate view camera ray origins
        cam_center = view_params.get_position()
        rays_ori = torch.tensor(cam_center,
                                device=self.DEFAULT_DEVICE, dtype=torch.float32).reshape(1, 1, 1, 3).expand(1, window_h,
                                                                                                            window_w, 3
                                                                                                            )

        return RayPack(
            rays_ori=rays_ori.contiguous(),
            rays_dir=rays_dir.contiguous(),
            pixel_x=torch.round(interp_x).contiguous(),
            pixel_y=torch.round(interp_y).contiguous(),
            mask=out_of_fov_mask
        )

    def raygen(self, view_params, window_w, window_h, use_spp=False) -> RayPack:
        ray_batch_size = 1 if not use_spp else self.spp.batch_size
        rays = []
        for _ in range(ray_batch_size):
            jitter = self.spp(window_h, window_w) if use_spp and self.spp is not None else None
            if self.camera_type == 'Pinhole':
                next_rays = self._raygen_pinhole(view_params, window_w, window_h, jitter)
            elif self.camera_type == 'Fisheye':
                next_rays = self._raygen_fisheye(view_params, window_w, window_h, jitter)
            else:
                raise ValueError(f"Unknown camera type: {self.camera_type}")
            rays.append(next_rays)
        return RayPack(
            mask=rays[0].mask,
            pixel_x=rays[0].pixel_x,
            pixel_y=rays[0].pixel_y,
            rays_ori=torch.cat([r.rays_ori for r in rays], dim=0),
            rays_dir=torch.cat([r.rays_dir for r in rays], dim=0)
        )

    @torch.cuda.nvtx.range("render_from_current_ps_view")
    @torch.no_grad()
    def render_from_current_ps_view(self, window_w=None, window_h=None):

        if window_w is None or window_h is None:
            window_w, window_h = ps.get_window_size()
        # If window size changed since the last render call, mark the canvas as dirty.
        # We check it here since the event comes from the windowing system and could prompt in between frame renders
        if self.last_state.get('window_size'):
            last_window_size = self.last_state.get('window_size')
            if (last_window_size[0] != window_h) or (last_window_size[1] != window_w):
                self.is_force_canvas_dirty = True

        if not self.is_dirty() and not self.has_progressive_effects_to_render():
            return self.last_state['rgb'], self.last_state['opacity']

        # Render a frame
        view_params = ps.get_view_camera_parameters()
        outputs = self.render(view_params, window_w, window_h)

        self.cache_last_state(view_params=view_params, outputs=outputs, window_size=(window_h, window_w))
        self.is_force_canvas_dirty = False
        return outputs['rgb'], outputs['opacity']

    def update_data_on_device(self, buffer, tensor_array):
        if self.ps_buffer_mode == 'host2device':
            buffer.update_data(tensor_array.detach().cpu().numpy())
        else:
            buffer.update_data_from_device(tensor_array.detach())

    @torch.cuda.nvtx.range("update_render_view_viz")
    @torch.no_grad()
    def update_render_view_viz(self, force=False):

        window_w, window_h = ps.get_window_size()

        # re-initialize if needed
        style = self.viz_render_styles[self.viz_render_style_ind]
        if force or self.viz_curr_render_style_ind != self.viz_render_style_ind or self.viz_curr_render_size != (
                window_w, window_h):
            self.viz_curr_render_style_ind = self.viz_render_style_ind
            self.viz_curr_render_size = (window_w, window_h)

            if style in ("color",):

                dummy_image = np.ones((window_h, window_w, 4), dtype=np.float32)

                ps.add_color_alpha_image_quantity(
                    self.viz_render_name,
                    dummy_image,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                )

                self.viz_render_color_buffer = ps.get_quantity_buffer(self.viz_render_name, "colors")
                self.viz_render_scalar_buffer = None

            elif style == "density":

                dummy_vals = np.zeros((window_h, window_w), dtype=np.float32)
                dummy_vals[0] = 1.0  # hack so the default polyscope scale gets set more nicely

                self.viz_main_image = ps.add_scalar_image_quantity(
                    self.viz_render_name,
                    dummy_vals,
                    enabled=self.viz_render_enabled,
                    image_origin="upper_left",
                    show_fullscreen=True,
                    show_in_imgui_window=False,
                    cmap="blues",
                    vminmax=(0, 1),
                )

                self.viz_render_color_buffer = None
                self.viz_render_scalar_buffer = ps.get_quantity_buffer(self.viz_render_name, "values")

        # do the actual rendering
        try:
            sple_orad, sple_odns = self.render_from_current_ps_view()
            sple_orad = sple_orad[0]
            sple_odns = sple_odns[0]
        except Exception:
            print("Rendering error occurred.")
            traceback.print_exc()
            return

        # update the data
        if style in ("color",):
            # append 1s for alpha
            sple_orad = torch.cat((sple_orad, torch.ones_like(sple_orad[:, :, 0:1])), dim=-1)
            self.update_data_on_device(self.viz_render_color_buffer, sple_orad)

        elif style == "density":
            self.update_data_on_device(self.viz_render_scalar_buffer, sple_odns)

    def _draw_preset_settings_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Quick Settings"):
            psim.PushItemWidth(150)

            if (psim.Button("Fast")):
                self.use_spp = False
                self.antialiasing_mode = '4x MSAA'
                self.spp.mode = 'msaa'
                self.spp.spp = 4
                self.spp.reset_accumulation()
                self.use_optix_denoiser = False
                self.is_force_canvas_dirty = True
            psim.SameLine()
            if (psim.Button("Balanced")):
                self.use_spp = True
                self.antialiasing_mode = '4x MSAA'
                self.spp.mode = 'msaa'
                self.spp.spp = 4
                self.spp.reset_accumulation()
                self.use_optix_denoiser = True
                self.is_force_canvas_dirty = True
            psim.SameLine()
            if (psim.Button("High Quality")):
                self.use_spp = True
                self.antialiasing_mode = 'Quasi-Random (Sobol)'
                self.spp.mode = 'low_discrepancy_seq'
                self.spp.spp = 64
                self.spp.reset_accumulation()
                self.use_optix_denoiser = True
                self.is_force_canvas_dirty = True

            psim.PushItemWidth(150)
            psim.TreePop()

    def _draw_render_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Render"):
            render_channel_changed, self.viz_render_style_ind = psim.Combo(
                "Style", self.viz_render_style_ind, self.viz_render_styles
            )
            if render_channel_changed:
                self.rebuild_bvh(self.scene_mog)
                self.is_force_canvas_dirty = True

            cam_idx = Playground.AVAILABLE_CAMERAS.index(self.camera_type)
            is_cam_changed, new_cam_idx = psim.Combo("Camera", cam_idx, Playground.AVAILABLE_CAMERAS)
            if is_cam_changed:
                self.camera_type = Playground.AVAILABLE_CAMERAS[new_cam_idx]
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_cam_changed
            if self.camera_type == 'Fisheye':
                psim.SameLine()
                is_cam_changed, self.camera_fov = psim.SliderFloat(
                    "FoV", self.camera_fov, v_min=60.0, v_max=180.0
                )
                self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_cam_changed

            psim.PushItemWidth(100)
            settings_changed, self.gamma_correction = psim.SliderFloat(
                "Gamma Correction", self.gamma_correction, v_min=0.5, v_max=3.0
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.SameLine()

            settings_changed, self.max_pbr_bounces = psim.SliderInt(
                "Max PBR Bounces", self.max_pbr_bounces, v_min=1, v_max=15
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.PopItemWidth()

            settings_changed, self.use_optix_denoiser = psim.Checkbox("Use Optix Denoiser", self.use_optix_denoiser)
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.PopItemWidth()
            psim.TreePop()

    def _draw_video_recording_controls(self):
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Record Trajectory Video"):
            _, self.trajectory_output_path = psim.InputText("Video Output Path", self.trajectory_output_path)
            if (psim.Button("Add Camera")):
                view_params = ps.get_view_camera_parameters()
                cam_center = view_params.get_position()
                self.trajectory.append((
                    cam_center,
                    cam_center + view_params.get_look_dir(),
                    view_params.get_up_dir()
                ))
            psim.SameLine()
            if (psim.Button("Reset")):
                self.trajectory = []
            psim.SameLine()
            if (psim.Button("Render Video")):
                self.render_trajectory()
            psim.SameLine()
            changed, self.continuous_trajectory = psim.Checkbox("Continuous Trajectory", self.continuous_trajectory)
            psim.PushItemWidth(150)
            _, self.frames_between_cameras = psim.SliderInt(
                "Frames Between", self.frames_between_cameras, v_min=1, v_max=120
            )
            psim.SameLine()
            _, self.trajectory_fps = psim.SliderInt(
                "FPS", self.trajectory_fps, v_min=1, v_max=120
            )
            _, self.window_w = psim.SliderInt(
                "Width", self.window_w, v_min=1, v_max=8192
            )
            psim.SameLine()
            _, self.window_h = psim.SliderInt(
                "Height", self.window_h, v_min=1, v_max=8192
            )
            psim.PopItemWidth()
            psim.Text(f"There are {len(self.trajectory)} cameras in the trajectory.")

            if len(self.trajectory) > 0 and psim.TreeNode("Cameras"):
                remained_cameras = []
                for i, (eye, target, up) in enumerate(self.trajectory):
                    is_not_removed = self._draw_single_trajectory_camera(i, eye, target, up)
                    remained_cameras.append(is_not_removed)
                self.trajectory = [self.trajectory[i] for i in range(len(self.trajectory)) if remained_cameras[i]]

                psim.TreePop()

            if psim.TreeNode("Load/Save a Trajectory"):
                _, self.cameras_save_path = psim.InputText("Cameras' Path", self.cameras_save_path)
                if (psim.Button("Save Trajectory")):
                    np.save(self.cameras_save_path, self.trajectory)
                psim.SameLine()
                if (psim.Button("Load Trajectory")):
                    self.trajectory = list(np.load(self.cameras_save_path))

                psim.TreePop()

            _, self.use_dof_in_trajectory = psim.Checkbox("Use DoF", self.use_dof_in_trajectory)
            psim.PushItemWidth(100)
            psim.SameLine()
            _, self.min_dof = psim.SliderFloat(
                "Min FoV", self.min_dof, v_min=0.0, v_max=24.0
            )
            psim.SameLine()
            _, self.max_dof = psim.SliderFloat(
                "Max FoV", self.max_dof, v_min=0.0, v_max=24.0
            )
            psim.PopItemWidth()

            psim.PopItemWidth()
            psim.TreePop()

    def _draw_slice_plane_controls(self):

        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Slice Planes"):
            any_plane_changed = False
            for sp_idx, slice_plane in enumerate(self.slice_planes):
                self.slice_planes[sp_idx].set_draw_widget(False)

                changed, is_enabled = psim.Checkbox(f"Slice Plane {sp_idx}", self.slice_plane_enabled[sp_idx])
                if changed:
                    # self.slice_planes[sp_idx].set_draw_plane(is_enabled)
                    self.slice_plane_enabled[sp_idx] = is_enabled
                    self.slice_planes[sp_idx].set_draw_plane(False)
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])
                any_plane_changed |= changed

                psim.PushItemWidth(350)
                changed, values = psim.SliderFloat3(
                    f"SPPos{sp_idx}",
                    [self.slice_plane_pos[sp_idx][0], self.slice_plane_pos[sp_idx][1], self.slice_plane_pos[sp_idx][2]],
                    v_min=-10.0, v_max=10.0,
                    format="%.2f",
                    power=1.0
                )
                any_plane_changed |= changed
                if changed:
                    self.slice_plane_pos[sp_idx] = values
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])

                changed, values = psim.SliderFloat3(
                    f"SPNorm{sp_idx}",
                    [self.slice_plane_normal[sp_idx][0], self.slice_plane_normal[sp_idx][1],
                     self.slice_plane_normal[sp_idx][2]],
                    v_min=-180.0, v_max=180.0,
                    format="%.2f",
                    power=1.0
                )
                any_plane_changed |= changed
                if changed:
                    self.slice_plane_normal[sp_idx] = values
                    self.slice_planes[sp_idx].set_pose(self.slice_plane_pos[sp_idx], self.slice_plane_normal[sp_idx])

                psim.PopItemWidth()

            if any_plane_changed:
                self._recompute_slice_planes()

            psim.TreePop()

    @torch.no_grad()
    def _recompute_slice_planes(self):
        p1 = self.scene_mog.positions
        enabled_planes = p1.new_tensor(self.slice_plane_enabled, dtype=torch.bool)
        self.scene_mog.density = copy.deepcopy(self.density_buffer)
        if enabled_planes.any():
            p0 = p1.new_tensor(self.slice_plane_pos)[None, enabled_planes]
            n = p1.new_tensor(self.slice_plane_normal)[None, enabled_planes]
            is_inside = (torch.sum((p1[:, None] - p0) * n, dim=-1) > 0).all(dim=1)
            self.scene_mog.density[~is_inside] = -100000  # Empty density
        self.is_force_canvas_dirty = True

    def _draw_antialiasing_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Antialiasing"):
            psim.PushItemWidth(150)
            settings_changed, self.use_spp = psim.Checkbox(
                "Enable", self.use_spp
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            aa_index = self.ANTIALIASING_MODES.index(self.antialiasing_mode)
            psim.PushItemWidth(200)
            is_antialiasing_changed, aa_index = psim.Combo(
                "Mode", aa_index, self.ANTIALIASING_MODES
            )
            psim.PopItemWidth()
            if is_antialiasing_changed:
                self.antialiasing_mode = self.ANTIALIASING_MODES[aa_index]
                # '4x MSAA', '8x MSAA', '16x MSAA', 'Quasi-Random (Sobol)'
                if self.antialiasing_mode == '4x MSAA':
                    self.spp.mode = 'msaa'
                    self.spp.spp = 4
                elif self.antialiasing_mode == '8x MSAA':
                    self.spp.mode = 'msaa'
                    self.spp.spp = 8
                elif self.antialiasing_mode == '16x MSAA':
                    self.spp.mode = 'msaa'
                    self.spp.spp = 16
                elif self.antialiasing_mode == 'Quasi-Random (Sobol)':
                    self.spp.mode = 'low_discrepancy_seq'
                    self.spp.spp = 64
                self.spp.reset_accumulation()
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or is_antialiasing_changed

            psim.SameLine()
            psim.PushItemWidth(75)
            spp_min, spp_max = 1, 256
            if self.antialiasing_mode == '4x MSAA':
                spp_min, spp_max = 4, 4
            elif self.antialiasing_mode == '8x MSAA':
                spp_min, spp_max = 8, 8
            elif self.antialiasing_mode == '16x MSAA':
                spp_min, spp_max = 16, 16
            settings_changed, self.spp.spp = psim.SliderInt(
                "SPP", self.spp.spp, v_min=spp_min, v_max=spp_max
            )
            psim.PopItemWidth()
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            settings_changed, self.spp.batch_size = psim.SliderInt(
                "Batch Size (#Frames)", self.spp.batch_size, v_min=1, v_max=min(1024, self.spp.spp)
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            if self.use_spp:
                panel_width = psim.GetContentRegionAvail()[0]
                progress_width = round(panel_width / 1.2)
                progress_label = f'{str(self.spp.spp_accumulated_for_frame)}/{str(self.spp.spp)}'
                psim.ProgressBar(fraction=self.spp.spp_accumulated_for_frame / self.spp.spp,
                                 size_arg=(progress_width, 0))

            psim.TreePop()

    def _draw_depth_of_field_widget(self):
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Depth of Field"):
            settings_changed, self.use_depth_of_field = psim.Checkbox(
                "Enable", self.use_depth_of_field
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            settings_changed, self.depth_of_field.spp = psim.SliderInt(
                "SPP", self.depth_of_field.spp, v_min=1, v_max=256
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            settings_changed, self.depth_of_field.focus_z = psim.SliderFloat(
                "Focus Z", self.depth_of_field.focus_z, v_min=0.25, v_max=24.0
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            psim.SameLine()
            settings_changed, self.depth_of_field.aperture_size = psim.SliderFloat(
                "Aperture Size", self.depth_of_field.aperture_size, v_min=1e-5, v_max=1e-1, power=10)
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

            if self.use_depth_of_field:
                panel_width = psim.GetContentRegionAvail()[0]
                progress_width = round(panel_width / 1.2)
                progress_label = f'{str(self.depth_of_field.spp_accumulated_for_frame)}/{str(self.depth_of_field.spp)}'
                psim.ProgressBar(fraction=self.depth_of_field.spp_accumulated_for_frame / self.depth_of_field.spp,
                                 size_arg=(progress_width, 0))

            psim.TreePop()

    def _draw_primitives_widget(self):
        removed_objs = []
        duped_objs = []
        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Primitives"):
            self._draw_general_primitive_settings_widget()
            for obj_name, obj in self.primitives.objects.items():

                psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                if psim.TreeNode(obj_name):

                    is_retained, is_duplicated = self._draw_single_primitive_main_settings_widget(obj_name, obj)
                    if not is_retained:
                        removed_objs.append(obj_name)
                    if is_duplicated:
                        duped_objs.append(obj_name)

                    psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
                    if psim.TreeNode("Properties"):

                        if obj.primitive_type == OptixPrimitiveTypes.DIFFUSE:
                            self._draw_diffuse_pbr_settings_widget(obj)
                        if obj.primitive_type == OptixPrimitiveTypes.GLASS:
                            self._draw_glass_settings_widget(obj)
                        elif obj.primitive_type == OptixPrimitiveTypes.MIRROR:
                            self._draw_mirror_settings_widget(obj)
                        psim.TreePop()

                    self._draw_transform_widget(obj)
                    psim.TreePop()
            psim.TreePop()

        psim.PopItemWidth()

        for obj_name in removed_objs:
            self.primitives.remove_primitive(obj_name)
            self.is_force_canvas_dirty = True

        for obj_name in duped_objs:
            self.primitives.duplicate_primitive(obj_name)
            self.is_force_canvas_dirty = True

    def _draw_materials_widget(self):
        material_changed = False
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Materials"):

            for mat_name, material in self.primitives.registered_materials.items():
                psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
                if psim.TreeNode(mat_name + f' [ID #{material.material_id}]'):
                    changed, values = psim.SliderFloat3(
                        "Diffuse Factor",
                        [material.diffuse_factor[0], material.diffuse_factor[1], material.diffuse_factor[2]],
                        v_min=0.0, v_max=1.4,
                        format="%.3f",
                        power=1.0
                    )
                    if changed:
                        material.diffuse_factor[0] = values[0]
                        material.diffuse_factor[1] = values[1]
                        material.diffuse_factor[2] = values[2]
                        material_changed = True

                    changed, values = psim.SliderFloat3(
                        "Emissive Factor",
                        [material.emissive_factor[0], material.emissive_factor[1], material.emissive_factor[2]],
                        v_min=0.0, v_max=1.0,
                        format="%.3f",
                        power=1.0
                    )
                    if changed:
                        material.emissive_factor[0] = values[0]
                        material.emissive_factor[1] = values[1]
                        material.emissive_factor[2] = values[2]
                        material_changed = True

                    changed, value = psim.SliderFloat("Metallic Factor", material.metallic_factor,
                                                      v_min=0.0, v_max=1.0, power=1)
                    if changed:
                        material.metallic_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("Roughness Factor", material.roughness_factor,
                                                      v_min=0.0, v_max=1.0, power=1)
                    if changed:
                        material.roughness_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("Transmission Factor", material.transmission_factor,
                                                      v_min=0.0, v_max=1.0, power=1)
                    if changed:
                        material.transmission_factor = value
                        material_changed = True

                    changed, value = psim.SliderFloat("IOR", material.ior,
                                                      v_min=0.2, v_max=2.0, power=1)
                    if changed:
                        material.ior = value
                        material_changed = True

                    if material.diffuse_map is not None:
                        psim.Text(f"Diffuse Texture: {material.diffuse_map.shape[0]}x{material.diffuse_map.shape[1]}")
                    else:
                        psim.Text(f"Diffuse Texture: No")

                    if material.emissive_map is not None:
                        psim.Text(
                            f"Emissive Texture: {material.emissive_map.shape[0]}x{material.emissive_map.shape[1]}")
                    else:
                        psim.Text(f"Emissive Texture: No")

                    if material.metallic_roughness_map is not None:
                        psim.Text(
                            f"Metal-Rough Texture: {material.metallic_roughness_map.shape[0]}x{material.metallic_roughness_map.shape[1]}")
                    else:
                        psim.Text(f"Metal-Rough Texture: No")

                    if material.normal_map is not None:
                        psim.Text(f"Normal Texture: {material.normal_map.shape[0]}x{material.normal_map.shape[1]}")
                    else:
                        psim.Text(f"Normal Texture: No")

                    psim.TreePop()
            psim.TreePop()

        if material_changed:
            self.is_force_canvas_dirty = True

    def _draw_general_primitive_settings_widget(self):
        primitives_disabled = not self.primitives.enabled
        settings_changed, self.primitives.enabled = psim.Checkbox(
            "Disable Primitives", primitives_disabled
        )
        self.primitives.enabled = not self.primitives.enabled
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()
        settings_changed, self.primitives.use_smooth_normals = psim.Checkbox(
            "Smooth Normals", self.primitives.use_smooth_normals
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()
        settings_changed, self.primitives.disable_pbr_textures = psim.Checkbox(
            "Disable Textures", self.primitives.disable_pbr_textures
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        settings_changed, self.primitives.disable_gaussian_tracing = psim.Checkbox(
            "Disable Gaussians", self.primitives.disable_gaussian_tracing
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()

        settings_changed, self.primitives.force_white_bg = psim.Checkbox(
            "Force White BG", self.primitives.force_white_bg
        )
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
        if self.envmap is not None:
            psim.SameLine()
            settings_changed, self.primitives.enable_envmap = psim.Checkbox(
                "Enable Envmap", self.primitives.enable_envmap
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed
            psim.SameLine()
            settings_changed, self.primitives.use_envmap_as_background = psim.Checkbox(
                "Use Envmap As Background", self.primitives.use_envmap_as_background
            )
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.PushItemWidth(100)
        is_add_primitive = psim.Button("Add Primitive")
        psim.PopItemWidth()

        psim.SameLine()
        if 'add_geom_select_type' not in self.gui_aux_fields:
            self.gui_aux_fields['add_geom_select_type'] = 0
        available_geometries = sorted(list(self.primitives.assets.keys()))
        _, new_geom_select_type_idx = psim.Combo(
            "Geometry", self.gui_aux_fields['add_geom_select_type'], available_geometries
        )
        self.gui_aux_fields['add_geom_select_type'] = new_geom_select_type_idx

        if is_add_primitive:
            geom_idx = self.gui_aux_fields['add_geom_select_type']
            self.primitives.add_primitive(
                geometry_type=available_geometries[geom_idx],
                primitive_type=OptixPrimitiveTypes.GLASS,
                device=self.scene_mog.device
            )
            self.primitives.rebuild_bvh_if_needed(True, True)
            self.is_force_canvas_dirty = True

        psim.SameLine()

        psim.PushItemWidth(100)
        scenes_folder = 'playground_scenes'
        scene_path = os.path.join(scenes_folder, self.scene_name) + '.pt'
        if psim.Button("Save"):
            os.makedirs(scenes_folder, exist_ok=True)
            data = dict(
                objects=self.primitives.objects,
                materials=self.primitives.registered_materials,
                slice_plane_enabled=self.slice_plane_enabled,
                slice_plane_pos=self.slice_plane_pos,
                slice_plane_normal=self.slice_plane_normal
            )
            torch.save(data, scene_path)

            print(f'Scene saved to {scene_path}')
        psim.SameLine()
        if psim.Button("Load"):
            if not os.path.exists(scene_path):
                ps.warning(f'No data stored for scene under {scene_path}')
            else:
                data = torch.load(scene_path)
                self.primitives.objects = data['objects']
                self.primitives.registered_materials = data.get('materials', dict())
                self.slice_plane_enabled = data['slice_plane_enabled']
                self.slice_plane_pos = data['slice_plane_pos']
                self.slice_plane_normal = data['slice_plane_normal']
                self._recompute_slice_planes()
                self.primitives.rebuild_bvh_if_needed(force=True, rebuild=True)
                self.is_force_canvas_dirty = True
                print(f'Scene loaded from {scene_path}')
        psim.PopItemWidth()

    def _draw_single_primitive_main_settings_widget(self, obj_name, obj):
        available_primitive_modes = OptixPrimitiveTypes.names()
        settings_changed, new_prim_type_idx = psim.Combo(
            "Type", obj.primitive_type.value, available_primitive_modes
        )
        if settings_changed:
            obj.primitive_type = OptixPrimitiveTypes(new_prim_type_idx)
            self.primitives.recompute_stacked_buffers()
            self.primitives.rebuild_bvh_if_needed(True, True)  # Rebuild so None types are truly ignored in BVH
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

        psim.SameLine()

        is_retained = True
        is_duplicated = False
        psim.PushItemWidth(100)  # button_width
        if psim.Button("Remove"):
            is_retained = False
        psim.SameLine()
        if psim.Button("Duplicate"):
            is_duplicated = True
        psim.PopItemWidth()

        return is_retained, is_duplicated

    def _draw_transform_widget(self, obj):

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if psim.TreeNode("Transform"):
            object_transform = obj.transform
            transform_changed = False
            psim.PushItemWidth(100)  # button_width
            if psim.Button("Reset"):
                object_transform.reset()
                transform_changed = True
            psim.PopItemWidth()

            psim.PushItemWidth(350)
            changed, values = psim.SliderFloat3(
                "Translate",
                [object_transform.tx, object_transform.ty, object_transform.tz],
                v_min=-5.0, v_max=5.0,
                format="%.4f",
                power=1.0
            )
            if changed:
                object_transform.tx = values[0]
                object_transform.ty = values[1]
                object_transform.tz = values[2]
                transform_changed = True

            changed, values = psim.SliderFloat3(
                "Rotate",
                [object_transform.rx, object_transform.ry, object_transform.rz],
                v_min=-180.0, v_max=180.0,
                format="%.3f",
                power=1.0
            )
            if changed:
                object_transform.rx = values[0]
                object_transform.ry = values[1]
                object_transform.rz = values[2]
                transform_changed = True

            changed, values = psim.SliderFloat3(
                "Scale",
                [object_transform.sx, object_transform.sy, object_transform.sz],
                v_min=-5.0, v_max=5.0,
                format="%.4f",
                power=1.0
            )
            if changed:
                object_transform.sx = values[0]
                object_transform.sy = values[1]
                object_transform.sz = values[2]
                transform_changed = True
            psim.PopItemWidth()

            if transform_changed:
                self.primitives.rebuild_bvh_if_needed(force=True, rebuild=False)
                self.is_force_canvas_dirty = True
            psim.TreePop()

    def _draw_diffuse_pbr_settings_widget(self, obj):
        has_single_material = torch.min(obj.material_id) == torch.max(obj.material_id)
        if not has_single_material:
            psim.Text('Multiple Materials')
        else:
            current_mat_id = obj.material_id[0].item()
            mat_id_to_mat_idx = {m.material_id: idx for idx, (m_name, m) in
                                 enumerate(self.primitives.registered_materials.items())}
            mat_name_to_mat_idx = {m_name: idx for idx, (m_name, m) in
                                   enumerate(self.primitives.registered_materials.items())}
            mat_idx_to_mat_id = {v: k for k, v in mat_id_to_mat_idx.items()}
            settings_changed, new_mat_idx = psim.Combo("Material", current_mat_id, list(mat_name_to_mat_idx.keys()))
            if settings_changed:
                obj.material_id[:] = mat_idx_to_mat_id[new_mat_idx]
                self.primitives.recompute_stacked_buffers()
                self.primitives.rebuild_bvh_if_needed(True, True)  # Rebuild so None types are truly ignored in BVH
            self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

    def _draw_glass_settings_widget(self, obj):
        settings_changed, obj.refractive_index = psim.SliderFloat(
            "Refractive Index", obj.refractive_index, v_min=0.5, v_max=2.0, power=1)
        if settings_changed:
            self.primitives.recompute_stacked_buffers()
        self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

    def _draw_mirror_settings_widget(self, obj):
        pass
        # settings_changed, self.mirrors.mirror_scatter = psim.SliderFloat(
        #     "Scatter (Imperfectness)", self.mirrors.mirror_scatter, v_min=0.0, v_max=1e-2, power=1)
        # self.is_force_canvas_dirty = self.is_force_canvas_dirty or settings_changed

    def _draw_single_trajectory_camera(self, i, eye, target, up):
        psim.PushItemWidth(200)
        if (psim.Button(f"view {i + 1}")):
            ps.look_at_dir(eye, target, up, fly_to=True)

        psim.SameLine()
        if (psim.Button(f"remove {i + 1}")):
            is_not_removed = False
        else:
            is_not_removed = True

        psim.SameLine()
        if psim.Button(f"replace {i + 1}"):
            view_params = ps.get_view_camera_parameters()
            cam_center = view_params.get_position()
            self.trajectory[i] = (
                cam_center,
                cam_center + view_params.get_look_dir(),
                view_params.get_up_dir()
            )

        psim.PopItemWidth()

        return is_not_removed

    @torch.cuda.nvtx.range("ps_ui_callback")
    def ps_ui_callback(self):
        self._draw_preset_settings_widget()
        psim.Separator()
        self._draw_render_widget()
        psim.Separator()
        self._draw_video_recording_controls()
        psim.Separator()
        self._draw_slice_plane_controls()
        psim.Separator()
        self._draw_antialiasing_widget()
        psim.Separator()
        self._draw_depth_of_field_widget()
        psim.Separator()
        self._draw_materials_widget()
        psim.Separator()
        self._draw_primitives_widget()

        if self.live_update:
            self.update_render_view_viz()
