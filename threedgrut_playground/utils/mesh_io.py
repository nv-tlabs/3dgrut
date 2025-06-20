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

import PIL
import numpy as np
import torch
import igl
from pathlib import Path
from pygltflib import GLTF2, OPAQUE, BLEND, MASK

# Silence dependencies that log unrelated warnings
import logging
import warnings
with warnings.catch_warnings():
    logging.getLogger('kaolin.render.mesh.nvdiffrast_context').setLevel(logging.WARNING)
    import kaolin

def _to_tensor(mat_data, is_normalize=False, is_pad_to_float4=False, device=None):
    if mat_data is None:
        return None
    elif isinstance(mat_data, PIL.Image.Image):
        mat_data = np.array(mat_data)

    if isinstance(mat_data, torch.Tensor):
        mat_tensor = mat_data.clone().detach().to(device=device)
    else:
        mat_tensor = torch.tensor(mat_data, device=device)

    if is_normalize and not torch.is_floating_point(mat_tensor):
        mat_tensor = mat_tensor.float() / 255.0
    # pad float3 texture to float4 due to cuda's tex2d
    if is_pad_to_float4 and mat_tensor.ndim > 2 and mat_tensor.shape[2] == 3:
        pad = mat_tensor.new_ones(mat_tensor.shape[0], mat_tensor.shape[1], 1)
        mat_tensor = torch.cat([mat_tensor, pad], dim=-1)
    elif is_pad_to_float4 and mat_tensor.ndim == 1:
        pad = mat_tensor.new_ones(1)
        mat_tensor = torch.cat([mat_tensor, pad], dim=-1)
    return mat_tensor.float()


@torch.no_grad()
def load_missing_material_info(path, materials, device):
    """ Populates materials with additional material fields currently missing from kaolin.
    """
    enum2alphamode = {
        OPAQUE: 0,
        BLEND: 1,
        MASK: 2
    }

    scene = GLTF2.load(path)
    scene_mats = {m.name: m for m in scene.materials}
    for mat in materials:
        missing_mat = scene_mats[mat['material_name']]
        emissiveFactor = missing_mat.emissiveFactor if missing_mat.emissiveFactor is not None else torch.zeros(3, device=device)
        emissiveTexture = None # TODO (operel): Support texture load
        alphaMode = enum2alphamode[missing_mat.alphaMode]
        alphaCutoff = missing_mat.alphaCutoff
        mat['emissive_map'] = _to_tensor(emissiveTexture, is_normalize=True, is_pad_to_float4=True)
        mat['emissive_factor'] = _to_tensor(emissiveFactor, is_normalize=True, device='cpu')
        mat['alpha_mode'] = alphaMode
        mat['alpha_cutoff'] = alphaCutoff


@torch.no_grad()
def load_materials(mesh, device):
    """ Loads additional material fields currently missing from kaolin.
    """
    out_mats = []
    materials = mesh.materials
    if materials is None:
        return []

    for mat in materials:
        diffuseFactor = mat.diffuse_color if mat.diffuse_color is not None \
            else torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

        diffuseTexture = mat.diffuse_texture
        if diffuseTexture is not None:
            # TODO (operel): diffuse should be read with alpha channel from kaolin
            diffuseAlphaTexture = torch.ones_like(mat.diffuse_texture[:, :, 0:1])
            diffuseTexture = torch.cat([diffuseTexture, diffuseAlphaTexture], dim=2)

        # Create a 2dim texture
        metallicRoughnessTexture = None
        if mat.metallic_texture is not None and mat.roughness_texture is not None:
            metallicRoughnessTexture = torch.cat([
                mat.metallic_texture,
                mat.roughness_texture,
            ], dim=2)
        elif mat.metallic_texture is not None and mat.roughness_texture is None:
            metallicRoughnessTexture = torch.cat([
                mat.metallic_texture,
                torch.zeros_like(mat.metallic_texture)
            ], dim=2)
        elif mat.metallic_texture is  None and mat.roughness_texture is not None:
            metallicRoughnessTexture = torch.cat([
                torch.zeros_like(mat.roughness_texture),
                mat.roughness_texture,
            ], dim=2)

        if mat.metallic_value is not None:
            metallicFactor = mat.metallic_value
        elif metallicRoughnessTexture is not None:
            metallicFactor = 1.0
        else:
            metallicFactor = 0.0
        if mat.roughness_value is not None:
            roughnessFactor = mat.roughness_value
        elif metallicRoughnessTexture is not None:
            roughnessFactor = 1.0
        else:
            roughnessFactor = 0.0

        # kaolin pre-scales normal channel: https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/io/gltf.py#L241
        normalTexture = mat.normals_texture

        transmissionFactor = mat.transmittance_value if mat.transmittance_value is not None else 0.0
        ior = mat.ior_value if mat.ior_value is not None else 1.0

        loaded_mat = dict(
            material_name=mat.material_name,
            diffuse_map=_to_tensor(diffuseTexture, is_normalize=True, is_pad_to_float4=True),
            metallic_roughness_map=_to_tensor(metallicRoughnessTexture, is_normalize=False, is_pad_to_float4=True),
            normal_map=_to_tensor(normalTexture, is_normalize=False, is_pad_to_float4=True),
            diffuse_factor=_to_tensor(diffuseFactor, is_normalize=True, device='cpu', is_pad_to_float4=True),
            metallic_factor=float(metallicFactor),
            roughness_factor=float(roughnessFactor),
            transmission_factor=float(transmissionFactor),
            ior=ior
        )
        out_mats.append(loaded_mat)
    return out_mats


@torch.no_grad()
def load_mesh(path: str, device):
    """ Load mesh from path with kaolin.
    Supported formats: .obj, .gltf, .glb
    """
    format = Path(path).suffix
    if format in ('.obj', '.gltf', '.glb'):
        mesh = kaolin.io.import_mesh(path, triangulate=True)
    else:
        raise ValueError(f'Cannot load mesh asset with unsupported format: {format}. '
                         f'Supported types: .obj, .glb, .gltf')

    mesh = mesh.float_tensors_to(torch.float32)

    # Center object
    mesh.vertices -= mesh.vertices.mean(dim=0)

    # Compute vertex normals if needed
    if not mesh.has_attribute('vertex_normals') or len(mesh.vertex_normals) == 0:
        vertex_normals = igl.per_vertex_normals(mesh.vertices.numpy(), mesh.faces.numpy())
        mesh.vertex_normals = torch.tensor(vertex_normals, device=device, dtype=torch.float32)

    num_verts = len(mesh.vertices)
    num_faces = len(mesh.faces)
    mesh = mesh.to(device=device)

    mesh.vertex_tangents = mesh.vertex_tangents if mesh.has_attribute('vertex_tangents') else None

    mesh.uvs = mesh.uvs if mesh.has_attribute('uvs') else mesh.vertices.new_zeros(num_verts, 2)
    # If uvs + face_uvs_idx are available, use to compute face_uvs
    if not mesh.has_attribute('face_uvs'):
        if mesh.has_attribute('face_uvs_idx') and len(mesh.uvs) > 0:
            mesh.face_uvs = mesh.uvs[mesh.face_uvs_idx].contiguous()
        else:
            mesh.face_uvs = mesh.vertices.new_zeros(num_faces, 3, 2)
    mesh.material_assignments = mesh.material_assignments if mesh.has_attribute('material_assignments') else\
        torch.zeros([num_faces], device=device)

    return mesh


def create_procedural_mesh(vertices, faces, face_uvs, device):
    mesh = kaolin.rep.SurfaceMesh(vertices=vertices, faces=faces, face_uvs=face_uvs)
    mesh.vertex_tangents = None
    mesh.material_assignments = torch.zeros([len(mesh.faces)], device=device)
    return mesh.to(device)


def create_quad_mesh(device):
    """ Creates a procedurally generated quad mesh. """
    MS = 1.0
    MZ = 2.5
    v0 = [-MS, -MS, MZ]
    v1 = [-MS, +MS, MZ]
    v2 = [+MS, -MS, MZ]
    v3 = [+MS, +MS, MZ]
    faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
    vertex_uvs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    mesh = create_procedural_mesh(
        vertices=torch.tensor([v0, v1, v2, v3]),
        faces=faces,
        face_uvs=vertex_uvs[faces].contiguous(), # (F, 3, 2)
        device=device
    )
    return mesh
