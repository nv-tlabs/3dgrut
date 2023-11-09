# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from dataclasses import dataclass
import torch
import torch.utils.cpp_extension

#----------------------------------------------------------------------------
# 
_plugin = None
if _plugin is None:

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        optix_include_dir = os.path.dirname(__file__) + r"\include"

        def find_cl_path():
            import glob
            for edition in ['Enterprise', 'Professional', 'BuildTools', 'Community']:
                paths = sorted(glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\%s\VC\Tools\MSVC\*\bin\Hostx64\x64" % edition), reverse=True)
                if paths:
                    return paths[0]

        # If cl.exe is not on path, try to find it.
        if os.system("where cl.exe >nul 2>nul") != 0:
            cl_path = find_cl_path()
            if cl_path is None:
                raise RuntimeError("Could not locate a supported Microsoft Visual C++ installation")
            os.environ['PATH'] += ';' + cl_path

    elif os.name == 'posix':
        optix_include_dir = os.path.dirname(__file__) + r"/include"

    include_paths = [optix_include_dir]

    # Compiler options.
    opts = ['-DNVDR_TORCH']

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/common.cpp',
        'c_src/optix_wrapper.cpp',
        'c_src/torch_bindings.cpp',
        'c_src/gaussianEnclosingPrim.cu'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    torch.utils.cpp_extension.load(name='optixtracer_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=opts, extra_ldflags=ldflags, extra_include_paths=include_paths, with_cuda=True, verbose=True)

    # Import, cache, and return the compiled module.
    import optixtracer_plugin
    _plugin = optixtracer_plugin

#----------------------------------------------------------------------------
# 
class _trace_mog_func(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph):
        ray_radiance, ray_density, ray_hit_distance = _plugin.trace_mog(
            optix_ctx.cpp_wrapper,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph
        )
        ctx.save_for_backward(ray_ori, ray_dir, ray_radiance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph)
        ctx.optix_ctx = optix_ctx
        return ray_radiance, ray_density, ray_hit_distance
    
    @staticmethod
    def backward(ctx, ray_radiance_grd, ray_density_grd, ray_hit_distance_grd):
        optix_ctx = ctx.optix_ctx
        ray_ori, ray_dir, ray_radiance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph = ctx.saved_variables
        mog_pos_grd, mog_rot_grd, mog_scl_grd, mog_dns_grd, mog_sph_grd = _plugin.trace_mog_bwd(
            optix_ctx.cpp_wrapper,
            ray_ori,
            ray_dir,
            ray_radiance,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            ray_radiance_grd,
            ray_density_grd,
            ray_hit_distance_grd
        )
        return None, None, None, mog_pos_grd, mog_rot_grd, mog_scl_grd, mog_dns_grd, mog_sph_grd

def trace_mog(optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph):
    ray_radiance, ray_density, ray_hit_distance = _trace_mog_func.apply(
        optix_ctx,
        ray_ori,
        ray_dir,
        mog_pos,
        mog_rot,
        mog_scl,
        mog_dns,
        mog_sph
    )
    return ray_radiance, ray_density, ray_hit_distance

def trace_mog_grad(optix_ctx, ray_ori, ray_dir, ray_radiance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph, ray_radiance_grd, ray_density_grd, ray_hit_distance_grd):
    return _plugin.trace_mog_bwd(
        optix_ctx.cpp_wrapper,
        ray_ori,
        ray_dir,
        ray_radiance,
        mog_pos,
        mog_rot,
        mog_scl,
        mog_dns,
        mog_sph,
        ray_radiance_grd,
        ray_density_grd,
        ray_hit_distance_grd
    )

#----------------------------------------------------------------------------
#
def build_mog_bvh(
        optix_ctx,
        mog_pos,
        mog_rot,
        mog_scl,
        rebuild
):
    _plugin.build_mog_bvh(
        optix_ctx.cpp_wrapper,
        mog_pos.view(-1, 3),
        mog_rot.view(-1, 4),
        mog_scl.view(-1, 3),
        rebuild
    )

#----------------------------------------------------------------------------
#
@dataclass
class OptixMogTracingParams:
    pipeline : int=1            # Rendering algo : 0 = closest-hit, 1 = any-hit (the one you really want), 2 = intersection-shader, 3 = mlat
    hit_mode : int=0            # Intersection distance : 0 = distance to the first hit triangle, 1 = distance to the projection of aussian center on the ray 
    max_hit_per_slab: int=32    # Size of the array of sorted gaussian per-slabs
    max_num_slabs: int=64       # Number of slabs along the diagonal of the scene AABB
    topk_hits: bool=False       # If true do not respawn rays, just keep the first max_hit_per_slab gaussians
    patch_size: int=1           # tile size (best performances achieve with 3 but need COHERENT rays inputs)
    sph_degree: int=3           # spherical harmonics degree
    gaussian_sigma_threshold: float=3.0 # sigma factor to decide the size of the octahedron enveloppe
    min_transmittance: float=0.03       # minimum transmittance at which we stop gathering gaussians

class OptiXContext:
    def __init__(self,params:OptixMogTracingParams=OptixMogTracingParams()):
        print("Cuda path", torch.utils.cpp_extension.CUDA_HOME)
        torch.zeros(1, device='cuda') # Create a dummy tensor to force cuda context init
        self.cpp_wrapper = _plugin.OptiXStateWrapper(
            os.path.dirname(__file__), 
            torch.utils.cpp_extension.CUDA_HOME,
            params.pipeline,
            params.hit_mode,
            params.max_hit_per_slab,
            params.max_num_slabs,
            params.topk_hits,
            params.patch_size,
            params.sph_degree,
            params.gaussian_sigma_threshold,
            params.min_transmittance
        )
    
    def set_sph_degree(self, degree:int):
        self.cpp_wrapper.set_sph_degree(degree)

