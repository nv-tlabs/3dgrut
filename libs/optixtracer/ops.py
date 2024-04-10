# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from enum import IntEnum
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
    def forward(ctx, optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph, mog_err_target):
        ray_radiance, ray_density, ray_hit_distance, hits_count, g_weights = _plugin.trace_mog(
            optix_ctx.cpp_wrapper,
            ray_ori,
            ray_dir,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph
        )
        ctx.save_for_backward(ray_ori, ray_dir, ray_radiance, ray_density, ray_hit_distance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph)
        ctx.optix_ctx = optix_ctx
        err_backprop_proxy = torch.ones_like(ray_density) # used to abuse autograd
        return ray_radiance, ray_density, ray_hit_distance, hits_count, g_weights, err_backprop_proxy
    
    @staticmethod
    def backward(ctx, ray_radiance_grd, ray_density_grd, ray_hit_distance_grd, ray_hits_count_grd_UNUSED, g_weights_grd_UNUSED, ray_fake_err):
        optix_ctx = ctx.optix_ctx
        ray_ori, ray_dir, ray_radiance, ray_density, ray_hit_distance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph = ctx.saved_variables
        mog_pos_grd, mog_rot_grd, mog_scl_grd, mog_dns_grd, mog_sph_grd, mog_error = _plugin.trace_mog_bwd(
            optix_ctx.cpp_wrapper,
            ray_ori,
            ray_dir,
            ray_radiance,
            ray_density,
            ray_hit_distance,
            mog_pos,
            mog_rot,
            mog_scl,
            mog_dns,
            mog_sph,
            ray_radiance_grd,
            ray_density_grd,
            ray_hit_distance_grd,
            ray_fake_err 
        )
        return None, None, None, mog_pos_grd, mog_rot_grd, mog_scl_grd, mog_dns_grd, mog_sph_grd, mog_error

@torch.cuda.nvtx.range("trace_mog")
def trace_mog(optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph, err_target):
    ray_radiance, ray_density, ray_hit_distance, ray_hits_count, g_weights, err_backprop_proxy = _trace_mog_func.apply(
        optix_ctx,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        mog_pos.contiguous(),
        mog_rot.contiguous(),
        mog_scl.contiguous(),
        mog_dns.contiguous(),
        mog_sph.contiguous(),
        err_target
    )
    return ray_radiance, ray_density, ray_hit_distance, ray_hits_count, g_weights, err_backprop_proxy

@torch.cuda.nvtx.range("trace_mog_grad")
def trace_mog_grad(optix_ctx, ray_ori, ray_dir, ray_radiance, ray_density, ray_hit_distance, mog_pos, mog_rot, mog_scl, mog_dns, mog_sph, ray_radiance_grd, ray_density_grd, ray_hit_distance_grd):
    return _plugin.trace_mog_bwd(
        optix_ctx.cpp_wrapper,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        ray_radiance.contiguous(),
        ray_density.contiguous(),
        ray_hit_distance.contiguous(),
        mog_pos.contiguous(),
        mog_rot.contiguous(),
        mog_scl.contiguous(),
        mog_dns.contiguous(),
        mog_sph.contiguous(),
        ray_radiance_grd.contiguous(),
        ray_density_grd.contiguous(),
        ray_hit_distance_grd.contiguous()
    )

@torch.cuda.nvtx.range("count_mog_hits")
def count_mog_hits(optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns):
    mog_hit_counts = _plugin.count_mog_hits(
        optix_ctx.cpp_wrapper,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        mog_pos.contiguous(),
        mog_rot.contiguous(),
        mog_scl.contiguous(),
        mog_dns.contiguous()
    )
    return mog_hit_counts

def trace_mog_inds(optix_ctx, ray_ori, ray_dir, mog_pos, mog_rot, mog_scl, mog_dns):

    ray_hits = _plugin.trace_mog_inds(
        optix_ctx.cpp_wrapper,
        ray_ori.contiguous(),
        ray_dir.contiguous(),
        mog_pos.contiguous(),
        mog_rot.contiguous(),
        mog_scl.contiguous(),
        mog_dns.contiguous()
    )
    return ray_hits

@torch.cuda.nvtx.range("get_mog_primitives")
def get_mog_primitives(optix_ctx):
    return _plugin.get_mog_primitives(optix_ctx.cpp_wrapper)

@torch.cuda.nvtx.range("create_camera_rays")
def create_camera_rays(image_width: int, image_height: int, tanfovx: float, tanfovy : float, viewmatrix : torch.Tensor):
    return _plugin.create_camera_rays(image_width, image_height, tanfovx, tanfovy, viewmatrix)

#----------------------------------------------------------------------------
#
@torch.cuda.nvtx.range("build_mog_bvh")
def build_mog_bvh(
        optix_ctx,
        mog_pos,
        mog_rot,
        mog_scl,
        rebuild
):
    _plugin.build_mog_bvh(
        optix_ctx.cpp_wrapper,
        mog_pos.view(-1, 3).contiguous(),
        mog_rot.view(-1, 4).contiguous(),
        mog_scl.view(-1, 3).contiguous(),
        rebuild
    )

#----------------------------------------------------------------------------
#
class OptixMogPipeline(IntEnum):
    TRACING_BASELINE = 0
    TRACING_DEFAULT = 1
    TRACING_MLAT = 2
    TRACING_MBOIT = 4

class OptixMogPrimitive(IntEnum):
    TRACING_ICOSAHEDRON = 0
    TRACING_OCTAHEDRON = 1
    TRACING_TETRAHEDRON = 2
    TRACING_DIAMOND = 3,
    TRACING_SPHERE = 4,
    TRACING_CUSTOM = 5,
    TRACING_TRIHEXA = 6,
    TRACING_TRISURFEL = 7

@dataclass
class OptixMogTracingParams:
    pipeline : int=OptixMogPipeline.TRACING_DEFAULT # Tracing algo
    primitive_type : int=OptixMogPrimitive.TRACING_ICOSAHEDRON # enveloping primitive
    hit_mode : int=0            # Intersection distance : 0 = distance to the first hit triangle, 1 = distance to the projection of aussian center on the ray 
    max_hit_per_slab: int=32    # Size of the array of sorted gaussian per-slabs
    max_num_slabs: int=64       # Number of slabs along the diagonal of the scene AABB
    topk_hits: bool=False       # If true do not respawn rays, just keep the first max_hit_per_slab gaussians
    patch_size: int=1           # tile size (best performances achieve with 3 but need COHERENT rays inputs)
    sph_degree: int=3           # spherical harmonics degree
    gaussian_sigma_threshold: float=3.0 # sigma factor to decide the size of the octahedron enveloppe
    min_transmittance: float=0.03       # minimum transmittance at which we stop gathering gaussians
    max_hits_returned : int=64       # total number of hits returned
    use_g_weights: bool=True    # are the gaussian weights used 

    @staticmethod
    def primitive_type_from_str(primitive_type:str):
        if primitive_type == 'icosahedron':
            return OptixMogPrimitive.TRACING_ICOSAHEDRON
        elif primitive_type == 'octahedron':
            return OptixMogPrimitive.TRACING_OCTAHEDRON
        elif primitive_type == 'tetrahedron':
            return OptixMogPrimitive.TRACING_TETRAHEDRON
        elif primitive_type == 'diamond':
            return OptixMogPrimitive.TRACING_DIAMOND
        elif primitive_type == 'sphere':
            return OptixMogPrimitive.TRACING_SPHERE
        elif primitive_type == 'custom':
            return OptixMogPrimitive.TRACING_CUSTOM
        elif primitive_type == 'trihexa':
            return OptixMogPrimitive.TRACING_TRIHEXA
        elif primitive_type == 'trisurfel':
            return OptixMogPrimitive.TRACING_TRISURFEL
        else :
            raise ValueError("Unknown OptixMogTracingParams.primitive_type")
        
    @staticmethod
    def pipeline_from_str(pipeline:str):
        if pipeline == 'baseline':
            return OptixMogPipeline.TRACING_BASELINE
        elif pipeline == 'default':
            return OptixMogPipeline.TRACING_DEFAULT
        elif pipeline == 'mlat':
            return OptixMogPipeline.TRACING_MLAT
        elif pipeline == 'mboit':
            return OptixMogPipeline.TRACING_MBOIT
        else :
            raise ValueError("Unknown OptixMogTracingParams.pipeline")
class OptiXContext:
    def __init__(self,params:OptixMogTracingParams=OptixMogTracingParams()):
        print("Cuda path", torch.utils.cpp_extension.CUDA_HOME)
        torch.zeros(1, device='cuda') # Create a dummy tensor to force cuda context init
        self.cpp_wrapper = _plugin.OptiXStateWrapper(
            os.path.dirname(__file__), 
            torch.utils.cpp_extension.CUDA_HOME,
            int(params.pipeline),
            int(params.primitive_type),
            params.hit_mode,
            params.max_hit_per_slab,
            params.max_num_slabs,
            params.topk_hits,
            params.patch_size,
            params.sph_degree,
            params.gaussian_sigma_threshold,
            params.min_transmittance,
            params.max_hits_returned,
            params.use_g_weights
        )
    
    def set_sph_degree(self, degree:int):
        self.cpp_wrapper.set_sph_degree(degree)

    def set_pipeline(self, pipeline:int):
        self.cpp_wrapper.set_pipeline(pipeline)

