import numpy as np

import torch
from torch.utils.checkpoint import checkpoint

from models.packed_ops_modules import packed_cumprod, packed_sum

from utils.misc import to_np, quaternion_to_so3
from models.geometry import safe_normalize

## NOTE: SPH code from gaussian-splatting, from plenoctree, from ???
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-2] >= coeff

    result = C0 * sh[..., 0, :]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]


        result = (result -
              C1 * y * sh[..., 1, :] +
              C1 * z * sh[..., 2, :] -
              C1 * x * sh[..., 3, :])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                  C2[0] * xy * sh[..., 4, :] +
                  C2[1] * yz * sh[..., 5, :] +
                  C2[2] * (2.0 * zz - xx - yy) * sh[..., 6, :] +
                  C2[3] * xz * sh[..., 7, :] +
                  C2[4] * (xx - yy) * sh[..., 8, :])

            if deg > 2:
                result = (result +
                      C3[0] * y * (3 * xx - yy) * sh[..., 9, :] +
                      C3[1] * xy * z * sh[..., 10, :] +
                      C3[2] * y * (4 * zz - xx - yy)* sh[..., 11, :] +
                      C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12, :] +
                      C3[4] * x * (4 * zz - xx - yy) * sh[..., 13, :] +
                      C3[5] * z * (xx - yy) * sh[..., 14, :] +
                      C3[6] * x * (xx - 3 * yy) * sh[..., 15, :])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16, :] +
                          C4[1] * yz * (3 * xx - yy) * sh[..., 17, :] +
                          C4[2] * xy * (7 * zz - 1) * sh[..., 18, :] +
                          C4[3] * yz * (7 * zz - 3) * sh[..., 19, :] +
                          C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20, :] +
                          C4[5] * xz * (7 * zz - 3) * sh[..., 21, :] +
                          C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22, :] +
                          C4[7] * xz * (xx - 3 * yy) * sh[..., 23, :] +
                          C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24, :])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def evaluate_rays(dense_hit_gIds, rays_o, rays_d, gpos, grot, gscl, gdns, gsh, err_targets, sph_deg, settings: dict):
    """
    """
    device = rays_o.device
    dtype = rays_o.dtype

    G = gpos.shape[0]

    # Flatten out to N parallel evaluations
    D = dense_hit_gIds.shape[-1] # max number of hits per ray
    orig_shape = rays_o.shape[:-1]
    rays_o = rays_o.reshape(-1,3)
    N = rays_o.shape[0] # total number of rays we are evaluating
    rays_d = rays_d.reshape(N,3)
    dense_hit_gIds = dense_hit_gIds.reshape(N,D) 

    # split to chunks
    i_c = 0
    ray_rad_out = []
    ray_opacity_out = []
    ray_dist_out = []
    ray_ohit_out = []
    err_backprop_proxy_out = []

    # Evaluate in checkpoint'd chunks to reduce peak memory usage
    # (note that this compes at a cost as it becomes python/CPU-heavy)

    g_weight_accum = torch.zeros_like(gdns)
    g_weight_accum.requires_grad = False

    while i_c < N:
        C = min(settings["chunk_size"], N-i_c)

        ray_rad, ray_opacity, ray_dist, ray_ohit, err_backprop_proxy, g_weight_accum = checkpoint(
                eval_ray_chunk_tup, 
                (i_c, C, D, dense_hit_gIds, rays_o, rays_d, gpos, grot, gscl, gdns, gsh, sph_deg, err_targets, g_weight_accum, settings),
                use_reentrant=False
            )

        ray_rad_out.append(ray_rad)
        ray_opacity_out.append(ray_opacity)
        ray_dist_out.append(ray_dist)
        ray_ohit_out.append(ray_ohit)
        err_backprop_proxy_out.append(err_backprop_proxy)

        i_c += C

    ray_rad_out = torch.cat(ray_rad_out, dim=0)
    ray_opacity_out = torch.cat(ray_opacity_out, dim=0)
    ray_dist_out = torch.cat(ray_dist_out, dim=0)
    ray_ohit_out = torch.cat(ray_ohit_out, dim=0)
    err_backprop_proxy_out = torch.cat(err_backprop_proxy_out, dim=0)
    

    ray_rad_out = ray_rad_out.reshape(*orig_shape,3)
    ray_opacity_out = ray_opacity_out.reshape(*orig_shape,1)
    ray_dist_out = ray_dist_out.reshape(*orig_shape,1)
    ray_ohit_out = ray_ohit_out.reshape(*orig_shape,1)

    err_backprop_proxy_out = err_backprop_proxy_out.reshape(*orig_shape,1)
    
    return ray_rad_out, ray_opacity_out, ray_ohit_out, ray_dist_out, g_weight_accum, err_backprop_proxy_out

def eval_ray_chunk_tup(tup):
    return eval_ray_chunk(*tup)

def eval_ray_chunk(i_c, C, D, dense_hit_gIds, rays_o, rays_d, gpos, grot, gscl, gdns, gsh, sph_deg, err_targets, g_weight_accum, settings: dict):
    
    device = rays_o.device
    dtype = rays_o.dtype
        
    hit_gId_chunk = dense_hit_gIds[i_c:(i_c+C),:]

    ## Mask out only the used elements
    hit_mask = (hit_gId_chunk != -1)
    hit_gIds = hit_gId_chunk[hit_mask]

    # start and end indices per-ray of the hits
    hit_count_ray = torch.sum(hit_mask, dim=-1, dtype=torch.int32) # [N]
    hitrange_end = torch.cumsum(hit_count_ray, dim=0, dtype=torch.int32) # [N]
    hitrange_start = torch.zeros_like(hitrange_end)
    hitrange_start[1:] = hitrange_end[:-1]
    hitrange = torch.stack((hitrange_start, hit_count_ray), dim=-1)
    ray_has_a_hit = hit_count_ray > 0
    last_hit_inds = hitrange_end[ray_has_a_hit] - 1 # only for rays that have a hit
    
    ## Gather all of the values associated with each hit
    hit_ray_inds = torch.arange(C, device=device)[:,None].expand(C,D)[hit_mask] + i_c
    hit_rays_o = rays_o[hit_ray_inds,:]
    hit_rays_d = rays_d[hit_ray_inds,:]

    # Gather gaussian params
    hit_gpos = gpos[hit_gIds,:]
    hit_grot = grot[hit_gIds,:]
    hit_gscl = gscl[hit_gIds,:]
    hit_gdns = gdns[hit_gIds,:]
    hit_gsh  =  gsh[hit_gIds,:]
    hit_err_targets = err_targets[hit_gIds,:]

    # Evaluate all of the individual Gaussians
    hit_grad, hit_galpha, hit_gdist = evaluate_gaussians(hit_rays_o, hit_rays_d, hit_gpos, hit_grot, hit_gscl, hit_gdns, hit_gsh, sph_deg, settings)

    unalpha = 1.0 - hit_galpha
    hit_transmit = packed_cumprod(unalpha, hitrange, True, False)
    ray_final_transmit = torch.ones((C,1), dtype=dtype, device=device)
    ray_final_transmit[ray_has_a_hit,:] = hit_transmit[last_hit_inds] * unalpha[last_hit_inds]
    ray_opacity = 1. - ray_final_transmit

    hit_weight = hit_transmit * hit_galpha
    ray_rad = packed_sum(hit_weight * hit_grad, hitrange)
    ray_dist = packed_sum(hit_weight * hit_gdist, hitrange) # TODO better to use max dist?
    ray_err_backprop_proxy = packed_sum(hit_weight * hit_err_targets, hitrange)
    ray_ohit = hit_count_ray[:,None]

    with torch.no_grad():
        g_weight_accum = g_weight_accum.scatter_add_(dim=0, index=hit_gIds[:,None].to(dtype=torch.int64), src=hit_weight)
    
    return ray_rad, ray_opacity, ray_dist, ray_ohit, ray_err_backprop_proxy, g_weight_accum


def evaluate_gaussians(rays_o, rays_d, gpos, grot, gscl, gdns, gsh, sph_deg, settings: dict):
    """
    Evaluate the response of Gaussians. All inputs are [N_gaussian,...]
    """

    grotMat = quaternion_to_so3(grot)
    giscl = 1.0 / gscl

    sh_dim = (sph_deg + 1) ** 2
    gsh_reshape = gsh.reshape(gsh.shape[0], sh_dim, 3)
    grad = eval_sh(sph_deg, gsh_reshape, safe_normalize(gpos - rays_o)) + 0.5  # this sign is weird... for consistency?)

    if settings["clamp_rad"]:
        clamp_rad_min_bound = settings["clamp_rad_min_bound"]
        grad = torch.where(
            grad > clamp_rad_min_bound, grad, torch.exp(grad - clamp_rad_min_bound) * clamp_rad_min_bound
        )

    match settings["gaussian_evaluation"]:
        case "geometric":
            gposc = rays_o - gpos
            gposcr = torch.bmm(gposc[:, None, :], grotMat)[:, 0, :]
            gro = giscl * gposcr
            rayDirR = torch.bmm(rays_d[:, None, :], grotMat)[:, 0, :]
            grdu = giscl * rayDirR
            grd = safe_normalize(grdu)
            gcrod = torch.linalg.cross(grd, gro, dim=-1)
            grayDir = torch.linalg.vecdot(gcrod, gcrod, dim=-1)
            gres = torch.exp(-0.5 * grayDir)
            galpha = gres[:, None] * gdns

            grds = gscl * grd * torch.linalg.vecdot(grd, -1 * gro, dim=-1)[:, None]
            gdist = torch.sqrt(torch.linalg.vecdot(grds, grds, dim=-1)[:, None])

        case "analytic":
            # remap some names
            R = grotMat

            inv_scale = giscl

            μ = gpos
            x = rays_o
            d = rays_d

            # Covariances are parameterized by Σ = R @ S @ S^T @ R^T
            sigma_inv = torch.einsum(
                "bij,bj,bj,bkj->bik", R, inv_scale, inv_scale, R
            )  # inv(Σ) = R @ inv(S^T) @ inv(S) @ R^T

            # Analytical distance computation as the *maximum* Gaussian response along the ray
            # Derived as the distance t along the ray that maximizes the Gaussian exponent s(t) = ((x+t*d) - μ)' * inv(Σ) * ((x+t*d) - μ)
            # ∇_t s = 2x'*inv(Σ)*d + 2d'*inv(Σ)*d*t - 2*μ*inv(Σ)*d ⊜ 0
            # → t = (μ - x)'*inv(Σ)*d / d'*inv(Σ)*d = (μ - x)'*ƌ / d'*ƌ with ƌ = inv(Σ)*d
            ƌ = torch.einsum("bij,bj->bi", sigma_inv, d)  # ƌ = inv(Σ)*d
            t = torch.einsum("bi,bi->b", μ - x, ƌ) / torch.einsum("bi,bi->b", d, ƌ)  # (μ - x)'*ƌ / d'*ƌ

            # Vector to maximum point on ray relative to Gaussian center
            m = (x + torch.einsum("b,bi->bi", t, d)) - μ  # m = (x + t*d) - μ

            # Evaluate Gaussian on maximum points
            p = torch.einsum("bj,bij,bi->b", m, sigma_inv, m)  # m' * inv(Σ) * m
            gaussian_response = torch.exp(-0.5 * p)

            # outputs
            galpha = gaussian_response[:, None] * gdns
            gdist = t[:, None]

    return grad, galpha, gdist
