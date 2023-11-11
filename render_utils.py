import numpy as np
import torch


from utils import to_np, quaternion_to_so3
from geometry import safe_normalize

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

def evaluate_rays(dense_hit_gIds, rays_o, rays_d, gpos, grot, gscl, gdns, gsh, sph_deg):
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

    ## Mask out only the used elements
    dense_hit_mask = (dense_hit_gIds != -1)
    dense_hit_mask3 = dense_hit_mask[...,None].expand(N,D,3)
    hit_gIds = dense_hit_gIds[dense_hit_mask]
    
    ## Shapes

    ## Gather all of the values associated with each hit

    # Gather ray source/dir for each hit
    hit_rays_o_dense = rays_o[:,None,:].expand(N,D,3)
    hit_rays_o = hit_rays_o_dense[dense_hit_mask3].reshape(-1,3)
    hit_rays_d_dense = rays_d[:,None,:].expand(N,D,3)
    hit_rays_d = hit_rays_d_dense[dense_hit_mask3].reshape(-1,3)

    # Gather gaussian params
    hit_gpos = gpos[hit_gIds,:]
    hit_grot = grot[hit_gIds,:]
    hit_gscl = gscl[hit_gIds,:]
    hit_gdns = gdns[hit_gIds,:]
    hit_gsh  =  gsh[hit_gIds,:]

    # Evaluate all of the individual Gaussians
    hit_grad, hit_galpha, hit_gdist = evaluate_gaussians(hit_rays_o, hit_rays_d, hit_gpos, hit_grot, hit_gscl, hit_gdns, hit_gsh, sph_deg)

    ## Broadcast the results back to being dense
    dense_grad = torch.zeros((N,D,3), dtype=dtype, device=device)
    dense_grad[dense_hit_mask3] = hit_grad.flatten()

    dense_alphas = torch.zeros((N,D), dtype=dtype, device=device)
    dense_alphas[dense_hit_mask] = hit_galpha.flatten()

    dense_gdist = torch.zeros((N,D), dtype=dtype, device=device)
    dense_gdist[dense_hit_mask] = hit_gdist.flatten()

    ## Compute transmittance along the ray
    dense_unalpha = torch.cat((torch.ones_like(dense_alphas[:,0:1]), 1. - dense_alphas), dim=-1)
    dense_transmit_prod = torch.cumprod(dense_unalpha, dim=-1)
    dense_transmit = dense_transmit_prod[:,:-1]
    ray_opacity = 1.0 - dense_transmit_prod[:,-1]

    ## Add up weighted contributions from each hit
    dense_weight = dense_transmit * dense_alphas
    ray_rad = torch.sum(dense_weight[...,None] * dense_grad, dim=-2)
    ray_dist = torch.sum(dense_weight*dense_gdist, dim=-1) # TODO better to use max dist?
    ray_ohit = torch.sum(dense_hit_mask, dim=-1)
    
    ## Reshape to original size
    ray_rad = ray_rad.reshape(*orig_shape,3)
    ray_opacity = ray_opacity.reshape(*orig_shape,1)
    ray_dist = ray_dist.reshape(*orig_shape,1)
    ray_ohit = ray_ohit.reshape(*orig_shape,1)
    
    return ray_rad, ray_opacity, ray_ohit, ray_dist

def evaluate_gaussians(rays_o, rays_d, gpos, grot, gscl, gdns, gsh, sph_deg):
    """
    Evaluate the response of gaussians. All inputs are [N_gaussian,...]
    """

    grotMat = quaternion_to_so3(grot)
    giscl = 1. / gscl

    gdir = safe_normalize(gpos - rays_o) # this sign is weird... for consistency?
    sh_dim = (sph_deg+1) ** 2
    gsh_reshape = gsh.reshape(gsh.shape[0], sh_dim, 3)
    grad = eval_sh(sph_deg, gsh_reshape, gdir) + 0.5

    gposc = rays_o - gpos
    gdist = torch.sqrt(torch.linalg.vecdot(gposc, rays_d, dim=-1)[:,None])
    gposcr = torch.bmm(gposc[:,None,:],grotMat)[:,0,:]
    gro = giscl * gposcr
    rayDirR = torch.bmm(rays_d[:,None,:],grotMat)[:,0,:]
    grdu = giscl * rayDirR
    grd = safe_normalize(grdu)
    gcrod = torch.linalg.cross(grd, gro, dim=-1)
    grayDir = torch.linalg.vecdot(gcrod, gcrod, dim=-1)
    gres = torch.exp(-0.5 * grayDir)
    galpha = gres[:,None] * gdns
    
    return grad, galpha, gdist
