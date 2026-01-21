import cv2
import numpy as np
import pycolmap  
import torch

## fisheye-to-omni funstions
def get_cam_ray_dirs(camera, mask=False, radius=80): #changed to True!
    x = np.arange(camera.width,  dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)

    if mask:
        cx, cy = camera.width / 2.0, camera.height / 2.0
        R = min(camera.width, camera.height) / 2.0
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        valid = r < (R - (radius+.5))
        pix_coords = np.stack([x[valid], y[valid]], axis=-1)
    else:
        pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
        
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
    ray_dirs  = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)

def xyz_to_latlon(dirs):
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lat = np.arcsin(-y)
    lon = np.arctan2(x, -z)
    return lat, lon

def latlon_to_uv(lat, lon, W, H):
    u = (1 + (  lon/np.pi)) * W/2
    v = (1 - (2*lat/np.pi)) * H/2
    
    return u.astype(np.int32), v.astype(np.int32)

def latlon_to_uv_clip_v(lat, lon, W, H):
    u = (1 + (  lon/np.pi)) * W/2
    v = (1 - (2*lat/np.pi)) * H/2
    
    # u = np.clip(np.round(u), 0, W-1).astype(np.int32)
    v = np.clip(np.round(v), 0, H-1).astype(np.int32)
    
    return u.astype(np.int32), v

def radial_mask(W_fi, H_fi):
    cx, cy = W_fi / 2.0, H_fi / 2.0
    R = min(W_fi, H_fi) / 2.0
    x, y = np.meshgrid(np.arange(W_fi, dtype=np.float32) + 0.5,
                       np.arange(H_fi, dtype=np.float32) + 0.5)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    return r, R

## omni-to-fisheye funstions
def latlon_to_xyz(lat, lon):
    x = np.cos(lat)*np.sin(lon)
    y = np.sin(lat)
    z = -np.cos(lat)*np.cos(lon)
    return np.stack([x,y,z], axis=1)

def uv_to_latlon(uv, W, H):
    u, v = uv[:,0], uv[:,1]
    lat =  np.pi * (0.5 - (v/H))
    lon = -np.pi * (1.0 - (u/W)*2.0)
    return lat, lon
    
def build_maps(xyz_world, mask, R_cam, camera):
    W_eq, H_eq = camera.width* 2, camera.height

    map_x = np.full((H_eq, W_eq), 1, np.float32)
    map_y = np.full((H_eq, W_eq), -1, np.float32)

    idx = mask.ravel()
    xyz = xyz_world[idx]
    xy  = camera.img_from_cam((R_cam.T @ xyz.T).T)

    map_x.ravel()[idx] = xy[:, 0]
    map_y.ravel()[idx] = xy[:, 1]
    
    return map_x, map_y


## visualization - projecting inliers on loaded images
def draw_points(img, pts, color, r=2):
    for u, v in pts.astype(int):
        cv2.circle(img, (u, v), r, color, -1)


## optimization functions
def residuals_rot_cam(p0, xy_f, xy_r, xyz_ref_f, xyz_ref_r):
    ## camera
    res = 3840
    params_f = p0[ : 6]
    params_r = p0[6:12]
    
    params_f = np.insert(params_f, 2, [res/2, res/2])
    params_r = np.insert(params_r, 2, [res/2, res/2])

    cam_f = pycolmap.Camera(camera_id=0, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=res, height=res, params=params_f)
    cam_r = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=res, height=res, params=params_r)

    ip_coords_f = cam_f.cam_from_img(xy_f)
    ip_coords_r = cam_r.cam_from_img(xy_r)

    ip_coords_f = np.concatenate([ip_coords_f, np.ones_like(ip_coords_f[:, :1])], axis=-1)
    ip_coords_r = np.concatenate([ip_coords_r, np.ones_like(ip_coords_r[:, :1])], axis=-1)
    
    ray_dirs_f = -(ip_coords_f / np.linalg.norm(ip_coords_f, axis=-1, keepdims=True))
    ray_dirs_r = -(ip_coords_r / np.linalg.norm(ip_coords_r, axis=-1, keepdims=True))
    
    ## rotation
    rvec_f = p0[12:15]
    rvec_r = p0[15:  ]
    
    R_front, _ = cv2.Rodrigues(rvec_f)
    R_rear , _ = cv2.Rodrigues(rvec_r)
    
    xyz_rot_f = (R_front @ ray_dirs_f.T).T
    xyz_rot_r = (R_rear  @ ray_dirs_r.T).T
    
    return np.concatenate([(xyz_rot_f - xyz_ref_f).ravel(), (xyz_rot_r - xyz_ref_r).ravel()])

def residuals_rot_cam_f(p0, xy_f, xyz_ref_f):
    ## camera
    params_f = p0[ :6]
    
    params_f = np.insert(params_f, 2, [1440, 1440])

    cam_f = pycolmap.Camera(camera_id=0, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=2880, height=2880, params=params_f)

    ip_coords_f = cam_f.cam_from_img(xy_f)

    ip_coords_f = np.concatenate([ip_coords_f, np.ones_like(ip_coords_f[:, :1])], axis=-1)
    
    ray_dirs_f = -(ip_coords_f / np.linalg.norm(ip_coords_f, axis=-1, keepdims=True))
    
    ## rotation
    rvec_f = p0[6:]
    
    R_front, _ = cv2.Rodrigues(rvec_f)
    
    xyz_rot_f = (R_front @ ray_dirs_f.T).T
    
    return (xyz_rot_f - xyz_ref_f).ravel()

def residuals_cam_f(p, xy_f, xyz_ref_f):
    cam_f = pycolmap.Camera(model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=2880, height=2880, params=p)

    ip_coords_f = cam_f.cam_from_img(xy_f)
    ip_coords_f = np.concatenate([ip_coords_f, np.ones_like(ip_coords_f[:, :1])], axis=-1)
    ray_dirs_f  = -(ip_coords_f / np.linalg.norm(ip_coords_f, axis=-1, keepdims=True))

    return (ray_dirs_f - xyz_ref_f).ravel()

def residuals_rot(rvec_f, pts3d_map_f, xyz_ref_f):
    
    R_front, _ = cv2.Rodrigues(rvec_f)
    xyz_f = (R_front @ pts3d_map_f.T).T
    return (xyz_f - xyz_ref_f).ravel()

def residuals_cams(p0, xy_f, xy_r, xyz_ref_f, xyz_ref_r):
    ## camera
    params_f = p0[ : 6]
    params_r = p0[6:12]
    
    params_f = np.insert(params_f, 2, [1920, 1920])
    params_r = np.insert(params_r, 2, [1920, 1920])

    cam_f = pycolmap.Camera(camera_id=0, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=3840, height=3840, params=params_f)
    cam_r = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=3840, height=3840, params=params_r)

    ip_coords_f = cam_f.cam_from_img(xy_f)
    ip_coords_r = cam_r.cam_from_img(xy_r)

    ip_coords_f = np.concatenate([ip_coords_f, np.ones_like(ip_coords_f[:, :1])], axis=-1)
    ip_coords_r = np.concatenate([ip_coords_r, np.ones_like(ip_coords_r[:, :1])], axis=-1)
    
    ray_dirs_f = -(ip_coords_f / np.linalg.norm(ip_coords_f, axis=-1, keepdims=True))
    ray_dirs_r = -(ip_coords_r / np.linalg.norm(ip_coords_r, axis=-1, keepdims=True))

    return np.concatenate([(ray_dirs_f - xyz_ref_f).ravel(), (ray_dirs_r - xyz_ref_r).ravel()])


## rotation
def solve_rotation(A, B):
    Hmat = A.T @ B
    U, _, Vt = np.linalg.svd(Hmat)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    return R


## synthetic fisheye
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def look_at_camZ(center_dir, world_up=np.array([0,1,0], dtype=np.float64)):
    """Return world_from_cam rotation with cam +Z aligned to center_dir."""
    z = normalize(center_dir.reshape(1,3))[0]
    up = world_up.astype(np.float64)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1,0,0], dtype=np.float64)
    x = normalize(np.cross(up, z).reshape(1,3))[0]
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return R


# --- load all R_synthetic from one CSV into a dict: {filename -> 3x3 R} ---
import csv

def load_R_lookup(csv_path):
    R_map = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        first = next(rdr, None)
        if first and len(first) >= 10:
            fname = first[0].strip()
            R = np.array(list(map(float, first[1:10]))).reshape(3,3)
            R_map[fname] = R
        for row in rdr:
            fname = row[0].strip()
            R = np.array(list(map(float, row[1:10]))).reshape(3,3)
            R_map[fname] = R
    return R_map