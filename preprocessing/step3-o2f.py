import os
import glob
import cv2
import numpy as np
import pycolmap
from lib.cam_utils import *
from lib.cam_utils import radial_mask
import sys

### ----- load images -----
for i, a in enumerate(sys.argv[1:]):
    if a == "--scene" and i + 2 <= len(sys.argv):
        scene = sys.argv[1:][i + 1].strip("\"'")
    if a == "--method" and i + 2 <= len(sys.argv):
        method = sys.argv[1:][i + 1].strip("\"'")
    if a == "--project" and i + 2 <= len(sys.argv):
        project = sys.argv[1:][i + 1].strip("\"'")

data_dir    = f"/home/youlenda/360/omni/3dgrut/data/{project}/{scene}"
colmap_dir  = f"{data_dir}/sparse/0"
int_rot_dir = "examples/no_lens_guards/42/out/optimized.txt"

omni_dir    = f"{data_dir}/out-{method}/omni_mask"
front_dir   = f"{data_dir}/out-{method}/front"
rear_dir    = f"{data_dir}/out-{method}/rear"

width, height = 2880, 2880
radius_f = 109
radius_r = 107

os.makedirs(front_dir, exist_ok=True)
os.makedirs(rear_dir,  exist_ok=True)

### ----- load intrinsics & rotations -----
with open(int_rot_dir, "r") as f:
    lines = f.readlines()

params_f = eval(lines[0].split('params=')[1].split('(')[0].strip())
params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

rvec_f = eval(lines[2].strip(), {'array': np.array})
rvec_r = eval(lines[3].strip(), {'array': np.array})

rec = pycolmap.Reconstruction()
try:
    rec.read(colmap_dir)
except:
    rec.read(f"/home/youlenda/360/omni/3dgrut/data/party_room/sparse/0")

camera_f = pycolmap.Camera(camera_id=0, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=width, height=height, params=params_f)
camera_r = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=width, height=height, params=params_r)
    
R_front, _ = cv2.Rodrigues(np.array(rvec_f))
R_rear , _ = cv2.Rodrigues(np.array(rvec_r))

### ----- precompute maps -----
dirs_f = get_cam_ray_dirs(camera_f).cpu().numpy()
dirs_r = get_cam_ray_dirs(camera_r).cpu().numpy()

W_fi, H_fi = camera_f.width, camera_f.height
W_eq, H_eq = W_fi * 2, H_fi

r, Rpix = radial_mask(W_fi, H_fi)
mask_f = (r < (Rpix - (radius_f + 0.5)))
mask_r = (r < (Rpix - (radius_r + 0.5)))

xyz_f = (R_front @ dirs_f.T).T
xyz_r = (R_rear  @ dirs_r.T).T

xyz_r = np.clip(xyz_r, -1.0, 1.0)

lat_f, lon_f = xyz_to_latlon(xyz_f)
lon_f = -(lon_f+np.pi)

lat_r, lon_r = xyz_to_latlon(xyz_r)
lon_r = -(lon_r)

# project into equirectangular UV
u_f, v_f = latlon_to_uv(lat_f, lon_f, W_eq, H_eq)
u_r, v_r = latlon_to_uv(lat_r, lon_r, W_eq, H_eq)

# reshape UV to maps
u_f_map = u_f.reshape(H_fi, W_fi)
v_f_map = v_f.reshape(H_fi, W_fi)

u_r_map = u_r.reshape(H_fi, W_fi)
v_r_map = v_r.reshape(H_fi, W_fi)

fisheye_front = np.zeros((H_fi, W_fi, 3), dtype=np.uint8)
fisheye_rear  = np.zeros((H_fi, W_fi, 3), dtype=np.uint8)

omni_paths = sorted(glob.glob(os.path.join(omni_dir, "*.png")))
print(f"Found {len(omni_paths)} omni frames in {omni_dir}")

for p in omni_paths:
    img = cv2.imread(p)

    fisheye_front[mask_f] = img[v_f_map[mask_f], u_f_map[mask_f]]
    fisheye_rear[mask_r]  = img[v_r_map[mask_r], u_r_map[mask_r]]

    base = os.path.basename(p)
    cv2.imwrite(os.path.join(front_dir, base), fisheye_front)
    cv2.imwrite(os.path.join(rear_dir,  base), fisheye_rear)
    print(f"[ok] {base} → front/ & rear/ saved")