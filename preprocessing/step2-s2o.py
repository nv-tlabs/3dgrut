import os, glob, csv
import cv2
import numpy as np
import pycolmap
from lib.cam_utils import *
from lib.cam_utils import radial_mask
import sys

### ----- config -----
for i, a in enumerate(sys.argv[1:]):
    if a == "--scene" and i + 2 <= len(sys.argv):
        scene = sys.argv[1:][i + 1].strip("\"'")
    if a == "--method" and i + 2 <= len(sys.argv):
        method = sys.argv[1:][i + 1].strip("\"'")
    if a == "--project" and i + 2 <= len(sys.argv):
        project = sys.argv[1:][i + 1].strip("\"'")
        
data_dir     = f"/home/youlenda/360/omni/3dgrut/data/{project}/{scene}"
fisheye_dir  = f"{data_dir}/out-{method}/fisheye_sy_mask"
R_table_path = f"{data_dir}/pre_masking_6/synth_R_list.csv"
out_eq_dir   = f"{data_dir}/out-{method}/omni_mask"
# out_ov_dir   = f"{data_dir}/out-dino/omni_mask_overlay"
# ref_dir      = f"{data_dir}/omni"

colmap_dir   = f"{data_dir}/sparse/0"
int_rot_dir  = "optimized.txt"
radius_r     = 107
camera_size  = 2880

overlay      = False
os.makedirs(out_eq_dir, exist_ok=True)
# os.makedirs(out_ov_dir, exist_ok=True)

### ----- load camera -----
with open(int_rot_dir, "r") as f:
    lines = f.readlines()
params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

rec = pycolmap.Reconstruction()
try:
    rec.read(colmap_dir)
except:
    rec.read(f"/home/youlenda/360/omni/3dgrut/data/party_room/sparse/0")

camera_r = pycolmap.Camera(
    camera_id=1,
    model=pycolmap.CameraModelId.OPENCV_FISHEYE,
    width=camera_size, height=camera_size,
    params=params_r
)

# pano size (2:1 equirect)
W_fi, H_fi = camera_r.width, camera_r.height
W_eq, H_eq = camera_r.width * 2, camera_r.height

### ----- precompute: pano grid -> WORLD rays (constant) -----
u = np.arange(W_eq, dtype=np.float32) + 0.5
v = np.arange(H_eq, dtype=np.float32) + 0.5
uu, vv = np.meshgrid(u, v, indexing="xy")
uv = np.stack([uu.ravel(), vv.ravel()], axis=1)         # (H_eq*W_eq, 2)

lat, lon = uv_to_latlon(uv, W_eq, H_eq)                 # radians
dirs_world = latlon_to_xyz(-lat, -lon)                  # (N,3) match your sign convention

### ----- precompute: lens mask (constant) -----
r, Rpix = radial_mask(W_fi, H_fi)
lens = (r < (Rpix - (radius_r + 0.5)))                  # valid fisheye pixels

### ----- process folder -----
fisheye_paths = sorted(glob.glob(os.path.join(fisheye_dir, "*.png")))
print(f"Found {len(fisheye_paths)} fisheye frames")

R_lookup = load_R_lookup(R_table_path)

for fe_path in fisheye_paths:
    stem = os.path.splitext(os.path.basename(fe_path))[0]
    img_r = cv2.imread(fe_path)

    img_r = cv2.resize(img_r, (W_fi, H_fi), interpolation=cv2.INTER_LINEAR)
    
    fe_name = os.path.basename(fe_path)
    R_synthetic = R_lookup[fe_name]

    # WORLD -> CAM: if you used world = R @ cam during synth, invert here: cam = R^T @ world
    dirs_cam = (R_synthetic.T @ dirs_world.T).T                     # (N,3)

    # only forward rays (z>0)
    forward = dirs_cam[:, 2] > 1e-8
    pts_cam = dirs_cam[forward]

    # project via COLMAP fisheye intrinsics
    px_py = camera_r.img_from_cam(pts_cam)                          # (M,2)

    # build equirect remap (source=fisheye, dest=equirect)
    Mx = np.full((H_eq, W_eq), -1, dtype=np.float32)
    My = np.full((H_eq, W_eq), -1, dtype=np.float32)
    My.ravel()[forward] = px_py[:, 1].astype(np.float32)
    Mx.ravel()[forward] = px_py[:, 0].astype(np.float32)

    # mask remap entries that land outside fisheye circle (becomes black)
    valid = (Mx >= 0) & (My >= 0)
    if np.any(valid):
        xi = np.clip(Mx[valid].astype(np.int32), 0, W_fi - 1)
        yi = np.clip(My[valid].astype(np.int32), 0, H_fi - 1)
        ok = np.zeros_like(valid)
        ok[valid] = lens[yi, xi]
        Mx[~ok] = -1
        My[~ok] = -1

    # remap: fisheye -> equirect; black where no source
    eq_img = cv2.remap(
        img_r, Mx, My,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    out_eq_path = os.path.join(out_eq_dir, f"{fe_name}")
    cv2.imwrite(out_eq_path, eq_img)
    # if overlay:
    #     ref_path = os.path.join(ref_dir, f"{fe_name}")
    #     img_ref = cv2.imread(ref_path)
    #     overlay = cv2.addWeighted(img_ref, 0.5, eq_img, 0.5, 0)
    #     cv2.imwrite(os.path.join(out_ov_dir, f"{fe_name}"), overlay)

    print(f"{fe_name} → {out_eq_path}")
    # assert False