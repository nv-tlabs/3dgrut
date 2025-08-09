#!/usr/bin/env python3
import os
import cv2
import numpy as np

# -------- settings (tune these) ---------------------------------------------
img_path = "frame_019.png"

# base circular shrink
shrink_top = 15
shrink_bot = 15

# smooth extra shrink vs. angle FROM BOTTOM (degrees → extra pixels)
deg_knots_bottom = np.array([0, 20, 35, 55], dtype=np.float32)
shave_knots_bottom = np.array([100, 80, 60, 0], dtype=np.float32)

# smooth extra shrink vs. angle FROM LEFT (degrees → extra pixels)
deg_knots_left = np.array([0, 25, 45, 70], dtype=np.float32)
shave_knots_left = np.array([10, 25, 25, 0], dtype=np.float32)

# optional bottom half-circle cutout (e.g., tripod). Set radius=0 to disable.
bottom_half_circle_radius = 400  # px (0 = off)

# optional hard bottom band (set to 0 to disable)
bottom_rect_h = 120  # px
# ---------------------------------------------------------------------------

# load image
img = cv2.imread(img_path)
assert img is not None, f"Cannot read {img_path}"
h, w = img.shape[:2]
cx, cy = w // 2, h // 2
R = min(w, h) / 2.0

# coordinate grids & distances
yy, xx = np.ogrid[:h, :w]
dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

# angles (degrees)
theta = np.degrees(np.arctan2(yy - cy, xx - cx))  # [-180,180]

# --- smooth extra shrink from BOTTOM ---
ang_from_bottom = np.abs((theta - 90.0 + 180.0) % 360.0 - 180.0)
extra_bottom = np.interp(
    np.clip(ang_from_bottom, 0, 180),
    deg_knots_bottom, shave_knots_bottom
) * (yy >= cy)  # only bottom half

# --- smooth extra shrink from LEFT ---
ang_from_left = np.abs((theta - 180.0 + 180.0) % 360.0 - 180.0)
extra_left = np.interp(
    np.clip(ang_from_left, 0, 180),
    deg_knots_left, shave_knots_left
) * (xx <= cx)  # only left half

# combine extras and build allowed radius
extra = extra_bottom + extra_left
R_top = R - shrink_top
R_bot = R - shrink_bot
R_allowed = np.where(yy < cy, R_top, R_bot) - extra
R_allowed = np.clip(R_allowed, 0, R)

# final mask (uint8: 255 keep, 0 ignore)
mask = (dist <= R_allowed).astype(np.uint8) * 255

# optional: small bottom half-circle carve-out
if bottom_half_circle_radius > 0:
    tmp = np.zeros((h, w), np.uint8)
    cv2.circle(tmp, (cx, h - 1), int(bottom_half_circle_radius), 255, -1)
    # keep only the lower half of the circle (near the bottom edge)
    tmp[: (h - 1 - int(bottom_half_circle_radius)), :] = 0
    mask[tmp > 0] = 0

# optional: hard bottom rectangle
if bottom_rect_h > 0:
    cv2.rectangle(mask, (0, h - int(bottom_rect_h)), (w - 1, h - 1), 0, -1)

# output paths
base = os.path.splitext(os.path.basename(img_path))[0]
root = os.path.dirname(img_path) or "."
out_dir = os.path.join(root, "mask")
os.makedirs(out_dir, exist_ok=True)

mask_path           = os.path.join(out_dir, f"{base}_mask.png")
masked_path         = os.path.join(out_dir, f"{base}_masked.png")
overlay_path        = os.path.join(out_dir, f"{base}_masked_overlay.png")
base_masked_path    = os.path.join(out_dir, f"{base}_masked_base.png")
extra_vs_base_path  = os.path.join(out_dir, f"{base}_masked_extra_vs_base.png")

# save mask
cv2.imwrite(mask_path, mask)
print(f"Wrote mask: {mask_path}")

# apply mask with magenta background
masked_img = cv2.bitwise_and(img, img, mask=mask)
masked_img[mask == 0] = (255, 0, 255)
cv2.imwrite(masked_path, masked_img)
print(f"Wrote masked image: {masked_path}")

# soft overlay for quick QA (blend with original)
overlay = cv2.addWeighted(masked_img, 0.7, img, 0.3, 0)
cv2.imwrite(overlay_path, overlay)
print(f"Wrote masked+GT overlay: {overlay_path}")

# --- stats & base-circle comparisons ---
total_px = h * w
kept_px = int(np.count_nonzero(mask))
ignored_px = total_px - kept_px
print(f"Kept:    {kept_px}/{total_px}  ({kept_px/total_px:.2%})")
print(f"Ignored: {ignored_px}/{total_px}  ({ignored_px/total_px:.2%})")

base_r = R - shrink_top  # same as R_top/R_bot when equal
base_circle = dist <= base_r
base_total = int(np.count_nonzero(base_circle))
base_kept = int(np.count_nonzero(base_circle & (mask > 0)))
base_ignored = base_total - base_kept
print(f"Ignored within base circle: {base_ignored}/{base_total} "
      f"({base_ignored/base_total:.2%})")

# overlay using base mask (outside base = magenta)
base_u8 = (base_circle.astype(np.uint8) * 255)
base_masked = cv2.bitwise_and(img, img, mask=base_u8)
base_masked[~base_circle] = (255, 0, 255)
cv2.imwrite(base_masked_path, base_masked)

# show only the extra removed beyond base circle
extra_removed = base_circle & (mask == 0)
extra_overlay = img.copy()
extra_overlay[~base_circle] = (30, 30, 30)
extra_overlay[extra_removed] = (255, 0, 255)
cv2.imwrite(extra_vs_base_path, extra_overlay)

