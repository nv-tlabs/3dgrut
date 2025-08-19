import os
import cv2
import torch
import numpy as np
from torchvision.io import read_image
from torchmetrics import PeakSignalNoiseRatio
from tqdm import tqdm
from PIL import Image

def create_circular_mask(image_shape: tuple, border_offset: float = 25.0, device = None) -> torch.Tensor:
    batch, height, width, _ = image_shape
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing='ij'
    )

    cx, cy = width / 2.0, height / 2.0
    R = min(width, height) / 2.0
    
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    circular_mask = (r < (R - border_offset)).float()
    
    circular_mask = circular_mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    circular_mask = circular_mask.repeat(batch, 1, 1, 1)      # [B, H, W, 1]
    return circular_mask


def calculate_psnr_with_mask(render_dir, gt_dir, mask_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    psnr_metric = PeakSignalNoiseRatio(data_range=1).to(device)

    metric_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    metric_mask = torch.from_numpy(metric_mask).float() / 255.0

    metric_mask = metric_mask.unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3).to(device)
    
    image_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])

    psnr_scores = []
    
    save_error_maps = False
    if save_error_maps:
        error_dir = os.path.join(os.path.dirname(render_dir), "error_maps")
        os.makedirs(error_dir, exist_ok=True)
    for filename in tqdm(image_files, desc="Processing Images"):
        render_path = os.path.join(render_dir, filename)
        gt_path = os.path.join(gt_dir, filename)

        re_img_tensor = read_image(render_path).to(device)
        gt_img_tensor = read_image(gt_path).to(device)

        re_img_tensor = re_img_tensor.float() / 255.0
        gt_img_tensor = gt_img_tensor.float() / 255.0

        re_for_error = re_img_tensor.permute(1, 2, 0)
        gt_for_error = gt_img_tensor.permute(1, 2, 0)

        if save_error_maps:
            error_map_np = torch.abs(gt_for_error - re_for_error).cpu().numpy()
            error_rgb = (error_map_np * 255).astype(np.uint8)
            error_path = os.path.join(error_dir, filename)
            Image.fromarray(error_rgb).save(error_path)

        re_img_tensor = re_img_tensor.permute(1, 2, 0).unsqueeze(0)
        gt_img_tensor = gt_img_tensor.permute(1, 2, 0).unsqueeze(0)
        
        circular_mask = create_circular_mask(re_img_tensor.shape, device=re_img_tensor.device).repeat(1, 1, 1, 3)

        re_circle = re_img_tensor * circular_mask
        gt_circle = gt_img_tensor * circular_mask
        
        re_masked = re_circle * metric_mask
        gt_masked = gt_circle * metric_mask
        
        psnr_value = psnr_metric(re_masked, gt_masked).item()
        psnr_scores.append((filename, psnr_value))

    suffix = "wo" if "wo" in render_dir else "w"
    txt_path = os.path.join(os.path.dirname(render_dir), f"psnr_{suffix}.txt")
    with open(txt_path, "w") as f:
        for name, val in psnr_scores:
            f.write(f"{name}: {val:.4f}\n")
    mean_psnr = np.mean([v for _, v in psnr_scores])
    print("\n--- Results ---")
    print(f"✅ Mean PSNR: {mean_psnr:.4f}")
    print(f"📄 Per-image scores saved to {txt_path}")
    print("----------------")

if __name__ == "__main__":
    # # re_dir = "runs/flat_yalda/wo_distractors-1908_111705/ours_7000/renders"
    # gt_dir = "runs/flat_yalda/wo_distractors-1908_111705/ours_30000/gt"
    # re_dir = "runs/flat_yalda/w_distractors-1808_225752/ours_7000/renders"
    
    re_dir = "runs/flat_ipek/wo_distractors-1808_192346/ours_30000/renders"
    gt_dir = "runs/flat_ipek/wo_distractors-1808_192346/ours_30000/gt"
    # re_dir = "runs/flat_ipek/w_distractors-1808_225743/ours_7000/renders"
    mask_dir = "mask.png"
    
    calculate_psnr_with_mask(re_dir, gt_dir, mask_dir)