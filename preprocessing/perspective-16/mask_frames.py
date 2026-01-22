import os
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images with YOLO and SAM for person detection and masking')
    parser.add_argument('input_dir', help='Base input directory containing camera folders')
    parser.add_argument('output_dir', help='Base output directory for masks and masked images')
    parser.add_argument('--yolo-model', default='yolov8s.pt', help='Path to YOLO model (default: yolov8s.pt)')
    parser.add_argument('--sam-checkpoint', default='/home/youlenda/360/segment-anything/sam_vit_h_4b8939.pth', help='Path to SAM checkpoint (default: sam_vit_h_4b8939.pth)')
    parser.add_argument('--sam-model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type (default: vit_h)')
    parser.add_argument('--device', default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth image (default: 30)')
    
    args = parser.parse_args()
    
    # Use provided arguments
    base_input_dir = args.input_dir
    output_base_dir = args.output_dir
    yolo_model_path = args.yolo_model
    sam_checkpoint = args.sam_checkpoint
    model_type = args.sam_model_type
    device = args.device
    step = args.step

    # Create output directory structure
    os.makedirs(output_base_dir, exist_ok=True)

    # Load models
    print("Loading YOLO model...")
    yolo = YOLO(yolo_model_path)
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Process each camera folder
    camera_folders = [f for f in os.listdir(base_input_dir) if f.startswith('pano_camera')]
    camera_folders.sort()  # Sort to ensure consistent ordering

    for camera_folder in camera_folders:
        print(f"Processing {camera_folder}...")
        
        camera_input_dir = os.path.join(base_input_dir, camera_folder)
        camera_output_dir = os.path.join(output_base_dir, camera_folder)
        camera_mask_dir = os.path.join(camera_output_dir, "masks")
        camera_masked_dir = os.path.join(camera_output_dir, "masked_images")
        
        # Create output directories for this camera
        os.makedirs(camera_output_dir, exist_ok=True)
        os.makedirs(camera_mask_dir, exist_ok=True)
        os.makedirs(camera_masked_dir, exist_ok=True)
        
        # Get all image files and sort them
        image_files = [f for f in os.listdir(camera_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        # Process every Nth image (indices 0, N, 2N, 3N, ...)
        selected_files = image_files[::step]  # This takes every Nth item starting from index 0
        
        print(f"  Found {len(image_files)} images, processing {len(selected_files)} images (every {step}th)")
        
        for i, fname in enumerate(selected_files):
            print(f"  Processing {fname} ({i+1}/{len(selected_files)})")
            
            fpath = os.path.join(camera_input_dir, fname)
            image = cv2.imread(fpath)
            
            if image is None:
                print(f"  Warning: Could not load {fpath}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # YOLO detection
            results = yolo(image_rgb, stream=True, conf=0.35, iou=0.6, classes=[0], agnostic_nms=False, max_det=20)

            all_boxes = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for j in range(len(result.boxes)):
                    cls  = int(result.boxes.cls[j])
                    conf = float(result.boxes.conf[j])
                    if cls != 0 or conf < 0.35:      # extra guard
                        continue
                    box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                    all_boxes.append(box)

            if len(all_boxes) == 0:
                print(f"  No persons detected in {fname}")
                # Still save the original image and an empty mask
                mask_path = os.path.join(camera_mask_dir, fname)
                empty_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                cv2.imwrite(mask_path, empty_mask)
                
                # # Save original image as masked image
                # out_path = os.path.join(camera_masked_dir, fname)
                # cv2.imwrite(out_path, image)
                continue
            # breakpoint()
            # SAM segmentation
            predictor.set_image(image_rgb)
            input_boxes = torch.tensor(all_boxes, device=device)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Apply all person masks
            final_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
            
            # --- Center of mass for this cubemap face (pixel coords) ---
            # Option A: OpenCV moments (robust, no extra deps)
            M = cv2.moments(final_mask, binaryImage=True)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]  # x center
                cy = M["m01"] / M["m00"]  # y center
                center_of_mass = (float(cx), float(cy))
            else:
                center_of_mass = (np.nan, np.nan)  # empty mask fallback

            # (optional) log or persist for later merging across faces
            centers_csv = os.path.join(camera_output_dir, "centers.csv")
            header_needed = (not os.path.exists(centers_csv)) or (i == 0)
            with open(centers_csv, "a") as f:
                if header_needed:
                    f.write("filename,cx,cy\n")
                f.write(f"{fname},{center_of_mass[0]:.3f},{center_of_mass[1]:.3f}\n")
            
            # # --- end center of mass ---
            # vis_dir = os.path.join(camera_output_dir, "mask_vis")
            # os.makedirs(vis_dir, exist_ok=True)

            # # visualize on the binary mask
            # vis_mask = (final_mask * 255).astype(np.uint8)
            # vis_mask = cv2.cvtColor(vis_mask, cv2.COLOR_GRAY2BGR)

            # if np.isfinite(center_of_mass[0]) and np.isfinite(center_of_mass[1]):
            #     px = int(round(center_of_mass[0]))
            #     py = int(round(center_of_mass[1]))
            #     cv2.circle(vis_mask, (px, py), 5, (0, 0, 255), -1)               # red dot
            #     cv2.drawMarker(vis_mask, (px, py), (0, 255, 255),                # yellow cross
            #                 markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)

            # # save the mask visualization with COM
            # cv2.imwrite(os.path.join(vis_dir, fname), vis_mask)
            
            # # Dilate mask
            # kernel = np.ones((15, 15), np.uint8)
            # # dilated_mask = cv2.dilate(final_mask, kernel, iterations=1)
            
            # # Save mask image
            mask_path = os.path.join(camera_mask_dir, fname)
            final_mask = (final_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, final_mask)
            
            # Apply mask to image (set masked areas to black)
            result_img = image_rgb.copy()
            result_img[final_mask > 0] = 0

            # Save masked image
            out_path = os.path.join(camera_masked_dir, fname)
            cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print("Processing complete!")
    print(f"Results saved to: {output_base_dir}")


if __name__ == "__main__":
    main()