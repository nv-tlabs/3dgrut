import argparse
import torch
import torchvision
import os
import numpy as np

from tqdm import tqdm
from model import MixtureOfGaussians
from datasets.utils import move_to_gpu

from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from datasets.ngp_dataset import NGPDataset
from datasets.ncore_dataset import NCoreDataset

from torchmetrics import PeakSignalNoiseRatio

BACKGROUND_COLOR = torch.zeros((3,), dtype=torch.float32, device='cuda')


class Renderer:
    def __init__(self, checkpoint_path, out_dir, path="", save_gt=True, writer=None) -> None:
        self.out_dir = out_dir
        self.path = path
        self.save_gt = save_gt


        self.checkpoint = torch.load(checkpoint_path)
        self.conf = self.checkpoint["config"]
        self.global_step = self.checkpoint['global_step']

        # Replace the path to the test data
        if self.path: 
            self.conf.path = path

        val_collate_fn = None
        # Create the dataset
        if self.conf.dataset.type == 'nerf':
            dataset = NeRFDataset(
                self.conf.path, 
                split='test', 
                sample_full_image=True, 
                batch_size=1,
                return_alphas=True
            )
        else:
            raise ValueError(f'Unsupported dataset type: {self.conf.dataset.type}. Choose between: ["colmap", "nerf", "ngp", "ncore"]. ')
        
        self.dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False, collate_fn=val_collate_fn)

        # Initialize the model and the optix context
        self.model = MixtureOfGaussians(self.conf)
        self.model.set_optix_context()


        # Initialize the parameters from checkpoint
        self.model.init_from_checkpoint(self.checkpoint)
        self.model.build_bvh()

        self.writer = writer
        
    def render_all(self):
        # Criterions that we log during training
        criterions = {"psnr":  PeakSignalNoiseRatio(data_range=1).to("cuda")}

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)
        
        psnr = []
        iteration = 0
        with tqdm(self.dataloader) as pbar:
            for batch in pbar:
                with torch.no_grad():
                    gpu_batch = move_to_gpu(batch)
                    rays_ori, rays_dir, rgb_gt = gpu_batch["rays_ori"], gpu_batch["rays_dir"], gpu_batch["rgb_gt"]
                    # Compute the outputs of a single batch
                    outputs = self.model(rays_ori, rays_dir)
                    
                    # The values are already alpha composited with the background
                    rgb_pred = outputs['pred_rgb']
                    torchvision.utils.save_image(rgb_pred.squeeze(0).permute(2,0,1), os.path.join(output_path_renders, '{0:05d}'.format(iteration) + ".png"))
                    if self.writer is not None:
                        self.writer.add_image('image/test', outputs['pred_rgb'][-1].clip(0,1.0), self.global_step, dataformats='HWC')


                    if self.save_gt:
                        if "alpha" in gpu_batch: 
                            rgb_gt = rgb_gt * gpu_batch["alpha"] + BACKGROUND_COLOR * (1 - gpu_batch["alpha"])
                        torchvision.utils.save_image(rgb_gt.squeeze(0).permute(2,0,1), os.path.join(output_path_gt, '{0:05d}'.format(iteration) + ".png"))

                    # Compute the loss
                    psnr.append(criterions["psnr"](rgb_pred, rgb_gt).item())

                    pbar.set_postfix({'iteration': iteration, 'psnr': psnr[-1]})
                    iteration += 1

            mean_psnr = np.mean(psnr)
            std_psnr = np.std(psnr)

            print(f"PSNR : {mean_psnr}, std: f{std_psnr}")
            
            if self.writer is not None:
                self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            return mean_psnr, std_psnr

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument("--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt")
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]")
    args = parser.parse_args()

    renderer = Renderer(checkpoint_path=args.checkpoint,
                        path=args.path,
                        out_dir=args.out_dir,
                        save_gt=args.save_gt)

    renderer.render_all()