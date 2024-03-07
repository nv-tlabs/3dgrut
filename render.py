import argparse
import torch
import torchvision
import os
import numpy as np

from tqdm import tqdm
from models.model import MixtureOfGaussians
from datasets.utils import move_to_gpu

from datasets.colmap_dataset import ColmapDataset
from datasets.nerf_dataset import NeRFDataset
from datasets.ngp_dataset import NGPDataset
from datasets.ncore_dataset import NCoreDataset
from datasets.ncore_utils import Batch as NCoreBatch

from torchmetrics import PeakSignalNoiseRatio


class Renderer:
    def __init__(self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None) -> None:
        self.model = model

        self.out_dir = out_dir
        self.path = path
        self.save_gt = save_gt
        # Replace the path to the test data
        if path:
            conf.path = path
        self.conf = conf
        self.global_step = global_step

        self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device='cuda')
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device='cuda')
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        val_collate_fn = None
        # Create the dataset
        match conf.dataset.type:
            case 'nerf':
                dataset = NeRFDataset(
                    conf.path,
                    split='test',
                    sample_full_image=True,
                    batch_size=1,
                    return_alphas=True, 
                    # bg_color=conf.model.background.color
                )
            case 'colmap':
                dataset = ColmapDataset(
                    conf.path,
                    split='val',
                    sample_full_image=True,
                    downsample_factor=conf.dataset.downsample_factor
                )
            case 'ngp':
                dataset = NGPDataset(
                    conf.path,
                    split='val',
                    sample_full_image=True,
                    val_downsample=5,
                    val_frame_subsample=5,
                    use_aux=conf.dataset.get("use_aux_data", False)
                )
            case 'ncore':
                # TODO: add all of the dataset parameters to config
                duration_sec = 2.0
                dataset = NCoreDataset(
                    conf.path,
                    split='val',
                    duration_sec=duration_sec,
                )
                # Dataset produces NCoreBatch requiring dedicated collate_fns
                val_collate_fn = NCoreBatch.collate_fn

            case _:
                raise ValueError(
                    f'Unsupported dataset type: {conf.dataset.type}. Choose between: ["colmap", "nerf", "ngp", "ncore"]. ')

        dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False,
                                                      collate_fn=val_collate_fn)
        return dataloader

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None):
        """ Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path)
        conf = checkpoint["config"]
        global_step = checkpoint['global_step']

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            model.set_optix_context()

            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_bvh()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer
        )

    @classmethod
    def from_preloaded_model(cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None):

        conf = model.conf
        if global_step is None:
            global_step = ''
        model.build_bvh()
        return Renderer(model=model,
                        conf=conf,
                        global_step=global_step,
                        out_dir=out_dir,
                        path=path,
                        save_gt=save_gt,
                        writer=writer)


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


                    if "alpha" in gpu_batch:
                        rgb_gt = rgb_gt * gpu_batch["alpha"] + self.bg_color * (1 - gpu_batch["alpha"])
                    if self.save_gt:
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

    renderer = Renderer.from_checkpoint(
                        checkpoint_path=args.checkpoint,
                        path=args.path,
                        out_dir=args.out_dir,
                        save_gt=args.save_gt)

    renderer.render_all()
