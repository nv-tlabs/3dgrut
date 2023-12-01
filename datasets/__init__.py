import torch

from datasets.nerf_dataset import NeRFDataset
from datasets.ngp_dataset import NGPDataset
from datasets.colmap_dataset import ColmapDataset
from datasets.ncore_dataset import NCoreDataset
from datasets.ncore_utils import Batch as NCoreBatch
from datasets.utils import PointCloud

def make(name:str, config, ray_jitter):
    train_collate_fn = None
    val_collate_fn = None

    match name:
        case 'nerf':
            train_dataset = NeRFDataset(
                config.path, 
                split='train', 
                sample_full_image=config.dataset.train.sample_full_image, 
                batch_size=config.dataset.train.batch_size,
                return_alphas=True,
                ray_jitter=ray_jitter
            )
            val_dataset = NeRFDataset(
                config.path,
                split='test', # TODO : change back to val, but ww can directly monitor what we will get :)
                sample_full_image=True,
                return_alphas=False
            )
        case 'colmap':
            train_dataset = ColmapDataset(
                config.path, 
                split='train', 
                sample_full_image=config.dataset.train.sample_full_image, 
                batch_size=config.dataset.train.batch_size,
                downsample_factor=config.dataset.downsample_factor,
                ray_jitter=ray_jitter
            )
            val_dataset = ColmapDataset(
                config.path,
                split='val',
                sample_full_image=True,
                downsample_factor=config.dataset.downsample_factor
            )
        case 'ngp':
            train_dataset = NGPDataset(
                config.path, 
                split='train', 
                sample_full_image=config.dataset.train.sample_full_image, 
                batch_size=config.dataset.train.batch_size,
                batch_size_lidar=config.dataset.train.batch_size // 4 if config.loss.use_lidardistance else 0,
                use_lidar=True,
                use_dynamic_masks=~config.dataset.train.sample_full_image,
                use_aux=config.dataset.get("use_aux_data", False)
        )
            val_dataset = NGPDataset(
                config.path,
                split='val',
                sample_full_image=True,
                val_downsample=5,
                val_frame_subsample=5,
                use_aux=config.dataset.get("use_aux_data", False)
            )
        case 'ncore':
            # TODO: add all of the dataset parameters to config
            duration_sec = 2.0
            n_train_sample_timepoints = 5

            train_dataset = NCoreDataset(
                config.path, 
                split='train', 
                duration_sec=duration_sec,
                n_train_sample_timepoints=n_train_sample_timepoints,
                n_train_sample_lidar_rays=1024 if config.loss.use_lidardistance else 0
            )
            val_dataset = NCoreDataset(
                config.path,
                split='val',
                duration_sec=duration_sec,
            )
            # Dataset produces NCoreBatch requiring dedicated collate_fns
            train_collate_fn = NCoreBatch.collate_fn
            val_collate_fn = NCoreBatch.collate_fn
        case _:
            raise ValueError(f'Unsupported dataset type: {config.dataset.type}. Choose between: ["colmap", "nerf", "ngp", "ncore"]. ')

    return train_dataset, val_dataset, train_collate_fn, val_collate_fn