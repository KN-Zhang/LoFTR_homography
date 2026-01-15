import torch
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed
from torchvision import transforms

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader
)

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from datasets.homography_dataset_large_size import HomographyDataset, RandomGaussianBlur
from src.datasets.sampler import RandomConcatSampler


def collate_fn(batch):
    filtered_batch = [sample for sample in batch if sample['source_points'] is not None]
    return torch.utils.data.dataloader.default_collate(filtered_batch)

class MultiSceneDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config, num_samples):
        super().__init__()

        # self.dataset = config.DATASET.DATASET
        # self.train_load_resolution = config.DATASET.TRAIN_LOAD_RESOLUTION
        # self.train_pairs = config.DATASET.TRAIN_PAIRS
        self.df = config.DATASET.MGDPT_DF
        self.coarse_down_scale = 8 #config.DATASET.COARSE_DOWN_SCALE
        self.args = args
        # self.val_load_resolution = config.DATASET.VAL_LOAD_RESOLUTION
        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'num_workers':  args.num_workers,
            'drop_last': True,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
        self.val_loader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1, #args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True),
        }
        
        self.num_samples = num_samples
        
        

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        if stage == 'fit':
            if 'glunet' not in self.args.train_dataset:
                self.train_dataset = HomographyDataset(dataset=self.args.train_dataset,
                                                    mode='train',
                                                    input_resolution=(448, 448),
                                                    initial_transforms =transforms.Compose([
                                                            transforms.Resize(size=640, antialias=None),
                                                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                                            RandomGaussianBlur(p=0.5),
                                                            transforms.ToTensor()
                                                            ]),
                                                    bi=True,
                                                    normalize=False,
                                                    deformation_ratio=[0.3],
                                                    input_format='gray',
                                                    )
            else:
                self.train_dataset = HomographyDataset(dataset=self.args.train_dataset,
                                                        mode='train',
                                                        input_resolution=(448, 448),
                                                        initial_transforms =transforms.Compose([
                                                                    transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2), 
                                                                    transforms.RandomGrayscale(p=0.2),
                                                                    RandomGaussianBlur(p=0.5),
                                                                    transforms.ToTensor()
                                                                    ]),
                                                        bi=True,
                                                        normalize=False,
                                                        input_format='gray'
                                                        )
            self.val_dataset = \
            HomographyDataset(
                    dataset=self.args.test_dataset,
                    mode='val',
                    input_resolution=self.args.test_resolution,
                    input_format='gray',
                    )
        elif stage == 'test':
            self.val_dataset = \
            HomographyDataset(
                    dataset=self.args.test_dataset,
                    mode='val',
                    input_resolution=self.args.test_resolution,
                    input_format='gray',
                    )           


    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        mega_sampler = torch.utils.data.RandomSampler(
            self.train_dataset, num_samples=self.num_samples, replacement=False
        )
        dataloader = DataLoader(self.train_dataset, **self.train_loader_params, sampler=mega_sampler)
        return dataloader
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = DataLoader(self.val_dataset, **self.val_loader_params)
        return dataloader
    
    def test_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        dataloader = DataLoader(self.val_dataset, **self.val_loader_params)
        return dataloader    
