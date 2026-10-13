import os
import torch
import numpy as np
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

_PIN = torch.cuda.is_available()   # pin_memory only helps (and only warns off) on GPU


def train_dataloader(path, batch_size=64, num_workers=0, ps=256):
    image_dir = os.path.join(path, 'train')

    dataloader = DataLoader(
        DeblurDataset(image_dir, ps=ps),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_PIN,
        drop_last=True,                          # constant batch -> stable BN
        persistent_workers=num_workers > 0,      # 1000 epochs: don't respawn workers
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0, center_crop=None, split='test'):
    image_dir = os.path.join(path, split)
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True, center_crop=center_crop),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_PIN
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0, center_crop=None, split='test'):
    """split='test' reproduces the paper (Haze4K has no val split, so the paper
    selects checkpoints on test). Use split='val' when a real val split exists
    (HazyDet) so the test numbers stay untouched by model selection."""
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, split), is_valid=True, center_crop=center_crop),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader

import random
class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, is_valid=False, ps=None,
                 center_crop=None):
        """
        ps          : random-crop size for training (None = full image)
        center_crop : deterministic centre crop for valid/test (None = full image).
                      Full 1280x720 validation needs ~7 GB RAM on CPU; use e.g. 256
                      when debugging without a GPU.
        """
        self.image_dir = image_dir
        # skip macOS resource forks ("._x.jpg") that ride along inside zips
        self.image_list = [f for f in os.listdir(os.path.join(image_dir, 'hazy/'))
                           if not f.startswith('._')]
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.is_valid = is_valid
        self.ps = ps
        self.center_crop = center_crop
    
    def __len__(self):
        return len(self.image_list)

    def _gt_path(self, hazy_name):
        """gt = hazy name without extension, up to the first '_'; .png or .jpg.
        Haze4K : 1_0.89_1.8.png -> 1.png      HazyDet : 000002.jpg -> 000002.jpg"""
        stem = os.path.splitext(hazy_name)[0].split('_')[0]
        for ext in ('.png', '.jpg', '.jpeg'):
            p = os.path.join(self.image_dir, 'gt', stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"no gt for {hazy_name} (looked for {stem}.png/.jpg)")

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx])).convert('RGB')
        label = Image.open(self._gt_path(self.image_list[idx])).convert('RGB')
        ps = self.ps
    
        file_name = os.path.splitext(self.image_list[idx])[0]
        depth_image_path = os.path.join(self.image_dir, 'depth2l', file_name + '.png')
        depth = Image.open(depth_image_path).convert("L")

        if self.ps is not None:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            depth = F.to_tensor(depth)

            hh, ww = label.shape[1], label.shape[2]

            rr = random.randint(0, hh-ps)
            cc = random.randint(0, ww-ps)
            
            image = image[:, rr:rr+ps, cc:cc+ps]
            label = label[:, rr:rr+ps, cc:cc+ps]
            depth = depth[:, rr:rr+ps, cc:cc+ps]

            if random.random() < 0.5:
                image = image.flip(2)
                label = label.flip(2)
                depth = depth.flip(2)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
            depth = F.to_tensor(depth)
            if self.center_crop is not None:
                c = self.center_crop
                hh, ww = label.shape[1], label.shape[2]
                r0, c0 = max(0, (hh - c) // 2), max(0, (ww - c) // 2)
                image = image[:, r0:r0 + c, c0:c0 + c]
                label = label[:, r0:r0 + c, c0:c0 + c]
                depth = depth[:, r0:r0 + c, c0:c0 + c]

        image = torch.cat([image, depth], dim=0)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError