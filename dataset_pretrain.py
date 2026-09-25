"""
Dataset for Stage 1: self-supervised MAE pre-training.
Loads unlabeled face images from a union of source-domain roots, each
expected to contain a "Real" and a "Spoof" sub-folder, and applies random
spatial patch masking (Section III.C / Fig. 1 of the paper).
"""

import os
import random
from glob import glob
from typing import List

import torch
from torch.utils.data import Dataset

from utils import open_image_as_tensor, image_to_patches, patches_to_image


class FASPretrainDataset(Dataset):
    def __init__(self, roots: List[str], img_size: int = 256, patch_size: int = 16, mask_ratio: float = 0.75):
        super().__init__()
        self.paths = []
        for r in roots:
            for cls in ("Real", "Spoof"):
                p = os.path.join(r, cls)
                if os.path.isdir(p):
                    self.paths.extend(glob(os.path.join(p, "*")))

        if not self.paths:
            raise FileNotFoundError(
                "No images found under the given PRETRAIN_DATA_ROOTS. "
                "Each root must contain 'Real' and/or 'Spoof' sub-folders."
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img_orig = open_image_as_tensor(path, self.img_size)

        # Random spatial masking (75% of patches removed), Section IV.C.
        m = max(1, int(round(self.num_patches * self.mask_ratio)))
        masked_idx = set(random.sample(range(self.num_patches), m))
        mae_mask = torch.tensor([i in masked_idx for i in range(self.num_patches)], dtype=torch.bool)

        patches_orig = image_to_patches(img_orig, self.patch_size)
        masked_patches = torch.stack(
            [torch.zeros_like(patches_orig[i]) if mae_mask[i] else patches_orig[i] for i in range(self.num_patches)],
            dim=0,
        )
        masked_img = patches_to_image(masked_patches, self.patch_size, self.img_size, self.img_size)

        return {"img_orig": img_orig, "img_masked": masked_img, "mae_mask": mae_mask}
