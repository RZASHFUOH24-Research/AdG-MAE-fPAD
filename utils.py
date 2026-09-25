"""
Shared utilities: patch <-> image conversion, image loading, resume-checkpoint
discovery, and the DataLoader collate function used by the pretraining set.
"""

import os
from glob import glob

import numpy as np
import torch
from PIL import Image


def open_image_as_tensor(path: str, img_size: int = 256) -> torch.FloatTensor:
    """Load an RGB image from disk, resize it, and return a CHW float tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def image_to_patches(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    """[C, H, W] -> [N, C, patch_size, patch_size]."""
    C, H, W = img.shape
    patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, C, patch_size, patch_size)


def patches_to_image(patches: torch.Tensor, patch_size: int, H: int, W: int) -> torch.Tensor:
    """[N, C, patch_size, patch_size] -> [C, H, W]."""
    N, C, p, _ = patches.shape
    ph, pw = H // patch_size, W // patch_size
    patches = patches.view(ph, pw, C, p, p).permute(2, 0, 3, 1, 4).contiguous()
    return patches.view(C, H, W)


def collate_fn(batch):
    """Stacks a list of dicts (all sharing the same keys) into a single batched dict."""
    if not batch:
        return {}
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def find_latest_checkpoint(save_folder: str):
    """
    Look inside `save_folder` for a checkpoint to resume pre-training from.
    Prefers the dedicated 'latest.pth' (saved every epoch, includes optimizer
    state), and falls back to the highest-numbered per-epoch checkpoint file.
    Returns the path, or None if nothing is found.
    """
    latest_path = os.path.join(save_folder, "latest.pth")
    if os.path.isfile(latest_path):
        return latest_path

    candidates = glob(os.path.join(save_folder, "pretrain_epoch*.pth"))
    if not candidates:
        return None

    def _epoch_num(p):
        base = os.path.basename(p)
        try:
            return int(base.replace("pretrain_epoch", "").replace(".pth", ""))
        except ValueError:
            return -1

    candidates.sort(key=_epoch_num)
    return candidates[-1] if _epoch_num(candidates[-1]) >= 0 else None
