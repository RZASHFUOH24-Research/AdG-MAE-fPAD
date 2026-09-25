"""
Dataset and augmentations for Stage 2: supervised fine-tuning.

- MultiFolderFASDataset: scans one or more root folders recursively and infers
  the live/spoof label from filename/path keywords.
- SpectralReweightingAugmentation (SRA): implements Eq. (3)-(4) of the paper --
  a continuous radial re-weighting of the 2D FFT magnitude that suppresses
  low-frequency domain identifiers (omega_low < 1.0) and amplifies
  high-frequency spoofing texture (omega_high > 1.0).
- RandomDownscale: optional auxiliary augmentation (down-/up-sampling) that
  simulates sensor/resolution variation across capture devices; not part of
  SRA itself and disabled by default to match the paper's reported transform
  pipeline, but kept here for experimentation.
"""

import math
import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset


class RandomDownscale(nn.Module):
    """Randomly down- then up-samples a PIL image to simulate resolution/sensor variation."""

    def __init__(self, p=0.5, scale_range=(0.25, 0.75)):
        super().__init__()
        self.p = p
        self.scale_range = scale_range

    def forward(self, img):
        if torch.rand(1) < self.p:
            w, h = img.size
            scale = random.uniform(*self.scale_range)
            new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
            img = img.resize((new_w, new_h), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        return img


class SpectralReweightingAugmentation(nn.Module):
    """
    Spectral Re-weighting Augmentation (SRA), Eq. (3)-(4) in the paper.

    Given an image tensor x, computes its 2D FFT, applies a radial weight
    matrix M (linearly interpolating from `low_w` at the DC component to
    `high_w` at the corner/highest frequency), and reconstructs the image via
    the inverse FFT:
        x' = IFFT( M (.) FFT(x) )
    """

    def __init__(self, p: float = 0.5, low_w: float = 0.75, high_w: float = 1.5):
        super().__init__()
        self.p = p
        self.low_w = low_w
        self.high_w = high_w

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > self.p:
            return img_tensor

        C, H, W = img_tensor.shape
        fft = torch.fft.fft2(img_tensor)
        fshift = torch.fft.fftshift(fft)

        y = torch.arange(H, device=img_tensor.device) - H // 2
        x = torch.arange(W, device=img_tensor.device) - W // 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        dist = torch.sqrt(xx.float() ** 2 + yy.float() ** 2)
        max_dist = math.sqrt((H // 2) ** 2 + (W // 2) ** 2)

        weights = self.low_w + (self.high_w - self.low_w) * (dist / max_dist)
        weights[H // 2, W // 2] = 1.0  # keep the DC (mean intensity) component untouched

        out = torch.fft.ifft2(torch.fft.ifftshift(fshift * weights.unsqueeze(0)))
        return torch.clamp(torch.abs(out), 0, 1)


# Backwards-compatible alias matching the name used in the original script.
FrequencyBoost = SpectralReweightingAugmentation


class MultiFolderFASDataset(Dataset):
    """
    Scans `root_paths` recursively for .jpg/.png files and infers a binary
    label from path keywords: 1 (live) if the path contains "real"/"live"/
    "true", 0 (spoof) if it contains "spoof"/"attack"/"fake". Files matching
    neither pattern are skipped.
    """

    def __init__(self, root_paths, transform=None):
        self.samples = []
        self.transform = transform
        print(f"Scanning {len(root_paths)} root(s)...")

        for root in root_paths:
            for p in glob(os.path.join(root, "**", "*.*"), recursive=True):
                if p.lower().endswith((".jpg", ".jpeg", ".png")):
                    lbl = (
                        1 if any(k in p.lower() for k in ["real", "live", "true"]) else
                        0 if any(k in p.lower() for k in ["spoof", "attack", "fake"]) else
                        None
                    )
                    if lbl is not None:
                        self.samples.append((p, lbl))

        np.random.shuffle(self.samples)
        print(f"  -> Found {len(self.samples)} labelled images.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return torch.zeros((3, 256, 256)), label

        return (self.transform(img) if self.transform else img), label
