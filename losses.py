"""
Reconstruction loss for Stage 1 (MAE pre-training), Eq. (2) in the paper:
mean squared error computed only over the masked patches.
"""

import torch
import torch.nn.functional as F


def compute_image_mse_on_mask(patch_preds: torch.Tensor, orig_patches: torch.Tensor, mae_mask: torch.Tensor) -> torch.Tensor:
    device = patch_preds.device
    mae_mask = mae_mask.to(device)
    B, N, C, p, _ = patch_preds.shape
    preds = patch_preds.view(B, N, -1)
    targets = orig_patches.view(B, N, -1).to(device)

    if torch.isnan(preds).any() or torch.isinf(preds).any():
        raise ValueError("Decoder output contains NaN/Inf values -- training is unstable.")
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        raise ValueError("Target patches contain NaN/Inf values.")

    losses = []
    for b in range(B):
        idx = torch.nonzero(mae_mask[b]).squeeze(-1)
        if idx.numel() == 0:
            continue
        losses.append(F.mse_loss(preds[b, idx, :], targets[b, idx, :], reduction="mean"))

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
