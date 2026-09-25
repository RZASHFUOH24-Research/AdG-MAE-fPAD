"""
Stage 1: Self-supervised MAE pre-training for AdG-MAE (Section III.D, Fig. 1).

Edit `config.py` to change paths and hyper-parameters, then simply run:
    python pretrain.py

Supports automatic resume: if a checkpoint already exists in
config.PRETRAIN_SAVE_DIR, training continues from the last completed epoch.
"""

import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset_pretrain import FASPretrainDataset
from models import RETinyViTEncoder, ImageDecoderMAE
from losses import compute_image_mse_on_mask
from utils import collate_fn, find_latest_checkpoint, image_to_patches, patches_to_image


def build_optimizer(encoder, decoder, lr, weight_decay):
    """
    Separates parameters into three groups so that biases, LayerNorm weights,
    positional/mask-token embeddings, and (crucially) the AGS gate `alpha`
    are excluded from weight decay -- letting alpha evolve freely, as
    required by the Adaptive Gated Self-Attention design (Section III.C).
    """
    decay, no_decay, gate_params = [], [], []
    for module in (encoder, decoder):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if name.endswith("alpha"):
                gate_params.append(p)
            elif p.ndim < 2 or "pos_embed" in name or "mask_token" in name:
                no_decay.append(p)
            else:
                decay.append(p)

    print(f"Gate parameters (no weight decay): {sum(p.numel() for p in gate_params)}")

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
            {"params": gate_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def pretrain_epoch(encoder, decoder, optimizer, dataloader, device, epoch):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    for step, batch in enumerate(pbar):
        img_orig = batch["img_orig"].to(device)
        img_masked = batch["img_masked"].to(device)
        mae_mask = batch["mae_mask"].to(device)

        enc_out, vis_inds = encoder(img_masked, mae_mask=mae_mask)
        patch_preds = decoder(enc_out, vis_inds)

        B = img_orig.shape[0]
        orig_patches = torch.stack(
            [image_to_patches(img_orig[b].cpu(), config.PATCH_SIZE) for b in range(B)], dim=0
        ).to(device)

        loss = compute_image_mse_on_mask(patch_preds, orig_patches, mae_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.5f}", avg=f"{total_loss / (step + 1):.5f}")

    return total_loss / len(dataloader)


def save_sample_visualisation(encoder, decoder, sample_batch, device, epoch, save_path):
    img_orig = sample_batch["img_orig"].to(device)
    img_masked = sample_batch["img_masked"].to(device)
    mae_mask = sample_batch["mae_mask"].to(device)

    with torch.no_grad():
        enc_out, vis_inds = encoder(img_masked, mae_mask=mae_mask)
        patch_preds = decoder(enc_out, vis_inds)

    idx = 0
    orig_display = img_orig[idx].cpu().permute(1, 2, 0).numpy()
    masked_display = img_masked[idx].cpu().permute(1, 2, 0).numpy()

    orig_patches = image_to_patches(img_orig[idx].cpu(), config.PATCH_SIZE)
    full_reconstructed = orig_patches.clone()
    mask_single = mae_mask[idx].cpu()
    predicted_masked = patch_preds[idx][mask_single].cpu()
    if predicted_masked.numel() > 0:
        full_reconstructed[mask_single] = predicted_masked

    recon_display = patches_to_image(full_reconstructed, config.PATCH_SIZE, config.IMG_SIZE, config.IMG_SIZE)
    recon_display = recon_display.permute(1, 2, 0).numpy().clip(0, 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1); plt.imshow(orig_display); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.imshow(masked_display); plt.title("Masked (75%)"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.imshow(recon_display); plt.title("Reconstructed"); plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    os.makedirs(config.PRETRAIN_SAVE_DIR, exist_ok=True)
    device = config.DEVICE
    print(f"Running on: {device}")

    # ---- Data ----
    dataset = FASPretrainDataset(
        roots=config.PRETRAIN_DATA_ROOTS,
        img_size=config.IMG_SIZE,
        patch_size=config.PATCH_SIZE,
        mask_ratio=config.MASK_RATIO,
    )
    loader = DataLoader(
        dataset, batch_size=config.PRETRAIN_BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn,
    )
    print(f"Total pretraining samples: {len(dataset)}")

    # ---- Model ----
    encoder = RETinyViTEncoder(
        img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, in_ch=3,
        embed_dim=config.ENCODER_EMBED_DIM, depth=config.ENCODER_DEPTH, num_heads=config.ENCODER_HEADS,
    ).to(device)
    decoder = ImageDecoderMAE(
        num_patches=encoder.patch_embed.num_patches,
        encoder_embed_dim=config.ENCODER_EMBED_DIM,
        decoder_embed_dim=config.DECODER_EMBED_DIM,
        patch_size=config.PATCH_SIZE, in_ch=3,
        decoder_depth=config.DECODER_DEPTH, decoder_heads=config.DECODER_HEADS,
    ).to(device)

    optimizer = build_optimizer(encoder, decoder, config.PRETRAIN_LR, config.PRETRAIN_WEIGHT_DECAY)

    # ---- Resume support ----
    start_epoch = 1
    resume_path = find_latest_checkpoint(config.PRETRAIN_SAVE_DIR)
    if resume_path is not None:
        print(f"Found existing checkpoint, resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming training at epoch {start_epoch}")
    else:
        print("No existing checkpoint found, starting from epoch 1.")

    # ---- Training loop ----
    for epoch in range(start_epoch, config.PRETRAIN_EPOCHS + 1):
        try:
            avg_loss = pretrain_epoch(encoder, decoder, optimizer, loader, device, epoch)
            print(f"Epoch {epoch} avg_loss={avg_loss:.5f}")

            sample_batch = next(iter(loader))
            vis_path = os.path.join(config.PRETRAIN_SAVE_DIR, f"epoch_{epoch}_sample.png")
            save_sample_visualisation(encoder, decoder, sample_batch, device, epoch, vis_path)

            ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(ckpt, os.path.join(config.PRETRAIN_SAVE_DIR, f"pretrain_epoch{epoch}.pth"))
            torch.save(ckpt, os.path.join(config.PRETRAIN_SAVE_DIR, "latest.pth"))

        except KeyboardInterrupt:
            print(f"\nInterrupted during/after epoch {epoch}. Re-run this script to resume automatically.")
            break
        except Exception as e:
            print(f"\nTraining stopped by an error during epoch {epoch}: {e}")
            print("The last completed epoch's checkpoint is still intact; fix the issue and re-run to resume.")
            raise
    else:
        print("Pre-training finished.")


if __name__ == "__main__":
    main()
