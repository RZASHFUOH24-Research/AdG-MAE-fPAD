"""
Stage 2: Supervised fine-tuning for AdG-MAE (Section III.E, Fig. 3).

Loads the Stage-1 pre-trained Gated ViT encoder, attaches a linear
classification head, and fine-tunes with:
  - a short warm-up (encoder frozen, AdamW, head only),
  - followed by full fine-tuning with SAM, differential learning rates, and
    Spectral Re-weighting Augmentation (SRA).

Edit `config.py` to change paths and hyper-parameters, then simply run:
    python finetune.py
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from dataset_finetune import MultiFolderFASDataset, SpectralReweightingAugmentation
from models import RETinyViTEncoder, FASClassifier
from sam import SAM
from evaluate import evaluate_model


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.ToTensor(),
            SpectralReweightingAugmentation(
                p=config.SRA_PROB, low_w=config.SRA_OMEGA_LOW, high_w=config.SRA_OMEGA_HIGH
            ),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return train_tf, eval_tf


def main():
    device = config.DEVICE
    print(f"Running on: {device}")
    os.makedirs(config.FT_SAVE_DIR, exist_ok=True)

    train_tf, eval_tf = build_transforms()

    train_ds = MultiFolderFASDataset(config.FT_TRAIN_ROOTS, train_tf)
    val_ds = MultiFolderFASDataset(config.FT_VAL_ROOTS, eval_tf)
    test_ds = MultiFolderFASDataset(config.FT_TEST_ROOTS, eval_tf)

    train_loader = DataLoader(train_ds, config.FT_BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, config.FT_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, config.FT_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---- Model: load Stage-1 pre-trained encoder ----
    encoder = RETinyViTEncoder(
        img_size=config.IMG_SIZE, patch_size=config.PATCH_SIZE, in_ch=3,
        embed_dim=config.ENCODER_EMBED_DIM, depth=config.ENCODER_DEPTH, num_heads=config.ENCODER_HEADS,
    )
    ckpt = torch.load(config.PRETRAINED_CKPT, map_location=device)
    encoder.load_state_dict(ckpt["encoder"] if "encoder" in ckpt else ckpt, strict=False)

    model = FASClassifier(encoder, num_classes=2, dropout_rate=config.DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # ---- Warm-up phase: freeze encoder, train head only (AdamW) ----
    is_frozen = True
    for p in model.encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR_HEAD, weight_decay=config.FT_WEIGHT_DECAY,
    )

    print(f">>> Warm-up phase (AdamW, {config.WARMUP_EPOCHS} epochs): encoder frozen")
    print("Starting training...")

    for epoch in range(1, config.FT_EPOCHS + 1):

        # ---- Switch to SAM once warm-up ends ----
        if epoch == config.WARMUP_EPOCHS + 1 and is_frozen:
            print("\n>>> Unfreezing encoder & switching to SAM (differential LR)...")
            is_frozen = False
            for p in model.encoder.parameters():
                p.requires_grad = True

            optimizer = SAM(
                [
                    {"params": model.encoder.parameters(), "lr": config.LR_BACKBONE},
                    {"params": model.classifier.parameters(), "lr": config.LR_HEAD},
                ],
                torch.optim.AdamW,
                rho=config.SAM_RHO,
                weight_decay=config.FT_WEIGHT_DECAY,
            )

        model.train()
        train_loss = 0.0

        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)

            if is_frozen:
                optimizer.zero_grad()
                loss = criterion(model(img), lbl)
                loss.backward()
                optimizer.step()
            else:
                # SAM's required two-step update.
                optimizer.zero_grad()
                loss = criterion(model(img), lbl)
                loss.backward()
                optimizer.first_step(zero_grad=True)

                criterion(model(img), lbl).backward()
                optimizer.second_step(zero_grad=True)

            train_loss += loss.item()

        mode = "Frozen (warm-up)" if is_frozen else "SAM"
        print(f"\nEp {epoch}/{config.FT_EPOCHS} | Train Loss: {train_loss / len(train_loader):.4f} | Mode: {mode}")

        print("-" * 50)
        evaluate_model(model, val_loader, device, criterion=None, set_name="VAL")
        evaluate_model(model, test_loader, device, criterion=None, set_name="TEST")
        print("-" * 50)

        torch.save(model.state_dict(), os.path.join(config.FT_SAVE_DIR, f"model_epoch{epoch}.pth"))

    print("\nFine-tuning finished.")


if __name__ == "__main__":
    main()
