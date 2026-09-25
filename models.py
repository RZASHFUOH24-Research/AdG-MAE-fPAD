"""
AdG-MAE model components.

- PatchEmbed:               patch embedding + learnable positional embedding.
- GatedSelfAttentionBlock:  Adaptive Gated Self-Attention (AGS), Section III.C.
                            X_attn = X_in + alpha * MHSA(LN(X_in)), with a
                            learnable scalar alpha initialised near zero.
- RETinyViTEncoder:         the shared TinyViT-style encoder used in BOTH
                            stages (Stage 1 pre-training with masking, and
                            Stage 2 fine-tuning without masking), matching the
                            paper's description that the same Gated ViT
                            encoder is carried over between stages.
- ImageDecoderMAE:          lightweight MAE reconstruction decoder (Stage 1
                            only, discarded before fine-tuning). Uses a
                            decoder-specific embedding width (Appendix
                            "Ablation on Decoder Architecture": Depth=8,
                            Width=512 is the configuration adopted in the
                            paper).
- FASClassifier:            linear classification head appended to the
                            encoder during Stage 2 fine-tuning.
"""

from typing import List, Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid[0] * self.grid[1]
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        return x


class GatedSelfAttentionBlock(nn.Module):
    """
    Adaptive Gated Self-Attention (AGS), Eq. (1) in the paper.
    `alpha` is a learnable scalar, initialised near zero (structural
    warm-up), that gates the contribution of the attention branch and is
    excluded from weight decay so it can grow freely (even beyond 1.0).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, alpha_init=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.alpha * self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class RETinyViTEncoder(nn.Module):
    """
    Gated TinyViT encoder shared between pre-training and fine-tuning.

    - forward(x)                  -> full-sequence mode, used for classification
                                      in Stage 2 (no masking).
    - forward(x, mae_mask=mask)   -> masked mode, used for MAE pre-training in
                                      Stage 1. Returns (visible_tokens,
                                      visible_indices) so the decoder can
                                      re-insert mask tokens at the correct
                                      positions.
    """

    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=192, depth=8, num_heads=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        self.blocks = nn.ModuleList(
            [GatedSelfAttentionBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mae_mask: torch.Tensor = None):
        full_tokens = self.patch_embed(x)

        if mae_mask is None:
            # Stage 2: process the full patch sequence (no masking).
            out = full_tokens
            for blk in self.blocks:
                out = blk(out)
            return self.norm(out)

        # Stage 1: keep only the visible (unmasked) patches.
        visible_tokens_list: List[torch.Tensor] = []
        visible_indices_list: List[torch.Tensor] = []
        for b in range(full_tokens.shape[0]):
            idx = torch.nonzero(~mae_mask[b]).squeeze(-1)
            visible_indices_list.append(idx)
            visible_tokens_list.append(full_tokens[b, idx, :])

        x = torch.stack(visible_tokens_list, dim=0)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, visible_indices_list


class ImageDecoderMAE(nn.Module):
    """
    Single-stream, lightweight MAE decoder (Section III.D.2). Projects the
    encoder's visible-token embeddings into a (typically wider) decoder
    embedding space, re-inserts learnable mask tokens at the masked
    positions, adds decoder positional embeddings, and reconstructs pixel
    values for every patch via a stack of standard Transformer encoder
    layers (self-attention over the full, reassembled sequence).
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int,
        decoder_embed_dim: int = 512,
        patch_size: int = 16,
        in_ch: int = 3,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.patch_dim = in_ch * patch_size * patch_size

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_heads, batch_first=True
        )
        self.decoder_blocks = nn.TransformerEncoder(layer, num_layers=decoder_depth)
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.to_patch = nn.Linear(decoder_embed_dim, self.patch_dim)

    def forward(self, enc_vis_tokens: torch.Tensor, vis_inds: List[torch.Tensor]):
        enc_vis_tokens = self.decoder_embed(enc_vis_tokens)
        B, _, D = enc_vis_tokens.shape
        N = self.num_patches

        decoder_input = self.mask_token.expand(B, N, D).clone()
        for b in range(B):
            idx = vis_inds[b]
            decoder_input[b, idx, :] = enc_vis_tokens[b, : len(idx), :]

        x = decoder_input + self.decoder_pos_embed
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        patch_preds = self.to_patch(x).view(B, N, -1, self.patch_size, self.patch_size)
        return patch_preds


class FASClassifier(nn.Module):
    """Linear classification head appended to the fine-tuned Gated ViT encoder."""

    def __init__(self, encoder: RETinyViTEncoder, num_classes: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x):
        tokens = self.encoder(x)          # [B, N, D], full sequence (no masking)
        pooled = tokens.mean(dim=1)       # global average pooling over patch tokens
        return self.classifier(self.dropout(pooled))
